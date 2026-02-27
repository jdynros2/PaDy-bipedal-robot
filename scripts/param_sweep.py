#!/usr/bin/env python3
"""param_sweep.py — Automated parameter grid search for PaDy gait tuning.

Launches headless.launch.py for each parameter combination, waits for fall
or timeout, then reads the CSV saved by gait_analyser to extract the 4 metrics.
Results are saved to ~/ros2_ws/data/sweep_TIMESTAMP.csv for use with plot_results.py.

BEFORE RUNNING
--------------
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/src/pady_robot/scripts/param_sweep.py

CONFIGURE YOUR SWEEP (edit section below)
------------------------------------------
"""

import csv
import datetime
import glob
import math
import os
import signal
import subprocess
import sys
import time

import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  SWEEP CONFIGURATION — edit these values
# ══════════════════════════════════════════════════════════════════════════════

# Parameters to sweep (all combinations will be tested)
KICK_TORQUES      = [20.0, 25.0, 30.0, 35.0, 40.0]   # N·m
HIP_PUSH_TORQUES  = [3.0,  5.0,  7.0,  9.0]           # N·m

# Fixed parameters (not swept)
BODY_FORCE        = 20.0    # N
SPAWN_PITCH       = 0.28    # rad
HIP_PUSH_START    = 7.0     # s
HIP_PUSH_STOP     = 12.0    # s

# Trial settings
N_TRIALS          = 3       # repetitions per parameter combo
MAX_TRIAL_TIME    = 55      # seconds before declaring "no fall" (robot walked full run)
KILL_TIMEOUT      = 12      # seconds allowed for graceful shutdown
INTER_TRIAL_PAUSE = 4       # seconds between trials to let Gazebo fully close

# Robot physical constants for Specific Resistance calculation
ROBOT_MASS        = 3.524   # kg (from URDF)
GRAVITY           = 9.81    # m/s²
STRIDE_LENGTH     = 0.30    # m per half-step (approximate, from leg geometry)

# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = os.path.expanduser('~/ros2_ws/data')


def check_environment():
    if 'AMENT_PREFIX_PATH' not in os.environ:
        print("ERROR: ROS2 workspace not sourced.")
        print("Run: source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash")
        sys.exit(1)


def compute_metrics(df: pd.DataFrame, kick_torque: float) -> dict:
    """Extract the 4 performance metrics from a run CSV."""
    metrics = {
        'step_count':          0,
        'fall_time_s':         float('nan'),
        'gait_cv':             float('nan'),
        'symmetry_index':      float('nan'),
        'specific_resistance': float('nan'),
    }

    if df.empty or 'step_count' not in df.columns:
        return metrics

    metrics['step_count'] = int(df['step_count'].max())

    # Fall time
    fallen = df[df['fall_detected'] == 1]
    if not fallen.empty:
        metrics['fall_time_s'] = float(fallen['sim_time_s'].iloc[0])

    # Gait CV — coefficient of variation of gait_period (only valid periods)
    periods = df[df['gait_period_s'] > 0.1]['gait_period_s']
    if len(periods) >= 2:
        metrics['gait_cv'] = float(periods.std() / periods.mean())

    # Symmetry index — mean |hip_symmetry| during active walking
    walking = df[(df['step_count'] > 0) & (df['fall_detected'] == 0)]
    if not walking.empty and 'hip_symmetry_rad' in df.columns:
        metrics['symmetry_index'] = float(walking['hip_symmetry_rad'].abs().mean())

    # Specific Resistance — E / (m·g·d)
    # Energy ≈ total angular displacement of hips × kick_torque (rough proxy)
    if 'hip_right_rad' in df.columns and metrics['step_count'] > 0:
        hip_work = (
            df['hip_right_rad'].diff().abs().sum() +
            df['hip_left_rad'].diff().abs().sum()
        ) * kick_torque
        distance = max(1, metrics['step_count']) * STRIDE_LENGTH
        metrics['specific_resistance'] = float(
            hip_work / (ROBOT_MASS * GRAVITY * distance)
        )

    return metrics


def find_new_csv(before_set: set, timeout: float = 10.0) -> str | None:
    """Wait for a new CSV to appear in DATA_DIR and return its path."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        after = set(glob.glob(os.path.join(DATA_DIR, 'run_*.csv')))
        new = after - before_set
        if new:
            return sorted(new)[-1]   # newest
        time.sleep(0.5)
    return None


def wait_for_fall(csv_path: str, max_time: float) -> bool:
    """Poll CSV for fall_detected==1. Returns True if fall occurred."""
    deadline = time.time() + max_time
    while time.time() < deadline:
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and df['fall_detected'].max() == 1:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def kill_launch(proc):
    """Send SIGINT to the process group, wait, then force-kill if needed."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=KILL_TIMEOUT)
    except subprocess.TimeoutExpired:
        print("  [sweep] graceful shutdown timed out — sending SIGKILL")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    except Exception:
        pass


def run_trial(kick_torque: float, hip_push_torque: float,
              trial_num: int, total: int) -> dict | None:
    """Launch one trial, wait for fall, return metrics dict or None on failure."""
    label = f"kt={kick_torque:.0f} hp={hip_push_torque:.0f} trial={trial_num}"
    print(f"\n[{total}] {label}")

    before = set(glob.glob(os.path.join(DATA_DIR, 'run_*.csv')))

    cmd = [
        'ros2', 'launch', 'pady_robot', 'headless.launch.py',
        f'kick_torque:={kick_torque}',
        f'hip_push_torque:={hip_push_torque}',
        f'body_force:={BODY_FORCE}',
        f'spawn_pitch:={SPAWN_PITCH}',
        f'hip_push_start_time:={HIP_PUSH_START}',
        f'hip_push_stop_time:={HIP_PUSH_STOP}',
    ]

    proc = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid,      # own process group → group kill works
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for the new CSV to appear (gait_analyser creates it at startup)
    csv_path = find_new_csv(before, timeout=20.0)
    if csv_path is None:
        print(f"  [sweep] WARNING: no CSV appeared — gait_analyser may not have started")
        kill_launch(proc)
        return None

    # Wait for fall or MAX_TRIAL_TIME
    fell = wait_for_fall(csv_path, MAX_TRIAL_TIME)
    status = "FELL" if fell else "SURVIVED (timeout)"
    print(f"  {status}  →  {os.path.basename(csv_path)}")

    # Brief extra pause to let CSV flush final row
    time.sleep(1.5)
    kill_launch(proc)

    # Read final CSV
    try:
        df = pd.read_csv(csv_path)
        metrics = compute_metrics(df, kick_torque)
        metrics.update({
            'kick_torque':     kick_torque,
            'hip_push_torque': hip_push_torque,
            'body_force':      BODY_FORCE,
            'spawn_pitch':     SPAWN_PITCH,
            'trial':           trial_num,
            'csv_file':        os.path.basename(csv_path),
            'fell':            int(fell),
        })
        print(f"  steps={metrics['step_count']}  "
              f"gait_cv={metrics['gait_cv']:.3f}  "
              f"sym={metrics['symmetry_index']:.4f}  "
              f"SR={metrics['specific_resistance']:.3f}")
        return metrics
    except Exception as e:
        print(f"  [sweep] ERROR reading CSV: {e}")
        return None


def main():
    check_environment()
    os.makedirs(DATA_DIR, exist_ok=True)

    combos = [(kt, hp) for kt in KICK_TORQUES for hp in HIP_PUSH_TORQUES]
    total_runs = len(combos) * N_TRIALS
    estimated_min = total_runs * (MAX_TRIAL_TIME + INTER_TRIAL_PAUSE + 10) / 60

    print("=" * 60)
    print("PaDy Parameter Sweep")
    print("=" * 60)
    print(f"  kick_torques:     {KICK_TORQUES}")
    print(f"  hip_push_torques: {HIP_PUSH_TORQUES}")
    print(f"  trials per combo: {N_TRIALS}")
    print(f"  total runs:       {total_runs}")
    print(f"  estimated time:   ~{estimated_min:.0f} minutes")
    print(f"  data directory:   {DATA_DIR}")
    print("=" * 60)
    input("Press Enter to start (Ctrl+C to abort)...")

    results = []
    run_counter = 0

    for kt, hp in combos:
        for trial in range(1, N_TRIALS + 1):
            run_counter += 1
            result = run_trial(kt, hp, trial, run_counter)
            if result:
                results.append(result)

            time.sleep(INTER_TRIAL_PAUSE)

    # ── Save summary CSV ─────────────────────────────────────────────────────
    if not results:
        print("\nNo results collected.")
        return

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = os.path.join(DATA_DIR, f'sweep_{ts}.csv')

    fieldnames = [
        'kick_torque', 'hip_push_torque', 'body_force', 'spawn_pitch', 'trial',
        'step_count', 'fall_time_s', 'gait_cv', 'symmetry_index',
        'specific_resistance', 'fell', 'csv_file',
    ]
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    print(f"\n{'=' * 60}")
    print(f"Sweep complete. Summary saved to:\n  {summary_path}")
    print(f"Run:  python3 scripts/plot_results.py {summary_path}")
    print("=" * 60)

    # ── Quick terminal table ─────────────────────────────────────────────────
    df = pd.DataFrame(results)
    grouped = df.groupby(['kick_torque', 'hip_push_torque']).agg(
        steps_mean=('step_count',         'mean'),
        steps_std= ('step_count',         'std'),
        gait_cv=   ('gait_cv',            'mean'),
        symmetry=  ('symmetry_index',     'mean'),
        SR=        ('specific_resistance', 'mean'),
    ).reset_index().sort_values('steps_mean', ascending=False)

    print("\nTOP RESULTS (sorted by mean step count):")
    print(f"{'kick':>6} {'hip':>5} | {'steps':>8} {'±':>5} | {'gait_cv':>8} {'sym':>8} {'SR':>8}")
    print("-" * 60)
    for _, row in grouped.head(10).iterrows():
        std = f"{row['steps_std']:.1f}" if not math.isnan(row['steps_std']) else " -"
        cv  = f"{row['gait_cv']:.3f}"   if not math.isnan(row['gait_cv'])   else " -"
        sym = f"{row['symmetry']:.4f}"  if not math.isnan(row['symmetry'])  else " -"
        sr  = f"{row['SR']:.3f}"        if not math.isnan(row['SR'])         else " -"
        print(f"{row['kick_torque']:>6.0f} {row['hip_push_torque']:>5.0f} | "
              f"{row['steps_mean']:>8.1f} {std:>5} | {cv:>8} {sym:>8} {sr:>8}")


if __name__ == '__main__':
    main()
