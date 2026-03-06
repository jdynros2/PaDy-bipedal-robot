#!/usr/bin/env python3
from __future__ import annotations

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

Notes
-----
- This script sends launch arguments directly to `headless.launch.py`.
- Keep fixed values here aligned with your manual baseline in `spawn_pady.launch.py`
    when you want direct comparability between manual and sweep runs.
"""

import csv
import argparse
import datetime
import glob
import math
import os
import signal
import statistics
import subprocess
import sys
import time

# ══════════════════════════════════════════════════════════════════════════════
#  SWEEP CONFIGURATION — edit these values
# ══════════════════════════════════════════════════════════════════════════════

# Parameters to sweep (all combinations will be tested)
# Edit these first when exploring gait stability regions.
KICK_TORQUES      = [20.0, 25.0, 30.0, 35.0, 40.0]   # N·m
HIP_PUSH_TORQUES  = [3.0,  5.0,  7.0,  9.0]           # N·m

# Fixed parameters (not swept)
# Adjust only if you want a different baseline experiment definition.
BODY_FORCE        = 20.0    # N
SPAWN_PITCH       = 0.28    # rad
HIP_PUSH_START    = 7.0     # s
HIP_PUSH_STOP     = 12.0    # s

# Trial settings
# Increase MAX_TRIAL_TIME for long-run stability tests.
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


def _to_float(value, default=float('nan')):
    try:
        if value in (None, ''):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_csv_rows(path: str) -> list[dict]:
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return statistics.fmean(vals) if vals else float('nan')


def _std(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return statistics.pstdev(vals) if len(vals) > 1 else float('nan')


def check_environment():
    if 'AMENT_PREFIX_PATH' not in os.environ:
        print("ERROR: ROS2 workspace not sourced.")
        print("Run: source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash")
        sys.exit(1)


def compute_metrics(rows: list[dict], kick_torque: float) -> dict:
    """Extract the 4 performance metrics from a run CSV."""
    metrics = {
        'step_count':          0,
        'fall_time_s':         float('nan'),
        'gait_cv':             float('nan'),
        'symmetry_index':      float('nan'),
        'specific_resistance': float('nan'),
    }

    if not rows:
        return metrics

    step_values = [_to_float(r.get('step_count'), 0.0) for r in rows]
    metrics['step_count'] = int(max(step_values, default=0.0))

    # Fall time
    for row in rows:
        if _to_float(row.get('fall_detected'), 0.0) >= 1.0:
            metrics['fall_time_s'] = _to_float(row.get('sim_time_s'))
            break

    # Gait CV — coefficient of variation of gait_period (only valid periods)
    periods = [
        _to_float(r.get('gait_period_s'))
        for r in rows
        if _to_float(r.get('gait_period_s')) > 0.1
    ]
    if len(periods) >= 2:
        p_mean = _mean(periods)
        if p_mean > 0.0 and not math.isnan(p_mean):
            metrics['gait_cv'] = float(_std(periods) / p_mean)

    # Symmetry index — mean |hip_symmetry| during active walking
    sym_values = [
        abs(_to_float(r.get('hip_symmetry_rad')))
        for r in rows
        if _to_float(r.get('step_count'), 0.0) > 0.0
        and _to_float(r.get('fall_detected'), 0.0) == 0.0
    ]
    if sym_values:
        metrics['symmetry_index'] = float(_mean(sym_values))

    # Specific Resistance — E / (m·g·d)
    # Energy ≈ total angular displacement of hips × kick_torque (rough proxy)
    hip_right = [_to_float(r.get('hip_right_rad')) for r in rows]
    hip_left = [_to_float(r.get('hip_left_rad')) for r in rows]
    if hip_right and metrics['step_count'] > 0:
        hip_work = 0.0
        for i in range(1, len(hip_right)):
            if not math.isnan(hip_right[i]) and not math.isnan(hip_right[i - 1]):
                hip_work += abs(hip_right[i] - hip_right[i - 1])
            if not math.isnan(hip_left[i]) and not math.isnan(hip_left[i - 1]):
                hip_work += abs(hip_left[i] - hip_left[i - 1])
        hip_work *= kick_torque
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
            rows = _load_csv_rows(csv_path)
            if any(_to_float(r.get('fall_detected'), 0.0) >= 1.0 for r in rows):
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
        rows = _load_csv_rows(csv_path)
        metrics = compute_metrics(rows, kick_torque)
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


def run_smoke(csv_path: str, kick_torque: float) -> int:
    """Quick metrics validation using one existing run CSV."""
    if not os.path.isfile(csv_path):
        print(f"ERROR: smoke CSV not found: {csv_path}")
        return 1

    rows = _load_csv_rows(csv_path)
    metrics = compute_metrics(rows, kick_torque)

    print("=" * 60)
    print("PaDy sweep smoke check")
    print("=" * 60)
    print(f"csv: {csv_path}")
    print(f"rows: {len(rows)}")
    print(f"step_count: {metrics['step_count']}")
    print(f"fall_time_s: {metrics['fall_time_s']}")
    print(f"gait_cv: {metrics['gait_cv']}")
    print(f"symmetry_index: {metrics['symmetry_index']}")
    print(f"specific_resistance: {metrics['specific_resistance']}")
    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(description='Run PaDy parameter sweep')
    parser.add_argument(
        '--smoke-csv',
        help='Path to a run_*.csv file for quick metric validation (no Gazebo launch)',
    )
    parser.add_argument(
        '--smoke-kick-torque',
        type=float,
        default=30.0,
        help='Kick torque value used for specific-resistance calculation in smoke mode',
    )
    args = parser.parse_args()

    if args.smoke_csv:
        sys.exit(run_smoke(os.path.expanduser(args.smoke_csv), args.smoke_kick_torque))

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
    grouped = {}
    for row in results:
        key = (row['kick_torque'], row['hip_push_torque'])
        grouped.setdefault(key, {
            'step_count': [],
            'gait_cv': [],
            'symmetry_index': [],
            'specific_resistance': [],
        })
        grouped[key]['step_count'].append(_to_float(row.get('step_count')))
        grouped[key]['gait_cv'].append(_to_float(row.get('gait_cv')))
        grouped[key]['symmetry_index'].append(_to_float(row.get('symmetry_index')))
        grouped[key]['specific_resistance'].append(_to_float(row.get('specific_resistance')))

    ranked = []
    for (kick, hip), vals in grouped.items():
        ranked.append({
            'kick_torque': kick,
            'hip_push_torque': hip,
            'steps_mean': _mean(vals['step_count']),
            'steps_std': _std(vals['step_count']),
            'gait_cv': _mean(vals['gait_cv']),
            'symmetry': _mean(vals['symmetry_index']),
            'SR': _mean(vals['specific_resistance']),
        })
    ranked.sort(key=lambda x: x['steps_mean'], reverse=True)

    print("\nTOP RESULTS (sorted by mean step count):")
    print(f"{'kick':>6} {'hip':>5} | {'steps':>8} {'±':>5} | {'gait_cv':>8} {'sym':>8} {'SR':>8}")
    print("-" * 60)
    for row in ranked[:10]:
        std = f"{row['steps_std']:.1f}" if not math.isnan(row['steps_std']) else " -"
        cv  = f"{row['gait_cv']:.3f}"   if not math.isnan(row['gait_cv'])   else " -"
        sym = f"{row['symmetry']:.4f}"  if not math.isnan(row['symmetry'])  else " -"
        sr  = f"{row['SR']:.3f}"        if not math.isnan(row['SR'])         else " -"
        print(f"{row['kick_torque']:>6.0f} {row['hip_push_torque']:>5.0f} | "
              f"{row['steps_mean']:>8.1f} {std:>5} | {cv:>8} {sym:>8} {sr:>8}")


if __name__ == '__main__':
    main()
