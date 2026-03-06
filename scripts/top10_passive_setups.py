#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime
import glob
import math
import os
import signal
import statistics
import subprocess
import time
from dataclasses import dataclass

DATA_DIR = os.path.expanduser('~/ros2_ws/data')
MAX_TRIAL_TIME = 45.0
CSV_APPEAR_TIMEOUT = 20.0
KILL_TIMEOUT = 12.0
INTER_TRIAL_PAUSE = 2.0


@dataclass
class Setup:
    name: str
    world: str
    kick_torque: float
    hip_push_torque: float
    hip_push_stop_time: float
    body_force: float
    spawn_x: float
    spawn_pitch: float
    spawn_roll: float
    rationale: str


@dataclass
class Metrics:
    fell: bool
    fall_time_s: float
    step_count: int
    gait_period_mean: float
    gait_period_cv: float
    symmetry_abs_mean: float
    hip_var_cv: float
    knee_var_cv: float
    lateral_drift_m: float
    max_abs_roll_rad: float
    score: float


def _to_float(value, default=float('nan')):
    try:
        if value in (None, ''):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return statistics.fmean(vals) if vals else float('nan')


def _std(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return statistics.pstdev(vals) if len(vals) > 1 else float('nan')


def _cv(values: list[float]) -> float:
    mu = _mean(values)
    if math.isnan(mu) or abs(mu) < 1e-9:
        return float('nan')
    return _std(values) / abs(mu)


def _load_csv_rows(path: str) -> list[dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def find_new_csv(before_set: set[str], timeout: float = CSV_APPEAR_TIMEOUT) -> str | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        after = set(glob.glob(os.path.join(DATA_DIR, 'run_*.csv')))
        new = sorted(after - before_set)
        if new:
            return new[-1]
        time.sleep(0.4)
    return None


def detect_fall_in_csv(csv_path: str) -> tuple[bool, float]:
    try:
        rows = _load_csv_rows(csv_path)
    except Exception:
        return False, float('nan')
    for row in rows:
        if _to_float(row.get('fall_detected'), 0.0) >= 1.0:
            return True, _to_float(row.get('sim_time_s'))
    return False, float('nan')


def wait_for_fall_or_timeout(csv_path: str, max_time: float) -> tuple[bool, float]:
    deadline = time.time() + max_time
    while time.time() < deadline:
        fell, t = detect_fall_in_csv(csv_path)
        if fell:
            return True, t
        time.sleep(0.7)
    return False, float('nan')


def kill_launch(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=KILL_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    except Exception:
        pass


def compute_metrics(rows: list[dict]) -> Metrics:
    if not rows:
        return Metrics(False, float('nan'), 0, float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), -1e9)

    fell = False
    fall_time = float('nan')
    for row in rows:
        if _to_float(row.get('fall_detected'), 0.0) >= 1.0:
            fell = True
            fall_time = _to_float(row.get('sim_time_s'))
            break

    step_count = int(max((_to_float(r.get('step_count'), 0.0) for r in rows), default=0.0))
    periods = [_to_float(r.get('gait_period_s')) for r in rows if _to_float(r.get('gait_period_s')) > 0.05]
    gait_period_mean = _mean(periods)
    gait_period_cv = _cv(periods)

    symmetry_vals = [_to_float(r.get('hip_symmetry_rad')) for r in rows if not math.isnan(_to_float(r.get('hip_symmetry_rad')))]
    symmetry_abs_mean = _mean([abs(v) for v in symmetry_vals])

    hip_var_vals = [_to_float(r.get('hip_variance_rad')) for r in rows if not math.isnan(_to_float(r.get('hip_variance_rad')))]
    knee_var_vals = [_to_float(r.get('knee_variance_rad')) for r in rows if not math.isnan(_to_float(r.get('knee_variance_rad')))]
    hip_var_cv = _cv(hip_var_vals)
    knee_var_cv = _cv(knee_var_vals)

    by = [_to_float(r.get('base_y_m')) for r in rows if not math.isnan(_to_float(r.get('base_y_m')))]
    if by:
        y0 = by[0]
        lateral_drift = max(abs(v - y0) for v in by)
    else:
        lateral_drift = float('nan')

    roll_vals = [abs(_to_float(r.get('base_roll_rad'))) for r in rows if not math.isnan(_to_float(r.get('base_roll_rad')))]
    max_abs_roll = max(roll_vals) if roll_vals else float('nan')

    score = 0.0
    score += 6.0 * step_count
    if fell:
        score -= 8.0
        if not math.isnan(fall_time):
            score += 0.20 * fall_time
    else:
        score += 10.0

    if not math.isnan(gait_period_cv):
        score -= 2.8 * min(gait_period_cv, 1.5)
    if not math.isnan(symmetry_abs_mean):
        score -= 2.2 * min(symmetry_abs_mean, 2.0)
    if not math.isnan(hip_var_cv):
        score -= 1.0 * min(hip_var_cv, 2.0)
    if not math.isnan(knee_var_cv):
        score -= 1.0 * min(knee_var_cv, 2.0)
    if not math.isnan(lateral_drift):
        score -= 14.0 * min(lateral_drift, 0.5)
    if not math.isnan(max_abs_roll):
        score -= 2.8 * max(0.0, max_abs_roll - 0.60)

    return Metrics(
        fell=fell,
        fall_time_s=fall_time,
        step_count=step_count,
        gait_period_mean=gait_period_mean,
        gait_period_cv=gait_period_cv,
        symmetry_abs_mean=symmetry_abs_mean,
        hip_var_cv=hip_var_cv,
        knee_var_cv=knee_var_cv,
        lateral_drift_m=lateral_drift,
        max_abs_roll_rad=max_abs_roll,
        score=score,
    )


def launch_trial(setup: Setup) -> str | None:
    before = set(glob.glob(os.path.join(DATA_DIR, 'run_*.csv')))
    cmd = [
        'ros2', 'launch', 'pady_robot', 'headless.launch.py',
        'use_sim_time:=true',
        f'world:={setup.world}',
        f'kick_torque:={setup.kick_torque:.3f}',
        f'hip_push_torque:={setup.hip_push_torque:.3f}',
        'hip_push_start_time:=7.000',
        f'hip_push_stop_time:={setup.hip_push_stop_time:.3f}',
        f'body_force:={setup.body_force:.3f}',
        f'spawn_x:={setup.spawn_x:.3f}',
        f'spawn_pitch:={setup.spawn_pitch:.4f}',
        f'spawn_roll:={setup.spawn_roll:.4f}',
    ]

    proc = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    csv_path = find_new_csv(before, timeout=CSV_APPEAR_TIMEOUT)
    if csv_path is None:
        kill_launch(proc)
        return None

    wait_for_fall_or_timeout(csv_path, MAX_TRIAL_TIME)
    time.sleep(1.0)
    kill_launch(proc)
    return csv_path


def weighted_hybrid(top_rows: list[dict]) -> dict:
    def w(row: dict) -> float:
        return max(0.5, float(row['score']) + 20.0) + 1.5 * int(row['step_count'])

    total = sum(w(r) for r in top_rows)
    if total <= 0:
        total = 1.0

    def avg(key: str) -> float:
        return sum(w(r) * float(r[key]) for r in top_rows) / total

    world_votes = {}
    for r in top_rows:
        world_votes[r['world']] = world_votes.get(r['world'], 0.0) + w(r)
    best_world = max(world_votes.items(), key=lambda x: x[1])[0]

    return {
        'world': best_world,
        'kick_torque': avg('kick_torque'),
        'hip_push_torque': avg('hip_push_torque'),
        'hip_push_stop_time': avg('hip_push_stop_time'),
        'body_force': avg('body_force'),
        'spawn_x': avg('spawn_x'),
        'spawn_pitch': avg('spawn_pitch'),
        'spawn_roll': avg('spawn_roll'),
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = os.path.join(DATA_DIR, f'top10_setups_{ts}.csv')
    out_md = os.path.join(DATA_DIR, f'top10_setups_{ts}.md')

    setups = [
        Setup('A_baseline_best2step', 'slope_3deg.sdf', 38.14, 8.95, 12.49, 0.61, 0.450, 0.2667, -0.1113,
              'Anchor on current best 2-step profile.'),
        Setup('B_soft_recovery', 'slope_3deg.sdf', 37.30, 7.76, 12.32, 0.00, 0.450, 0.2642, -0.1090,
              'Softer energy to reduce toe catch while keeping transfer.'),
        Setup('C_long_push_window', 'slope_3deg.sdf', 37.85, 8.40, 12.75, 0.35, 0.450, 0.2660, -0.1080,
              'Longer push to sustain step 3 without abrupt collapse.'),
        Setup('D_3p25_low_energy', 'slope_3p25deg.sdf', 35.80, 6.20, 12.10, 0.00, 0.450, 0.2615, -0.1000,
              'Slightly steeper slope with lower forcing for passive cadence.'),
        Setup('E_3p25_mid_energy', 'slope_3p25deg.sdf', 37.00, 7.20, 12.30, 0.15, 0.450, 0.2635, -0.1060,
              'Middle-energy blend on steeper slope for cleaner transfer.'),
        Setup('F_3p25_high_energy', 'slope_3p25deg.sdf', 38.60, 9.40, 12.55, 0.50, 0.450, 0.2668, -0.1130,
              'Higher drive variant for reaching visual step 4.'),
        Setup('G_3p50_passive', 'slope_3p50deg.sdf', 34.80, 5.40, 11.90, 0.00, 0.450, 0.2585, -0.0950,
              'Steepest slope, mostly passive regime.'),
        Setup('H_3p50_transfer', 'slope_3p50deg.sdf', 36.20, 6.80, 12.15, 0.08, 0.448, 0.2612, -0.1020,
              'Steep slope with moderate transfer boost.'),
        Setup('I_com_forward', 'slope_3deg.sdf', 37.90, 8.50, 12.40, 0.35, 0.462, 0.2658, -0.1090,
              'Forward COM proxy (spawn_x) for easier first transfers.'),
        Setup('J_com_backward', 'slope_3p25deg.sdf', 37.40, 8.00, 12.45, 0.25, 0.438, 0.2672, -0.1070,
              'Backward COM proxy to improve toe clearance on later swing.'),
    ]

    rows_out = []
    print('Running 10 curated passive setups...')
    for idx, setup in enumerate(setups, start=1):
        print(f'[{idx}/10] {setup.name} | world={setup.world}')
        csv_path = launch_trial(setup)
        if csv_path is None:
            metrics = Metrics(True, float('nan'), 0, float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), -1e9)
            csv_name = ''
        else:
            metrics = compute_metrics(_load_csv_rows(csv_path))
            csv_name = os.path.basename(csv_path)

        row = {
            'setup': setup.name,
            'world': setup.world,
            'csv_file': csv_name,
            'kick_torque': f'{setup.kick_torque:.3f}',
            'hip_push_torque': f'{setup.hip_push_torque:.3f}',
            'hip_push_stop_time': f'{setup.hip_push_stop_time:.3f}',
            'body_force': f'{setup.body_force:.3f}',
            'spawn_x': f'{setup.spawn_x:.3f}',
            'spawn_pitch': f'{setup.spawn_pitch:.4f}',
            'spawn_roll': f'{setup.spawn_roll:.4f}',
            'rationale': setup.rationale,
            'step_count': metrics.step_count,
            'fell': int(metrics.fell),
            'fall_time_s': '' if math.isnan(metrics.fall_time_s) else f'{metrics.fall_time_s:.3f}',
            'gait_period_mean': '' if math.isnan(metrics.gait_period_mean) else f'{metrics.gait_period_mean:.4f}',
            'gait_period_cv': '' if math.isnan(metrics.gait_period_cv) else f'{metrics.gait_period_cv:.4f}',
            'symmetry_abs_mean': '' if math.isnan(metrics.symmetry_abs_mean) else f'{metrics.symmetry_abs_mean:.6f}',
            'hip_var_cv': '' if math.isnan(metrics.hip_var_cv) else f'{metrics.hip_var_cv:.6f}',
            'knee_var_cv': '' if math.isnan(metrics.knee_var_cv) else f'{metrics.knee_var_cv:.6f}',
            'lateral_drift_m': '' if math.isnan(metrics.lateral_drift_m) else f'{metrics.lateral_drift_m:.6f}',
            'max_abs_roll_rad': '' if math.isnan(metrics.max_abs_roll_rad) else f'{metrics.max_abs_roll_rad:.6f}',
            'score': f'{metrics.score:.6f}',
        }
        rows_out.append(row)
        print(f"  -> steps={metrics.step_count} fell={int(metrics.fell)} score={metrics.score:.3f} csv={csv_name}")
        time.sleep(INTER_TRIAL_PAUSE)

    rows_sorted = sorted(rows_out, key=lambda r: float(r['score']), reverse=True)

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    top3 = rows_sorted[:3]
    hybrid = weighted_hybrid(top3)

    with open(out_md, 'w') as f:
        f.write('# Top-10 Passive Setup Sweep\n\n')
        f.write('- objective: maximize step_count while reducing roll collapse and lateral drift\n')
        f.write('- methodology: 10 distinct slope/COM/torque setups, one trial each\n\n')
        f.write('## Top 5 Results\n')
        for i, r in enumerate(rows_sorted[:5], start=1):
            f.write(
                f"- {i}. {r['setup']} | world={r['world']} | steps={r['step_count']} | fell={r['fell']} "
                f"| score={float(r['score']):.3f} | csv={r['csv_file']}\n"
            )

        f.write('\n## Hybrid Candidate (blended from top 3)\n')
        f.write(f"- world: {hybrid['world']}\n")
        f.write(f"- kick_torque: {hybrid['kick_torque']:.3f}\n")
        f.write(f"- hip_push_torque: {hybrid['hip_push_torque']:.3f}\n")
        f.write(f"- hip_push_stop_time: {hybrid['hip_push_stop_time']:.3f}\n")
        f.write(f"- body_force: {hybrid['body_force']:.3f}\n")
        f.write(f"- spawn_x: {hybrid['spawn_x']:.3f}\n")
        f.write(f"- spawn_pitch: {hybrid['spawn_pitch']:.4f}\n")
        f.write(f"- spawn_roll: {hybrid['spawn_roll']:.4f}\n")

        cmd = (
            f"source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash && "
            f"ros2 launch pady_robot analysis.launch.py world:={hybrid['world']} "
            f"kick_torque:={hybrid['kick_torque']:.3f} hip_push_torque:={hybrid['hip_push_torque']:.3f} "
            f"hip_push_stop_time:={hybrid['hip_push_stop_time']:.3f} body_force:={hybrid['body_force']:.3f} "
            f"spawn_x:={hybrid['spawn_x']:.3f} spawn_pitch:={hybrid['spawn_pitch']:.4f} "
            f"spawn_roll:={hybrid['spawn_roll']:.4f}"
        )
        f.write('\n## Visual Test Command\n')
        f.write(f'- `{cmd}`\n')

    print('\nTop-10 sweep complete')
    print('Summary CSV:', out_csv)
    print('Notes file :', out_md)


if __name__ == '__main__':
    main()
