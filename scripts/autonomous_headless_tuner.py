#!/usr/bin/env python3
from __future__ import annotations

import csv
import argparse
import datetime
import glob
import math
import os
import random
import signal
import statistics
import subprocess
import time
from dataclasses import dataclass

DATA_DIR = os.path.expanduser('~/ros2_ws/data')

DEFAULT_RUNS = 20
MAX_TRIAL_TIME = 45.0
CSV_APPEAR_TIMEOUT = 20.0
KILL_TIMEOUT = 12.0
INTER_TRIAL_PAUSE = 3.0

WORLD = 'slope_3deg.sdf'
SPAWN_X = 0.45
HIP_PUSH_START = 0.8


@dataclass
class Params:
    kick_torque: float
    hip_push_torque: float
    hip_push_stop_time: float
    body_force: float
    spawn_pitch: float
    spawn_roll: float


@dataclass
class Metrics:
    fell: bool
    fall_time_s: float
    step_count: int
    gait_period_mean: float
    gait_period_cv: float
    gait_period_positive: bool
    symmetry_abs_mean: float
    symmetry_bias_abs: float
    symmetry_crosses_zero: bool
    hip_var_mean: float
    hip_var_cv: float
    knee_var_mean: float
    knee_var_cv: float
    lateral_drift_m: float
    lateral_drift_signed_m: float
    max_abs_roll_rad: float
    score: float


def _to_float(value, default=float('nan')):
    try:
        if value in (None, ''):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_csv_rows(path: str) -> list[dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


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


def _fmt_num(value: float, digits: int = 4) -> str:
    if math.isnan(value):
        return 'nan'
    return f'{value:.{digits}f}'


def _is_similar_run(prev: Metrics | None, cur: Metrics) -> bool:
    if prev is None:
        return False
    if prev.step_count != cur.step_count:
        return False
    if prev.fell != cur.fell:
        return False

    checks = []
    if not math.isnan(prev.fall_time_s) and not math.isnan(cur.fall_time_s):
        checks.append(abs(prev.fall_time_s - cur.fall_time_s) < 0.20)
    if not math.isnan(prev.gait_period_cv) and not math.isnan(cur.gait_period_cv):
        checks.append(abs(prev.gait_period_cv - cur.gait_period_cv) < 0.06)
    if not math.isnan(prev.symmetry_abs_mean) and not math.isnan(cur.symmetry_abs_mean):
        checks.append(abs(prev.symmetry_abs_mean - cur.symmetry_abs_mean) < 0.06)
    if not math.isnan(prev.max_abs_roll_rad) and not math.isnan(cur.max_abs_roll_rad):
        checks.append(abs(prev.max_abs_roll_rad - cur.max_abs_roll_rad) < 0.12)

    return bool(checks) and all(checks)


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
        return Metrics(
            fell=False,
            fall_time_s=float('nan'),
            step_count=0,
            gait_period_mean=float('nan'),
            gait_period_cv=float('nan'),
            gait_period_positive=False,
            symmetry_abs_mean=float('nan'),
            symmetry_bias_abs=float('nan'),
            symmetry_crosses_zero=False,
            hip_var_mean=float('nan'),
            hip_var_cv=float('nan'),
            knee_var_mean=float('nan'),
            knee_var_cv=float('nan'),
            lateral_drift_m=float('nan'),
            lateral_drift_signed_m=float('nan'),
            max_abs_roll_rad=float('nan'),
            score=-1e9,
        )

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
    gait_period_positive = len(periods) > 0 and not any(p <= 0.0 for p in periods)

    symmetry_vals = [_to_float(r.get('hip_symmetry_rad')) for r in rows if not math.isnan(_to_float(r.get('hip_symmetry_rad')))]
    symmetry_abs_mean = _mean([abs(v) for v in symmetry_vals])
    symmetry_bias_abs = abs(_mean(symmetry_vals))
    symmetry_crosses_zero = (min(symmetry_vals) < 0.0 and max(symmetry_vals) > 0.0) if symmetry_vals else False

    hip_var_vals = [_to_float(r.get('hip_variance_rad')) for r in rows if not math.isnan(_to_float(r.get('hip_variance_rad')))]
    knee_var_vals = [_to_float(r.get('knee_variance_rad')) for r in rows if not math.isnan(_to_float(r.get('knee_variance_rad')))]
    hip_var_mean = _mean(hip_var_vals)
    knee_var_mean = _mean(knee_var_vals)
    hip_var_cv = _cv(hip_var_vals)
    knee_var_cv = _cv(knee_var_vals)

    by = [_to_float(r.get('base_y_m')) for r in rows if not math.isnan(_to_float(r.get('base_y_m')))]
    if by:
        y0 = by[0]
        lateral_drift = max(abs(v - y0) for v in by)
        lateral_drift_signed = by[-1] - y0
    else:
        lateral_drift = float('nan')
        lateral_drift_signed = float('nan')

    roll_vals = [abs(_to_float(r.get('base_roll_rad'))) for r in rows if not math.isnan(_to_float(r.get('base_roll_rad')))]
    max_abs_roll = max(roll_vals) if roll_vals else float('nan')

    score = 0.0
    score += 1.8 * step_count
    if fell:
        score -= 6.0
        if not math.isnan(fall_time):
            score += 0.08 * fall_time
    else:
        score += 4.0

    if gait_period_positive:
        score += 1.0
    if not math.isnan(gait_period_cv):
        score -= 2.2 * min(gait_period_cv, 1.5)

    if not math.isnan(symmetry_abs_mean):
        score -= 2.4 * min(symmetry_abs_mean, 2.0)
    if not math.isnan(symmetry_bias_abs):
        score -= 1.8 * min(symmetry_bias_abs, 2.0)
    if symmetry_crosses_zero:
        score += 0.8
    else:
        score -= 0.6

    if not math.isnan(hip_var_cv):
        score -= 1.0 * min(hip_var_cv, 2.0)
    if not math.isnan(knee_var_cv):
        score -= 1.0 * min(knee_var_cv, 2.0)
    if not math.isnan(lateral_drift):
        score -= 10.0 * min(lateral_drift, 0.5)
    if not math.isnan(max_abs_roll):
        score -= 2.5 * max(0.0, max_abs_roll - 0.45)

    return Metrics(
        fell=fell,
        fall_time_s=fall_time,
        step_count=step_count,
        gait_period_mean=gait_period_mean,
        gait_period_cv=gait_period_cv,
        gait_period_positive=gait_period_positive,
        symmetry_abs_mean=symmetry_abs_mean,
        symmetry_bias_abs=symmetry_bias_abs,
        symmetry_crosses_zero=symmetry_crosses_zero,
        hip_var_mean=hip_var_mean,
        hip_var_cv=hip_var_cv,
        knee_var_mean=knee_var_mean,
        knee_var_cv=knee_var_cv,
        lateral_drift_m=lateral_drift,
        lateral_drift_signed_m=lateral_drift_signed,
        max_abs_roll_rad=max_abs_roll,
        score=score,
    )


def clamp_params(params: Params) -> Params:
    return Params(
        kick_torque=max(0.0, min(2.0, params.kick_torque)),
        hip_push_torque=max(0.0, min(2.0, params.hip_push_torque)),
        hip_push_stop_time=max(0.0, min(20.0, params.hip_push_stop_time)),
        body_force=max(0.0, min(8.0, params.body_force)),
        spawn_pitch=max(0.24, min(0.44, params.spawn_pitch)),
        spawn_roll=max(-0.26, min(0.02, params.spawn_roll)),
    )


def propose_next(best: Params, latest_m: Metrics, run_index: int, jump_multiplier: float = 1.0) -> tuple[Params, str]:
    scale = max(0.20, 1.0 - 0.03 * run_index) * max(1.0, min(2.5, jump_multiplier))
    roll_nudge = 0.0
    if not math.isnan(latest_m.lateral_drift_signed_m) and abs(latest_m.lateral_drift_signed_m) > 0.12:
        roll_nudge = 0.006 * scale if latest_m.lateral_drift_signed_m < 0.0 else -0.006 * scale

    if latest_m.fell:
        if latest_m.step_count <= 1:
            cand = Params(
                kick_torque=best.kick_torque + 0.9 * scale,
                hip_push_torque=best.hip_push_torque + 1.1 * scale,
                hip_push_stop_time=best.hip_push_stop_time + 0.16 * scale,
                body_force=best.body_force + 0.2 * scale,
                spawn_pitch=best.spawn_pitch + 0.0012 * scale,
                spawn_roll=best.spawn_roll - 0.004 * scale + 0.5 * roll_nudge,
            )
            reason = 'Fall happened before meaningful progression; increased transfer energy and slightly increased initial lean.'
            return clamp_params(cand), reason

        if not math.isnan(latest_m.max_abs_roll_rad) and latest_m.max_abs_roll_rad > 0.9:
            cand = Params(
                kick_torque=best.kick_torque - 1.0 * scale,
                hip_push_torque=best.hip_push_torque - 1.4 * scale,
                hip_push_stop_time=best.hip_push_stop_time - 0.2 * scale,
                body_force=best.body_force - 0.8 * scale,
                spawn_pitch=best.spawn_pitch - 0.003 * scale,
                spawn_roll=best.spawn_roll + roll_nudge,
            )
            reason = 'Fall with excessive roll/lateral behavior; reduced forward drive and slightly reduced pitch.'
        else:
            cand = Params(
                kick_torque=best.kick_torque - 0.6 * scale,
                hip_push_torque=best.hip_push_torque - 0.8 * scale,
                hip_push_stop_time=best.hip_push_stop_time - 0.1 * scale,
                body_force=best.body_force - 0.4 * scale,
                spawn_pitch=best.spawn_pitch,
                spawn_roll=best.spawn_roll + roll_nudge,
            )
            reason = 'Fall detected; softened actuation slightly to avoid toe catch/over-rotation.'
        return clamp_params(cand), reason

    # No fall: keep stability and try to improve progress.
    if latest_m.step_count < 3:
        cand = Params(
            kick_torque=best.kick_torque + 0.7 * scale,
            hip_push_torque=best.hip_push_torque + 0.9 * scale,
            hip_push_stop_time=best.hip_push_stop_time + 0.12 * scale,
            body_force=best.body_force + 0.5 * scale,
            spawn_pitch=best.spawn_pitch + 0.0015 * scale,
            spawn_roll=best.spawn_roll + 0.6 * roll_nudge,
        )
        reason = 'Stable but low progression; increased forward energy slightly to reach next step.'
        return clamp_params(cand), reason

    # If progression is good, improve quality metrics.
    if (not math.isnan(latest_m.gait_period_cv) and latest_m.gait_period_cv > 0.28) or \
       (not math.isnan(latest_m.lateral_drift_m) and latest_m.lateral_drift_m > 0.10):
        cand = Params(
            kick_torque=best.kick_torque - 0.4 * scale,
            hip_push_torque=best.hip_push_torque - 0.5 * scale,
            hip_push_stop_time=best.hip_push_stop_time - 0.1 * scale,
            body_force=best.body_force - 0.25 * scale,
            spawn_pitch=best.spawn_pitch - 0.001 * scale,
            spawn_roll=best.spawn_roll + roll_nudge,
        )
        reason = 'Progress good but variability/drift high; trimmed energy to improve consistency.'
        return clamp_params(cand), reason

    # Exploit best region with mild random local search
    cand = Params(
        kick_torque=best.kick_torque + random.uniform(-0.5, 0.5) * scale,
        hip_push_torque=best.hip_push_torque + random.uniform(-0.6, 0.6) * scale,
        hip_push_stop_time=best.hip_push_stop_time + random.uniform(-0.10, 0.10) * scale,
        body_force=best.body_force + random.uniform(-0.4, 0.4) * scale,
        spawn_pitch=best.spawn_pitch + random.uniform(-0.0015, 0.0015) * scale,
        spawn_roll=best.spawn_roll + random.uniform(-0.010, 0.010) * scale,
    )
    reason = 'Maintaining stable region while probing nearby settings for better combined score.'
    return clamp_params(cand), reason


def launch_trial(params: Params) -> tuple[str | None, bool, float]:
    before = set(glob.glob(os.path.join(DATA_DIR, 'run_*.csv')))

    cmd = [
        'ros2', 'launch', 'pady_robot', 'headless.launch.py',
        'use_sim_time:=true',
        f'world:={WORLD}',
        f'kick_torque:={params.kick_torque:.3f}',
        f'hip_push_torque:={params.hip_push_torque:.3f}',
        f'hip_push_start_time:={HIP_PUSH_START:.3f}',
        f'hip_push_stop_time:={params.hip_push_stop_time:.3f}',
        f'body_force:={params.body_force:.3f}',
        f'spawn_x:={SPAWN_X:.3f}',
        f'spawn_pitch:={params.spawn_pitch:.4f}',
        f'spawn_roll:={params.spawn_roll:.4f}',
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
        return None, False, float('nan')

    fell, fall_t = wait_for_fall_or_timeout(csv_path, MAX_TRIAL_TIME)
    time.sleep(1.2)
    kill_launch(proc)
    return csv_path, fell, fall_t


def main():
    parser = argparse.ArgumentParser(description='Autonomous headless gait tuner')
    parser.add_argument('--runs', type=int, default=DEFAULT_RUNS,
                        help='Number of autonomous tuning runs')
    parser.add_argument('--kick-torque', type=float, default=41.0,
                        help='Initial kick torque (N·m)')
    parser.add_argument('--hip-push-torque', type=float, default=13.0,
                        help='Initial hip push torque (N·m)')
    parser.add_argument('--hip-push-stop-time', type=float, default=13.0,
                        help='Initial hip push stop time (s)')
    parser.add_argument('--body-force', type=float, default=3.0,
                        help='Initial body force (N)')
    parser.add_argument('--spawn-pitch', type=float, default=0.28,
                        help='Initial spawn pitch (rad)')
    parser.add_argument('--spawn-roll', type=float, default=-0.09,
                        help='Initial spawn roll (rad)')
    parser.add_argument('--arm-mass', type=float, default=0.3, help='Arm mass (kg)')
    parser.add_argument('--arm-com', type=float, default=-0.14, help='Arm COM (m)')
    parser.add_argument('--spawn-yaw', type=float, default=0.1, help='Initial spawn yaw (rad)')
    parser.add_argument('--spawn-y', type=float, default=0.3, help='Initial spawn y-coordinate')
    args = parser.parse_args()

    runs = max(1, int(args.runs))

    os.makedirs(DATA_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    summary_csv = os.path.join(DATA_DIR, f'autotune_headless_{ts}.csv')
    notes_md = os.path.join(DATA_DIR, f'autotune_notes_{ts}.md')

    current = Params(
        kick_torque=args.kick_torque,
        hip_push_torque=args.hip_push_torque,
        hip_push_stop_time=args.hip_push_stop_time,
        body_force=args.body_force,
        spawn_pitch=args.spawn_pitch,
        spawn_roll=args.spawn_roll,
    )

    best = current
    best_score = -1e18

    fieldnames = [
        'run', 'csv_file',
        'kick_torque', 'hip_push_torque', 'hip_push_stop_time', 'body_force', 'spawn_pitch', 'spawn_roll',
        'fell', 'fall_time_s', 'step_count',
        'gait_period_mean', 'gait_period_cv', 'gait_period_positive',
        'symmetry_abs_mean', 'symmetry_bias_abs', 'symmetry_crosses_zero',
        'hip_var_mean', 'hip_var_cv', 'knee_var_mean', 'knee_var_cv',
        'lateral_drift_m', 'max_abs_roll_rad', 'score',
        'decision_reason',
    ]

    with open(summary_csv, 'w', newline='') as sf, open(notes_md, 'w') as nf:
        writer = csv.DictWriter(sf, fieldnames=fieldnames)
        writer.writeheader()

        nf.write('# Autonomous Headless Tuning Notes\n\n')
        nf.write(f'- timestamp: {ts}\n')
        nf.write(f'- runs: {runs}\n')
        nf.write(f'- objective: maximize walking progression while satisfying gait stability criteria\n\n')

        prev_metrics = None
        stagnation_count = 0

        for run_idx in range(1, runs + 1):
            print(f'[{run_idx}/{runs}] running params: '
                  f'kick={current.kick_torque:.2f}, hip_push={current.hip_push_torque:.2f}, '
                  f'stop={current.hip_push_stop_time:.2f}, body={current.body_force:.2f}, pitch={current.spawn_pitch:.4f}, roll={current.spawn_roll:.4f}')

            csv_path, _, _ = launch_trial(current)
            if csv_path is None:
                reason = 'Run failed: no CSV appeared; nudging parameters conservatively.'
                metrics = Metrics(
                    fell=True, fall_time_s=float('nan'), step_count=0,
                    gait_period_mean=float('nan'), gait_period_cv=float('nan'), gait_period_positive=False,
                    symmetry_abs_mean=float('nan'), symmetry_bias_abs=float('nan'), symmetry_crosses_zero=False,
                    hip_var_mean=float('nan'), hip_var_cv=float('nan'), knee_var_mean=float('nan'), knee_var_cv=float('nan'),
                    lateral_drift_m=float('nan'), lateral_drift_signed_m=float('nan'), max_abs_roll_rad=float('nan'), score=-9999,
                )
            else:
                rows = _load_csv_rows(csv_path)
                metrics = compute_metrics(rows)
                reason = ''

            improved = metrics.score > best_score
            if improved:
                best = current
                best_score = metrics.score

            if _is_similar_run(prev_metrics, metrics):
                stagnation_count += 1
            else:
                stagnation_count = 0
            jump_multiplier = 1.0 + min(1.5, 0.35 * stagnation_count)

            next_params, model_reason = propose_next(best, metrics, run_idx, jump_multiplier=jump_multiplier)
            decision_reason = reason or model_reason
            if stagnation_count > 0:
                decision_reason += f' Similar run pattern x{stagnation_count+1}; increased parameter jump ({jump_multiplier:.2f}x).'

            row = {
                'run': run_idx,
                'csv_file': os.path.basename(csv_path) if csv_path else '',
                'kick_torque': f'{current.kick_torque:.3f}',
                'hip_push_torque': f'{current.hip_push_torque:.3f}',
                'hip_push_stop_time': f'{current.hip_push_stop_time:.3f}',
                'body_force': f'{current.body_force:.3f}',
                'spawn_pitch': f'{current.spawn_pitch:.4f}',
                'spawn_roll': f'{current.spawn_roll:.4f}',
                'fell': int(metrics.fell),
                'fall_time_s': '' if math.isnan(metrics.fall_time_s) else f'{metrics.fall_time_s:.3f}',
                'step_count': metrics.step_count,
                'gait_period_mean': '' if math.isnan(metrics.gait_period_mean) else f'{metrics.gait_period_mean:.4f}',
                'gait_period_cv': '' if math.isnan(metrics.gait_period_cv) else f'{metrics.gait_period_cv:.4f}',
                'gait_period_positive': int(metrics.gait_period_positive),
                'symmetry_abs_mean': '' if math.isnan(metrics.symmetry_abs_mean) else f'{metrics.symmetry_abs_mean:.6f}',
                'symmetry_bias_abs': '' if math.isnan(metrics.symmetry_bias_abs) else f'{metrics.symmetry_bias_abs:.6f}',
                'symmetry_crosses_zero': int(metrics.symmetry_crosses_zero),
                'hip_var_mean': '' if math.isnan(metrics.hip_var_mean) else f'{metrics.hip_var_mean:.6f}',
                'hip_var_cv': '' if math.isnan(metrics.hip_var_cv) else f'{metrics.hip_var_cv:.6f}',
                'knee_var_mean': '' if math.isnan(metrics.knee_var_mean) else f'{metrics.knee_var_mean:.6f}',
                'knee_var_cv': '' if math.isnan(metrics.knee_var_cv) else f'{metrics.knee_var_cv:.6f}',
                'lateral_drift_m': '' if math.isnan(metrics.lateral_drift_m) else f'{metrics.lateral_drift_m:.6f}',
                'max_abs_roll_rad': '' if math.isnan(metrics.max_abs_roll_rad) else f'{metrics.max_abs_roll_rad:.6f}',
                'score': f'{metrics.score:.6f}',
                'decision_reason': decision_reason,
            }
            writer.writerow(row)
            sf.flush()

            nf.write(f'## Run {run_idx}\n')
            nf.write(f'- csv: {os.path.basename(csv_path) if csv_path else "(none)"}\n')
            nf.write(f'- params: kick={current.kick_torque:.2f}, hip_push={current.hip_push_torque:.2f}, '
                     f'stop={current.hip_push_stop_time:.2f}, body={current.body_force:.2f}, '
                     f'pitch={current.spawn_pitch:.4f}, roll={current.spawn_roll:.4f}\n')
            nf.write(f'- results: steps={metrics.step_count}, fell={metrics.fell}, '
                     f'gait_period_cv={_fmt_num(metrics.gait_period_cv)}, '
                     f'sym_abs_mean={_fmt_num(metrics.symmetry_abs_mean)}, '
                     f'lat_drift={_fmt_num(metrics.lateral_drift_m)}, '
                     f'max_roll={_fmt_num(metrics.max_abs_roll_rad)}, '
                     f'score={metrics.score:.3f}\n')
            nf.write(f'- decision: {decision_reason}\n')
            nf.write(f'- next params: kick={next_params.kick_torque:.2f}, hip_push={next_params.hip_push_torque:.2f}, '
                     f'stop={next_params.hip_push_stop_time:.2f}, body={next_params.body_force:.2f}, '
                     f'pitch={next_params.spawn_pitch:.4f}, roll={next_params.spawn_roll:.4f}\n\n')
            nf.flush()

            print(f'  -> steps={metrics.step_count} fell={int(metrics.fell)} score={metrics.score:.3f} | {decision_reason}')
            if improved:
                print(f'  -> new best score {best_score:.3f}')

            prev_metrics = metrics
            current = next_params
            time.sleep(INTER_TRIAL_PAUSE)

        nf.write('## Final Best Parameters\n')
        nf.write(f'- kick_torque: {best.kick_torque:.3f}\n')
        nf.write(f'- hip_push_torque: {best.hip_push_torque:.3f}\n')
        nf.write(f'- hip_push_stop_time: {best.hip_push_stop_time:.3f}\n')
        nf.write(f'- body_force: {best.body_force:.3f}\n')
        nf.write(f'- spawn_pitch: {best.spawn_pitch:.4f}\n')
        nf.write(f'- spawn_roll: {best.spawn_roll:.4f}\n')
        nf.write(f'- best_score: {best_score:.6f}\n')

    print('\nAutonomous tuning complete')
    print('Summary CSV:', summary_csv)
    print('Notes file :', notes_md)


if __name__ == '__main__':
    main()
