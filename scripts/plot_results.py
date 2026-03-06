#!/usr/bin/env python3
from __future__ import annotations

"""plot_results.py — Analysis figures and tables for PaDy gait parameter sweeps.

Usage
-----
python3 scripts/plot_results.py ~/ros2_ws/data/sweep_TIMESTAMP.csv
python3 scripts/plot_results.py ~/ros2_ws/data/sweep_TIMESTAMP.csv --best-run

Outputs
-------
Figures saved to ~/ros2_ws/data/figures/  (PNG, 300 dpi)
  fig1_heatmap.png        — step count vs (kick_torque, hip_push_torque)
  fig2_metric_bars.png    — 4 metrics for top parameter sets
  fig3_time_series.png    — joint angles + height for best run
  fig4_scatter.png        — step count vs gait_CV and symmetry_index
"""

import argparse
import csv
import math
import os
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

FIGURES_DIR = os.path.expanduser('~/ros2_ws/data/figures')
DATA_DIR = os.path.expanduser('~/ros2_ws/data')

METRICS = {
    'step_count': ('Step Count', False),
    'gait_cv': ('Gait Period CV', True),
    'symmetry_index': ('Hip Symmetry Index (rad)', True),
    'specific_resistance': ('Specific Resistance', True),
}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f8f8',
    'axes.grid': True,
    'grid.alpha': 0.4,
})


def _to_float(value, default=float('nan')):
    try:
        if value in (None, ''):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv_rows(path: str) -> list[dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return statistics.fmean(vals) if vals else float('nan')


def _std(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return statistics.pstdev(vals) if len(vals) > 1 else float('nan')


def _group_by_combo(rows: list[dict]) -> dict[tuple[float, float], list[dict]]:
    grouped: dict[tuple[float, float], list[dict]] = {}
    for row in rows:
        kt = _to_float(row.get('kick_torque'))
        hp = _to_float(row.get('hip_push_torque'))
        if math.isnan(kt) or math.isnan(hp):
            continue
        grouped.setdefault((kt, hp), []).append(row)
    return grouped


def fig1_heatmap(rows: list[dict], save_dir: str):
    grouped = _group_by_combo(rows)
    kick_vals = sorted({k for (k, _) in grouped.keys()})
    hp_vals = sorted({h for (_, h) in grouped.keys()})

    grid = np.full((len(hp_vals), len(kick_vals)), np.nan)
    for (kick, hip), entries in grouped.items():
        r = hp_vals.index(hip)
        c = kick_vals.index(kick)
        steps = [_to_float(e.get('step_count')) for e in entries]
        grid[r, c] = _mean(steps)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, cmap='YlOrRd', aspect='auto', origin='lower')
    plt.colorbar(im, ax=ax, label='Mean Step Count')

    ax.set_xticks(range(len(kick_vals)))
    ax.set_xticklabels([f'{v:.0f}' for v in kick_vals])
    ax.set_yticks(range(len(hp_vals)))
    ax.set_yticklabels([f'{v:.0f}' for v in hp_vals])
    ax.set_xlabel('Kick Torque (N·m)')
    ax.set_ylabel('Hip Push Torque (N·m)')
    ax.set_title('Fig 1 — Mean Step Count\n(darker = more steps = better)')

    for r in range(len(hp_vals)):
        for c in range(len(kick_vals)):
            v = grid[r, c]
            if not np.isnan(v):
                ax.text(c, r, f'{v:.1f}', ha='center', va='center', fontsize=9, color='black')

    if not np.isnan(grid).all():
        best_idx = np.unravel_index(np.nanargmax(grid), grid.shape)
        ax.add_patch(plt.Rectangle(
            (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
            fill=False, edgecolor='blue', linewidth=2.5, label='Best'
        ))
        ax.legend(loc='upper left')

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig1_heatmap.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')
    return path


def fig2_metric_bars(rows: list[dict], save_dir: str, top_n: int = 6):
    grouped = _group_by_combo(rows)

    ranked = []
    for (kick, hip), entries in grouped.items():
        steps = [_to_float(e.get('step_count')) for e in entries]
        ranked.append((f'kt={kick:.0f}\nhp={hip:.0f}', _mean(steps), entries))
    ranked.sort(key=lambda x: x[1], reverse=True)
    ranked = ranked[:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Fig 2 — 4 Performance Metrics for Top Parameter Sets\n(ranked by mean step count)',
                 fontsize=14, fontweight='bold')

    metric_keys = ['step_count', 'gait_cv', 'symmetry_index', 'specific_resistance']
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

    labels = [x[0] for x in ranked]
    entries_by_label = {x[0]: x[2] for x in ranked}

    for ax, key, color in zip(axes.flat, metric_keys, colors):
        name, lower_is_better = METRICS[key]
        data = []
        for label in labels:
            vals = [_to_float(e.get(key)) for e in entries_by_label[label]]
            vals = [v for v in vals if not math.isnan(v)]
            data.append(vals)

        bp = ax.boxplot(data, patch_artist=True, medianprops={'color': 'black', 'linewidth': 2})
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(name)
        direction = '↓ better' if lower_is_better else '↑ better'
        ax.set_ylabel(f'{name}  ({direction})')

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig2_metric_bars.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')
    return path


def fig3_time_series(run_csv: str, save_dir: str, label: str = ''):
    rows = _read_csv_rows(run_csv)
    t = [_to_float(r.get('sim_time_s')) for r in rows]

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f'Fig 3 — Joint Angle Time Series: Best Run\n{label}', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(4, 1, hspace=0.45)

    fall_t = None
    for row in rows:
        if _to_float(row.get('fall_detected'), 0.0) >= 1.0:
            fall_t = _to_float(row.get('sim_time_s'))
            break

    def add_fall(ax):
        if fall_t is not None and not math.isnan(fall_t):
            ax.axvline(fall_t, color='red', linestyle='--', linewidth=1.5, label=f'Fall t={fall_t:.1f}s')

    hip_r = [_to_float(r.get('hip_right_rad')) for r in rows]
    hip_l = [_to_float(r.get('hip_left_rad')) for r in rows]
    knee_r = [_to_float(r.get('knee_right_rad')) for r in rows]
    knee_l = [_to_float(r.get('knee_left_rad')) for r in rows]
    height = [_to_float(r.get('slope_height_m')) for r in rows]
    sym_abs = [abs(_to_float(r.get('hip_symmetry_rad'))) for r in rows]
    hip_var = [_to_float(r.get('hip_variance_rad')) for r in rows]

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, hip_r, label='hip_right', color='#1976D2', linewidth=1.2)
    ax1.plot(t, hip_l, label='hip_left', color='#42A5F5', linewidth=1.2, linestyle='--')
    ax1.axhline(0, color='gray', linewidth=0.5)
    add_fall(ax1)
    ax1.set_ylabel('Angle (rad)')
    ax1.set_title('Hip Joint Angles')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(-0.7, 0.7)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, knee_r, label='knee_right', color='#388E3C', linewidth=1.2)
    ax2.plot(t, knee_l, label='knee_left', color='#81C784', linewidth=1.2, linestyle='--')
    add_fall(ax2)
    ax2.set_ylabel('Angle (rad)')
    ax2.set_title('Knee Joint Angles')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(-0.1, 1.3)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t, height, label='slope height', color='#7B1FA2', linewidth=1.2)
    add_fall(ax3)
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Projected Height Above Slope')
    ax3.legend(loc='upper right', fontsize=9)

    ax4 = fig.add_subplot(gs[3])
    ax4.plot(t, sym_abs, label='|hip symmetry|', color='#E65100', linewidth=1.2)
    ax4.plot(t, hip_var, label='hip variance', color='#FF8F00', linewidth=1.2, linestyle='--')
    add_fall(ax4)
    ax4.set_ylabel('rad')
    ax4.set_title('Gait Quality')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_xlabel('Simulation Time (s)')

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig3_time_series.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')
    return path


def fig4_scatter(rows: list[dict], save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Fig 4 — Step Count vs Gait Quality Indicators\n(predictors of performance)',
                 fontsize=14, fontweight='bold')

    combos = sorted(_group_by_combo(rows).keys())
    cmap = plt.cm.get_cmap('tab10', len(combos) if combos else 1)
    colour_map = {combo: cmap(i) for i, combo in enumerate(combos)}

    for ax, (x_col, x_label) in zip(axes, [
        ('gait_cv', 'Gait Period CV  (lower = more regular)'),
        ('symmetry_index', 'Hip Symmetry Index (rad)  (lower = more symmetric)'),
    ]):
        for row in rows:
            xv = _to_float(row.get(x_col))
            yv = _to_float(row.get('step_count'), 0.0)
            kick = _to_float(row.get('kick_torque'))
            hip = _to_float(row.get('hip_push_torque'))
            if math.isnan(xv) or math.isnan(kick) or math.isnan(hip):
                continue

            colour = colour_map.get((kick, hip), 'gray')
            ax.scatter(xv, yv, color=colour, s=60, alpha=0.7,
                       label=f'kt={kick:.0f}/hp={hip:.0f}')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc='upper right', ncol=2)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Step Count')
        ax.set_title(f'Step Count vs {x_col}')

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig4_scatter.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')
    return path


def print_table(rows: list[dict]):
    grouped = _group_by_combo(rows)
    ranked = []
    for (kick, hip), entries in grouped.items():
        step_vals = [_to_float(e.get('step_count')) for e in entries]
        cv_vals = [_to_float(e.get('gait_cv')) for e in entries]
        sym_vals = [_to_float(e.get('symmetry_index')) for e in entries]
        sr_vals = [_to_float(e.get('specific_resistance')) for e in entries]
        ranked.append({
            'kick_torque': kick,
            'hip_push_torque': hip,
            'steps_mean': _mean(step_vals),
            'steps_std': _std(step_vals),
            'gait_cv': _mean(cv_vals),
            'symmetry': _mean(sym_vals),
            'SR': _mean(sr_vals),
            'n_trials': len(entries),
        })

    ranked.sort(key=lambda x: x['steps_mean'], reverse=True)

    sep = '─' * 74
    print(f'\n{sep}')
    print(f"  {'Rank':>4}  {'kt':>6}  {'hp':>5}  {'Steps (μ±σ)':>12}  {'GaitCV':>7}  {'Symmetry':>9}  {'SR':>7}  {'n':>3}")
    print(sep)
    for rank, row in enumerate(ranked, 1):
        std = f"±{row['steps_std']:.1f}" if not math.isnan(row['steps_std']) else '     '
        cv = f"{row['gait_cv']:.3f}" if not math.isnan(row['gait_cv']) else '  -  '
        sym = f"{row['symmetry']:.4f}" if not math.isnan(row['symmetry']) else '    -   '
        sr = f"{row['SR']:.3f}" if not math.isnan(row['SR']) else '  -  '
        star = ' ★' if rank == 1 else '  '
        print(f"  {rank:>4}  {row['kick_torque']:>6.0f}  {row['hip_push_torque']:>5.0f}  {row['steps_mean']:>7.1f}{std:>5}  {cv:>7}  {sym:>9}  {sr:>7}  {int(row['n_trials']):>3}{star}")
    print(sep)

    if ranked:
        best = ranked[0]
        print(f"\n  OPTIMAL: kick_torque={best['kick_torque']:.0f} N·m  hip_push_torque={best['hip_push_torque']:.0f} N·m")
        print(f"  → mean {best['steps_mean']:.1f} steps, gait_CV={best['gait_cv']:.3f}, symmetry={best['symmetry']:.4f} rad, SR={best['SR']:.3f}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Plot PaDy sweep results')
    parser.add_argument('sweep_csv', help='Path to sweep_TIMESTAMP.csv from param_sweep.py')
    parser.add_argument('--best-run', action='store_true', help='Also plot time series for the single best run CSV')
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Parse and summarize only (no figures generated)',
    )
    args = parser.parse_args()

    sweep_csv = os.path.expanduser(args.sweep_csv)
    if not os.path.isfile(sweep_csv):
        print(f'ERROR: file not found: {sweep_csv}')
        sys.exit(1)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    rows = _read_csv_rows(sweep_csv)
    unique_combos = len(_group_by_combo(rows))

    print(f"\nLoaded {len(rows)} trial rows from {os.path.basename(sweep_csv)}")
    print(f'Parameter combinations: {unique_combos}')
    print(f"\nGenerating figures → {FIGURES_DIR}/\n")

    print_table(rows)

    if args.smoke:
        print('\nSmoke mode enabled: skipped figure generation.')
        return

    fig1_heatmap(rows, FIGURES_DIR)
    fig2_metric_bars(rows, FIGURES_DIR)
    fig4_scatter(rows, FIGURES_DIR)

    if rows:
        best_row = max(rows, key=lambda r: _to_float(r.get('step_count'), -1.0))
        csv_file = best_row.get('csv_file', '')
        if csv_file:
            run_csv = os.path.join(DATA_DIR, csv_file)
            if os.path.isfile(run_csv):
                label = (
                    f"kick_torque={_to_float(best_row.get('kick_torque')):.0f} N·m  "
                    f"hip_push_torque={_to_float(best_row.get('hip_push_torque')):.0f} N·m  "
                    f"steps={int(_to_float(best_row.get('step_count'), 0.0))}"
                )
                fig3_time_series(run_csv, FIGURES_DIR, label)
            else:
                print(f'  [fig3] individual run CSV not found: {run_csv}')

    print(f'\nAll figures saved to {FIGURES_DIR}')
    print('Open them with:  eog ~/ros2_ws/data/figures/  (or any image viewer)')


if __name__ == '__main__':
    main()
