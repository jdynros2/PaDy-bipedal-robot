#!/usr/bin/env python3
"""plot_results.py — Analysis figures and tables for PaDy gait parameter sweeps.

Usage
-----
# After a param_sweep.py run:
python3 scripts/plot_results.py ~/ros2_ws/data/sweep_TIMESTAMP.csv

# To also overlay the best run's joint-angle time series:
python3 scripts/plot_results.py ~/ros2_ws/data/sweep_TIMESTAMP.csv --best-run

Outputs
-------
Figures saved to ~/ros2_ws/data/figures/  (PNG, 300 dpi)
  fig1_heatmap.png        — step count vs (kick_torque, hip_push_torque)
  fig2_metric_bars.png    — 4 metrics for top parameter sets
  fig3_time_series.png    — joint angles + height for best run
  fig4_scatter.png        — step count vs gait_CV and symmetry_index

Also prints a full ranked results table to the terminal.
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')   # headless rendering — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

FIGURES_DIR = os.path.expanduser('~/ros2_ws/data/figures')
DATA_DIR    = os.path.expanduser('~/ros2_ws/data')

# Metric display names and "lower is better" flags
METRICS = {
    'step_count':          ('Step Count',              False),   # higher is better
    'gait_cv':             ('Gait Period CV',          True),    # lower is better
    'symmetry_index':      ('Hip Symmetry Index (rad)', True),
    'specific_resistance': ('Specific Resistance',     True),
}

plt.rcParams.update({
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f8f8',
    'axes.grid':        True,
    'grid.alpha':       0.4,
})


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Step Count Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig1_heatmap(df: pd.DataFrame, save_dir: str):
    grouped = df.groupby(['kick_torque', 'hip_push_torque'])['step_count'].mean().reset_index()

    kick_vals = sorted(df['kick_torque'].unique())
    hp_vals   = sorted(df['hip_push_torque'].unique())

    grid = np.full((len(hp_vals), len(kick_vals)), np.nan)
    for _, row in grouped.iterrows():
        r = hp_vals.index(row['hip_push_torque'])
        c = kick_vals.index(row['kick_torque'])
        grid[r, c] = row['step_count']

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

    # Annotate cells with value
    for r in range(len(hp_vals)):
        for c in range(len(kick_vals)):
            v = grid[r, c]
            if not np.isnan(v):
                ax.text(c, r, f'{v:.1f}', ha='center', va='center',
                        fontsize=9, color='black')

    # Mark best cell
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
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — 4-Metric Comparison (top N parameter sets, box plots)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_metric_bars(df: pd.DataFrame, save_dir: str, top_n: int = 6):
    df = df.copy()
    df['param_label'] = df.apply(
        lambda r: f"kt={r['kick_torque']:.0f}\nhp={r['hip_push_torque']:.0f}", axis=1
    )

    # Rank by mean step count to pick top N combos
    ranked = (df.groupby('param_label')['step_count']
              .mean()
              .sort_values(ascending=False)
              .head(top_n)
              .index.tolist())
    top_df = df[df['param_label'].isin(ranked)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Fig 2 — 4 Performance Metrics for Top Parameter Sets\n'
                 '(ranked by mean step count)', fontsize=14, fontweight='bold')

    metric_keys = ['step_count', 'gait_cv', 'symmetry_index', 'specific_resistance']
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

    for ax, key, color in zip(axes.flat, metric_keys, colors):
        name, lower_is_better = METRICS[key]
        data = [top_df[top_df['param_label'] == lbl][key].dropna().values
                for lbl in ranked]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 2})
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(range(1, len(ranked) + 1))
        ax.set_xticklabels(ranked, fontsize=8)
        ax.set_title(name)
        direction = '↓ better' if lower_is_better else '↑ better'
        ax.set_ylabel(f'{name}  ({direction})')

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig2_metric_bars.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 3 — Time Series for Best Run
# ══════════════════════════════════════════════════════════════════════════════

def fig3_time_series(run_csv: str, save_dir: str, label: str = ''):
    df = pd.read_csv(run_csv)
    t = df['sim_time_s']

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f'Fig 3 — Joint Angle Time Series: Best Run\n{label}',
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(4, 1, hspace=0.45)

    # Fall time marker
    fall_rows = df[df['fall_detected'] == 1]
    fall_t = fall_rows['sim_time_s'].iloc[0] if not fall_rows.empty else None

    def add_fall(ax):
        if fall_t is not None:
            ax.axvline(fall_t, color='red', linestyle='--', linewidth=1.5,
                       label=f'Fall t={fall_t:.1f}s')

    # ── Hip angles ────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, df['hip_right_rad'], label='hip_right', color='#1976D2', linewidth=1.2)
    ax1.plot(t, df['hip_left_rad'],  label='hip_left',  color='#42A5F5',
             linewidth=1.2, linestyle='--')
    ax1.axhline(0, color='gray', linewidth=0.5)
    add_fall(ax1)
    ax1.set_ylabel('Angle (rad)')
    ax1.set_title('Hip Joint Angles  (expect ±0.5 rad oscillation)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(-0.7, 0.7)

    # ── Knee angles ───────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, df['knee_right_rad'], label='knee_right', color='#388E3C', linewidth=1.2)
    ax2.plot(t, df['knee_left_rad'],  label='knee_left',  color='#81C784',
             linewidth=1.2, linestyle='--')
    add_fall(ax2)
    ax2.set_ylabel('Angle (rad)')
    ax2.set_title('Knee Joint Angles  (0 = locked stance,  ~1.0 = swing flexion)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(-0.1, 1.3)

    # ── Base height ───────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    if 'base_height_m' in df.columns:
        height = df['base_height_m'].replace('', float('nan')).astype(float)
        ax3.plot(t, height, label='base height', color='#7B1FA2', linewidth=1.2)
        ax3.axhline(0.35, color='red', linestyle=':', linewidth=1, label='fall threshold')
    add_fall(ax3)
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Hip Height in World Frame  (fall when < 0.35 m)')
    ax3.legend(loc='upper right', fontsize=9)

    # ── Symmetry + variance ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(t, df['hip_symmetry_rad'].abs(), label='|hip symmetry|',
             color='#E65100', linewidth=1.2)
    ax4.plot(t, df['hip_variance_rad'], label='hip variance',
             color='#FF8F00', linewidth=1.2, linestyle='--')
    add_fall(ax4)
    ax4.set_ylabel('rad')
    ax4.set_title('Gait Quality  (both should stay low — rising = instability)')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_xlabel('Simulation Time (s)')

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig3_time_series.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 4 — Scatter: step count vs leading indicators
# ══════════════════════════════════════════════════════════════════════════════

def fig4_scatter(df: pd.DataFrame, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Fig 4 — Step Count vs Gait Quality Indicators\n'
                 '(predictors of performance)', fontsize=14, fontweight='bold')

    combos = df[['kick_torque', 'hip_push_torque']].drop_duplicates()
    cmap   = plt.cm.get_cmap('tab10', len(combos))
    colour_map = {
        (row['kick_torque'], row['hip_push_torque']): cmap(i)
        for i, (_, row) in enumerate(combos.iterrows())
    }

    for ax, (x_col, x_label) in zip(axes, [
        ('gait_cv',        'Gait Period CV  (lower = more regular)'),
        ('symmetry_index', 'Hip Symmetry Index (rad)  (lower = more symmetric)'),
    ]):
        for _, row in df.iterrows():
            xv = row.get(x_col, float('nan'))
            yv = row.get('step_count', 0)
            if math.isnan(float(xv if xv is not None else 'nan')):
                continue
            colour = colour_map.get((row['kick_torque'], row['hip_push_torque']), 'gray')
            ax.scatter(xv, yv, color=colour, s=60, alpha=0.7,
                       label=f"kt={row['kick_torque']:.0f}/hp={row['hip_push_torque']:.0f}")

        # De-duplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7,
                  loc='upper right', ncol=2)

        ax.set_xlabel(x_label)
        ax.set_ylabel('Step Count')
        ax.set_title(f'Step Count vs {x_col}')

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig4_scatter.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Terminal summary table
# ══════════════════════════════════════════════════════════════════════════════

def print_table(df: pd.DataFrame):
    grouped = df.groupby(['kick_torque', 'hip_push_torque']).agg(
        steps_mean=('step_count',          'mean'),
        steps_std= ('step_count',          'std'),
        gait_cv=   ('gait_cv',             'mean'),
        symmetry=  ('symmetry_index',      'mean'),
        SR=        ('specific_resistance',  'mean'),
        n_trials=  ('trial',               'count'),
    ).reset_index().sort_values('steps_mean', ascending=False)

    sep = "─" * 74
    print(f"\n{sep}")
    print(f"  {'Rank':>4}  {'kt':>6}  {'hp':>5}  {'Steps (μ±σ)':>12}  "
          f"{'GaitCV':>7}  {'Symmetry':>9}  {'SR':>7}  {'n':>3}")
    print(sep)
    for rank, (_, row) in enumerate(grouped.iterrows(), 1):
        std = f"±{row['steps_std']:.1f}" if not math.isnan(row['steps_std']) else "     "
        cv  = f"{row['gait_cv']:.3f}"    if not math.isnan(row['gait_cv'])   else "  -  "
        sym = f"{row['symmetry']:.4f}"   if not math.isnan(row['symmetry'])  else "    -   "
        sr  = f"{row['SR']:.3f}"         if not math.isnan(row['SR'])        else "  -  "
        star = " ★" if rank == 1 else "  "
        print(f"  {rank:>4}  {row['kick_torque']:>6.0f}  {row['hip_push_torque']:>5.0f}  "
              f"{row['steps_mean']:>7.1f}{std:>5}  {cv:>7}  {sym:>9}  {sr:>7}  "
              f"{int(row['n_trials']):>3}{star}")
    print(sep)

    best = grouped.iloc[0]
    print(f"\n  OPTIMAL: kick_torque={best['kick_torque']:.0f} N·m  "
          f"hip_push_torque={best['hip_push_torque']:.0f} N·m")
    print(f"  → mean {best['steps_mean']:.1f} steps, "
          f"gait_CV={best['gait_cv']:.3f}, "
          f"symmetry={best['symmetry']:.4f} rad, "
          f"SR={best['SR']:.3f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Plot PaDy sweep results')
    parser.add_argument('sweep_csv',
                        help='Path to sweep_TIMESTAMP.csv from param_sweep.py')
    parser.add_argument('--best-run', action='store_true',
                        help='Also plot time series for the single best run CSV')
    args = parser.parse_args()

    sweep_csv = os.path.expanduser(args.sweep_csv)
    if not os.path.isfile(sweep_csv):
        print(f"ERROR: file not found: {sweep_csv}")
        sys.exit(1)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = pd.read_csv(sweep_csv)
    # Ensure numeric columns
    for col in ['step_count', 'gait_cv', 'symmetry_index', 'specific_resistance',
                'kick_torque', 'hip_push_torque', 'trial']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"\nLoaded {len(df)} trial rows from {os.path.basename(sweep_csv)}")
    print(f"Parameter combinations: {df[['kick_torque','hip_push_torque']].drop_duplicates().shape[0]}")
    print(f"\nGenerating figures → {FIGURES_DIR}/\n")

    print_table(df)

    fig1_heatmap(df, FIGURES_DIR)
    fig2_metric_bars(df, FIGURES_DIR)
    fig4_scatter(df, FIGURES_DIR)

    # Time series for best run (optional — needs the individual run CSV)
    if args.best_run or True:   # always attempt if csv_file column present
        best_row = df.sort_values('step_count', ascending=False).iloc[0]
        if 'csv_file' in best_row and pd.notna(best_row['csv_file']):
            run_csv = os.path.join(DATA_DIR, best_row['csv_file'])
            if os.path.isfile(run_csv):
                label = (f"kick_torque={best_row['kick_torque']:.0f} N·m  "
                         f"hip_push_torque={best_row['hip_push_torque']:.0f} N·m  "
                         f"steps={int(best_row['step_count'])}")
                fig3_time_series(run_csv, FIGURES_DIR, label)
            else:
                print(f"  [fig3] individual run CSV not found: {run_csv}")

    print(f"\nAll figures saved to {FIGURES_DIR}")
    print("Open them with:  eog ~/ros2_ws/data/figures/  (or any image viewer)")


if __name__ == '__main__':
    main()
