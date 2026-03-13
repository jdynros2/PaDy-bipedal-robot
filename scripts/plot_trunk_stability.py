#!/usr/bin/env python3
"""Plot 3: Hip body (base_link) orientation vs time — 3 stacked subplots."""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CSV = '/home/elec330-admin/ros2_ws/data/run_20260310_153542.csv'
OUT = '/home/elec330-admin/ros2_ws/src/pady_robot/prerequisite+docs/figures/trunk_stability.png'

with open(CSV) as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader]

t     = np.array([float(r['sim_time_s']) for r in rows])
roll  = np.degrees(np.array([float(r['base_roll_rad']) for r in rows]))
pitch = np.degrees(np.array([float(r['base_pitch_rad']) for r in rows]))
yaw   = np.degrees(np.array([float(r['base_yaw_rad']) for r in rows]))
steps = np.array([int(r['step_count']) for r in rows])
fall  = np.array([int(r['fall_detected']) for r in rows])

# Find fall onset
fall_idx = np.argmax(fall) if fall.any() else len(fall)

# Real stride boundaries (full stride = 2 zero-crossings)
all_transitions = []
for i in range(1, len(steps)):
    if steps[i] != steps[i-1]:
        all_transitions.append((t[i], steps[i]))

stride_labels = []
stride_num = 0
for trans_t, sc in all_transitions:
    if sc % 2 == 0:
        stride_num += 1
        stride_labels.append((trans_t, f'Step {stride_num}'))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

data   = [roll, pitch, yaw]
labels = ['Roll (lateral tilt)', 'Pitch (forward lean)', 'Yaw (heading drift)']
colors = ['#9b59b6', '#e67e22', '#3498db']
human_bands = [(-5, 5), (-4, 4), (-2, 2)]
human_labels = ['Human pelvis \u00b13\u20135\u00b0', 'Human pelvis \u00b12\u20134\u00b0', 'Human pelvis \u22480\u00b0']

for ax, d, lbl, col, hband, hlbl in zip(axes, data, labels, colors, human_bands, human_labels):
    # Human reference
    ax.axhspan(hband[0], hband[1], color='#2ecc71', alpha=0.15, label=hlbl)
    ax.axhline(0, color='grey', lw=0.5, ls='--')

    # Data — full run including fall
    ax.plot(t, d, color=col, lw=1.5)

    # Fall zone
    if fall_idx < len(t):
        ax.axvspan(t[fall_idx], t[-1], color='#e74c3c', alpha=0.10)
        if ax == axes[0]:
            ax.text(t[fall_idx] + 0.02, ax.get_ylim()[1]*0.8, 'FALL',
                    fontsize=10, color='#e74c3c', fontweight='bold')

    # Step markers — only full strides
    for st, slbl in stride_labels:
        ax.axvline(st, color='#2c3e50', lw=1.2, ls='--', alpha=0.7)
        if ax == axes[0]:
            ax.text(st, ax.get_ylim()[1], slbl, ha='center', va='bottom',
                    fontsize=8, color='#2c3e50', fontweight='bold')

    ax.set_ylabel(lbl, fontsize=10)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    # Clean grid — only at major tick positions, thin and subtle
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.grid(True, which='major', axis='both', color='#d0d0d0', lw=0.4, ls='-')
    ax.tick_params(axis='both', which='major', labelsize=9)

    # Remove minor ticks and top/right spines for cleanliness
    ax.minorticks_off()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[-1].set_xlabel('Simulation Time (s)', fontsize=11)
fig.suptitle('PaDy Hip Body (base_link) Orientation vs Human Pelvis Reference',
             fontsize=13, fontweight='bold', y=1.02)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved \u2192 {OUT}')
