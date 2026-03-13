#!/usr/bin/env python3
"""Plot forward distance vs time for 5 slope angles.

Reads CSVs from ~/ros2_ws/data/slope_tests/slope_Xdeg.csv
Saves to prerequisite+docs/figures/slope_stability.png
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUT = '/home/elec330-admin/ros2_ws/src/pady_robot/prerequisite+docs/figures/slope_stability.png'
DATA_DIR = os.path.expanduser('~/ros2_ws/data/slope_tests')

SLOPES = [
    ('3p00', '3.00°', '#2196F3', '3 steps'),
    ('3p25', '3.25°', '#4CAF50', '3 steps'),
    ('3p50', '3.50°', '#FF9800', '2.5 steps'),
    ('3p75', '3.75°', '#9C27B0', '2 steps'),
    ('4p00', '4.00°', '#F44336', '0 steps'),
]

fig, ax = plt.subplots(figsize=(10, 5.5))

for slope_id, label, colour, steps_note in SLOPES:
    csv_path = os.path.join(DATA_DIR, f'slope_{slope_id}deg.csv')
    if not os.path.exists(csv_path):
        print(f'  SKIP {label} — {csv_path} not found')
        continue

    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r['base_x_m'] != '']

    t = np.array([float(r['sim_time_s']) for r in rows])
    x = np.array([float(r['base_x_m']) for r in rows])
    fall = np.array([int(r['fall_detected']) for r in rows])

    # Trim to pre-fall data (include fall moment)
    fall_idx = np.argmax(fall) if fall.any() else len(fall) - 1
    t_plot = t[:fall_idx + 1]
    x_plot = x[:fall_idx + 1]

    ax.plot(t_plot, x_plot, color=colour, linewidth=2.2,
            label=f'{label}  ({steps_note})', zorder=3)

    # Mark fall point
    if fall.any():
        ax.plot(t_plot[-1], x_plot[-1], 'x', color=colour, markersize=11,
                markeredgewidth=2.8, zorder=4)

# Fall marker legend entry
ax.plot([], [], 'kx', markersize=9, markeredgewidth=2.5, label='Fall event')

# Style
ax.set_xlabel('Simulation Time (s)', fontsize=12, fontweight='bold')
ax.set_ylabel('Forward Distance (m)', fontsize=12, fontweight='bold')
ax.set_title('Walking Stability Across Slope Angles', fontsize=14, fontweight='bold')
ax.legend(fontsize=10.5, loc='upper left', framealpha=0.9,
          edgecolor='#cccccc')
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.set_xlim(left=0)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.tick_params(labelsize=10)
ax.axhline(0, color='black', linewidth=0.5, zorder=1)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved → {OUT}')
plt.close()
