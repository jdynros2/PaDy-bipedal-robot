#!/usr/bin/env python3
"""Plot forward distance vs time for ultra-fine slope sweep.

Two bands: 2.85°–3.05° and 3.35°–3.55° in 0.05° increments.
Saves to prerequisite+docs/figures/slope_stability_ultrafine.png
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUT = '/home/elec330-admin/ros2_ws/src/pady_robot/prerequisite+docs/figures/slope_stability_ultrafine.png'
DATA_DIR = os.path.expanduser('~/ros2_ws/data/slope_tests')

# Two bands with distinct colour families
BAND_LOW = [
    ('2p85', '2.85°', '1.5 steps'),
    ('2p90', '2.90°', '2 steps'),
    ('2p95', '2.95°', '3 steps'),
    ('3p00', '3.00°', '3 steps'),
    ('3p05', '3.05°', '1.5 steps'),
]
BAND_HIGH = [
    ('3p35', '3.35°', '2 steps'),
    ('3p40', '3.40°', '3 steps'),
    ('3p45', '3.45°', '3 steps'),
    ('3p50', '3.50°', '3 steps'),
    ('3p55', '3.55°', '3 steps'),
]

# Blues for low band, reds/oranges for high band
low_colours = ['#0D47A1', '#1565C0', '#1E88E5', '#42A5F5', '#90CAF9']
high_colours = ['#B71C1C', '#D32F2F', '#F44336', '#FF7043', '#FFAB91']

fig, ax = plt.subplots(figsize=(10, 5.5))

def plot_band(slopes, colours):
    for (slope_id, label, steps_note), colour in zip(slopes, colours):
        csv_path = os.path.join(DATA_DIR, f'slope_{slope_id}deg.csv')
        if not os.path.exists(csv_path):
            print(f'  SKIP {label} — not found')
            continue
        with open(csv_path) as f:
            rows = [r for r in csv.DictReader(f) if r['base_x_m'] != '']
        t = np.array([float(r['sim_time_s']) for r in rows])
        x = np.array([float(r['base_x_m']) for r in rows])
        fall = np.array([int(r['fall_detected']) for r in rows])
        fall_idx = np.argmax(fall) if fall.any() else len(fall) - 1
        t_plot = t[:fall_idx + 1]
        x_plot = x[:fall_idx + 1]
        ax.plot(t_plot, x_plot, color=colour, linewidth=2.0,
                label=f'{label}  ({steps_note})', zorder=3)
        if fall.any():
            ax.plot(t_plot[-1], x_plot[-1], 'x', color=colour,
                    markersize=10, markeredgewidth=2.5, zorder=4)

plot_band(BAND_LOW, low_colours)
plot_band(BAND_HIGH, high_colours)

ax.plot([], [], 'kx', markersize=9, markeredgewidth=2.5, label='Fall event')

ax.set_xlabel('Simulation Time (s)', fontsize=12, fontweight='bold')
ax.set_ylabel('Forward Distance (m)', fontsize=12, fontweight='bold')
ax.set_title('Walking Stability — Ultra-Fine Slope Sweep', fontsize=14, fontweight='bold')
ax.legend(fontsize=9.5, loc='upper left', framealpha=0.9,
          edgecolor='#cccccc', ncol=2, columnspacing=1.5,
          title='Blue: 2.85°–3.05°    Red: 3.35°–3.55°',
          title_fontsize=9)
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
