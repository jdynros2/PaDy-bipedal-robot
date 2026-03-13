#!/usr/bin/env python3
"""Plot 4: Visual comparison table — PaDy vs Human gait parameters."""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

CSV = '/home/elec330-admin/ros2_ws/data/run_20260310_153542.csv'
OUT = '/home/elec330-admin/ros2_ws/src/pady_robot/prerequisite+docs/figures/gait_comparison_table.png'

with open(CSV) as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader]

# ── Extract PaDy metrics (pre-fall only) ─────────────────────────────────────
fall = [int(r['fall_detected']) for r in rows]
fall_idx = fall.index(1) if 1 in fall else len(fall)
pre = rows[:fall_idx]

hip_r = [float(r['hip_right_rad']) for r in pre]
hip_l = [float(r['hip_left_rad']) for r in pre]
knee_r = [float(r['knee_right_rad']) for r in pre]
knee_l = [float(r['knee_left_rad']) for r in pre]
pitch = [float(r['base_pitch_rad']) for r in pre]
roll = [float(r['base_roll_rad']) for r in pre]
yaw = [float(r['base_yaw_rad']) for r in pre]
y_pos = [float(r['base_y_m']) for r in pre]
x_pos = [float(r['base_x_m']) for r in pre]
height = [float(r['slope_height_m']) for r in pre]

# Step periods
steps = [int(r['step_count']) for r in pre]
gp = [float(r['gait_period_s']) for r in pre]
periods = []
for i in range(1, len(steps)):
    if steps[i] != steps[i-1] and gp[i] > 0:
        periods.append(gp[i])

pady_hip_rom = np.degrees(max(max(hip_r), max(hip_l)) - min(min(hip_r), min(hip_l)))
pady_knee_peak = np.degrees(max(max(knee_r), max(knee_l)))
pady_step_period = f'{min(periods):.2f}\u2013{max(periods):.2f}' if periods else 'N/A'
pady_step_mean = np.mean(periods) if periods else 0
pady_cadence = 60.0 / pady_step_mean if pady_step_mean > 0 else 0
pady_stride_len = (x_pos[-1] - x_pos[0]) / (max(steps) / 2) if max(steps) > 0 else 0
pady_com_excursion = (max(height) - min(height)) * 100  # cm
pady_pitch_range = np.degrees(max(pitch) - min(pitch))
pady_roll_range = np.degrees(max(roll) - min(roll))
pady_yaw_range = np.degrees(max(yaw) - min(yaw))
pady_lat_drift = abs(y_pos[-1] - y_pos[0])

# Symmetry index
hr = np.array(hip_r); hl = np.array(hip_l)
sym_idx = abs(np.mean(np.abs(hr)) - np.mean(np.abs(hl))) / (np.mean(np.abs(hr) + np.abs(hl)) / 2) * 100

# ── Table data ────────────────────────────────────────────────────────────────
params = [
    'Hip ROM',
    'Knee Peak Flexion',
    'Step Period',
    'Cadence',
    'Stride Length',
    'CoM Vertical Excursion',
    'Hip Body Pitch Oscillation',
    'Hip Body Roll Oscillation',
    'Yaw Drift (total)',
    'Lateral Drift',
    'Symmetry Index',
    'Steps Before Fall',
]

human_vals = [
    '~40\u00b0',
    '~60\u00b0',
    '0.50\u20130.60 s',
    '100\u2013120 steps/min',
    '~1.4 m',
    '~5 cm',
    '\u00b12\u20134\u00b0',
    '\u00b13\u20135\u00b0',
    '\u2248 0\u00b0',
    '\u2248 0 m',
    '< 5%',
    '\u221e (stable)',
]

pady_vals = [
    f'{pady_hip_rom:.1f}\u00b0',
    f'{pady_knee_peak:.1f}\u00b0',
    pady_step_period + ' s',
    f'{pady_cadence:.0f} steps/min',
    f'{pady_stride_len:.2f} m',
    f'{pady_com_excursion:.1f} cm',
    f'\u00b1{pady_pitch_range/2:.1f}\u00b0',
    f'\u00b1{pady_roll_range/2:.1f}\u00b0',
    f'{pady_yaw_range:.1f}\u00b0',
    f'{pady_lat_drift:.2f} m',
    f'{sym_idx:.0f}%',
    f'{max(steps)}',
]

# 0=green, 1=yellow, 2=red
ratings = [2, 1, 2, 1, 2, 0, 2, 2, 2, 2, 2, 2]

GREEN  = '#27ae60'
YELLOW = '#f39c12'
RED    = '#e74c3c'
rating_colors = [GREEN, YELLOW, RED]

# Row background tints — light wash of the rating colour
rating_tints = ['#eafaf1', '#fef9e7', '#fdedec']

HEADER_BG = '#2c3e50'
ROW_EVEN  = '#ffffff'
ROW_ODD   = '#f7f9fb'

# ── Create figure ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.axis('off')

col_labels = ['Parameter', 'Human', 'PaDy (Best Run)', 'Rating']
n = len(params)

# Cell colours — alternate rows, rating column gets the traffic colour
cell_colours = []
for i in range(n):
    base = ROW_EVEN if i % 2 == 0 else ROW_ODD
    row = [base, base, base, rating_tints[ratings[i]]]
    cell_colours.append(row)

table_data = [[params[i], human_vals[i], pady_vals[i], ''] for i in range(n)]

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellColours=cell_colours,
    colColours=[HEADER_BG] * 4,
    loc='center',
    cellLoc='center',
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.55)

# Header styling
for j in range(4):
    cell = table[0, j]
    cell.set_text_props(color='white', fontweight='bold', fontsize=11)
    cell.set_edgecolor(HEADER_BG)
    cell.set_height(0.06)

# Column widths
for i in range(n + 1):
    table[i, 0].set_width(0.30)
    table[i, 1].set_width(0.20)
    table[i, 2].set_width(0.20)
    table[i, 3].set_width(0.10)

# Row styling
for i in range(1, n + 1):
    # Parameter column — left-aligned, bold
    table[i, 0].set_text_props(ha='left', fontweight='bold', fontsize=10)
    # Data columns
    table[i, 1].set_text_props(fontsize=10)
    table[i, 2].set_text_props(fontsize=10, fontweight='bold')
    # Rating column — coloured square drawn manually
    table[i, 3].set_text_props(fontsize=10)

    # Light cell borders
    for j in range(4):
        table[i, j].set_edgecolor('#dee2e6')
        table[i, j].set_linewidth(0.5)

# ── Draw coloured squares in rating column ────────────────────────────────────
fig.canvas.draw()  # need this to get cell positions
renderer = fig.canvas.get_renderer()

for i in range(1, n + 1):
    cell = table[i, 3]
    bbox = cell.get_window_extent(renderer)
    # Convert to figure coords
    inv = fig.transFigure.inverted()
    p0 = inv.transform((bbox.x0, bbox.y0))
    p1 = inv.transform((bbox.x1, bbox.y1))
    cx = (p0[0] + p1[0]) / 2
    cy = (p0[1] + p1[1]) / 2
    sq_size = min(p1[0] - p0[0], p1[1] - p0[1]) * 0.45

    colour = rating_colors[ratings[i - 1]]
    rect = FancyBboxPatch(
        (cx - sq_size/2, cy - sq_size/2), sq_size, sq_size,
        boxstyle='round,pad=0.002',
        facecolor=colour, edgecolor='white', linewidth=1.5,
        transform=fig.transFigure, zorder=10,
    )
    fig.patches.append(rect)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title('Gait Parameter Comparison: PaDy vs Human',
             fontsize=14, fontweight='bold', pad=20)

# ── Legend — coloured squares with labels ─────────────────────────────────────
legend_items = [
    (GREEN,  'Within human range'),
    (YELLOW, 'Moderate deviation'),
    (RED,    'Significant deviation'),
]

legend_x = 0.28
legend_y = -0.03
sq = 0.018  # square size in axes fraction

for colour, label in legend_items:
    rect = FancyBboxPatch(
        (legend_x, legend_y - sq/2), sq, sq,
        boxstyle='round,pad=0.002',
        facecolor=colour, edgecolor='white', linewidth=1.5,
        transform=ax.transAxes, zorder=10,
    )
    ax.add_patch(rect)
    ax.text(legend_x + sq + 0.01, legend_y, label, fontsize=9,
            transform=ax.transAxes, va='center')
    legend_x += 0.22

fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved \u2192 {OUT}')
