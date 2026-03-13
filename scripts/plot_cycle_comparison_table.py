#!/usr/bin/env python3
"""Comparison table: PaDy single gait cycle metrics vs human gait norms."""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

CSV = '/home/elec330-admin/ros2_ws/data/run_20260310_153542.csv'
OUT = '/home/elec330-admin/ros2_ws/src/pady_robot/prerequisite+docs/figures/cycle_comparison_table.png'

with open(CSV) as f:
    rows = list(csv.DictReader(f))

t  = np.array([float(r['sim_time_s']) for r in rows])
hr = np.degrees(np.array([float(r['hip_right_rad']) for r in rows]))
hl = np.degrees(np.array([float(r['hip_left_rad']) for r in rows]))
kr = np.degrees(np.array([float(r['knee_right_rad']) for r in rows]))
kl = np.degrees(np.array([float(r['knee_left_rad']) for r in rows]))
roll  = np.degrees(np.array([float(r['base_roll_rad']) for r in rows]))
pitch = np.degrees(np.array([float(r['base_pitch_rad']) for r in rows]))
yaw   = np.degrees(np.array([float(r['base_yaw_rad']) for r in rows]))
sc = np.array([int(float(r['step_count'])) for r in rows])

# ── Extract cycle (t≈1.279 to t≈3.253) ──
c_start = np.argmin(np.abs(t - 1.279))
c_end   = np.argmin(np.abs(t - 3.253))

tc = t[c_start:c_end+1]
hrc = hr[c_start:c_end+1]
hlc = hl[c_start:c_end+1]
krc = kr[c_start:c_end+1]
klc = kl[c_start:c_end+1]
rollc  = roll[c_start:c_end+1]
pitchc = pitch[c_start:c_end+1]
yawc   = yaw[c_start:c_end+1]

cycle_dur = tc[-1] - tc[0]

# ── Compute PaDy metrics for this cycle ──
hip_rom_r = np.max(hrc) - np.min(hrc)
hip_rom_l = np.max(hlc) - np.min(hlc)
hip_rom_avg = (hip_rom_r + hip_rom_l) / 2

knee_peak_r = np.max(krc)
knee_peak_l = np.max(klc)
knee_peak_avg = (knee_peak_r + knee_peak_l) / 2

hip_peak_flex = max(np.max(hrc), np.max(hlc))   # max forward (positive)
hip_peak_ext  = min(np.min(hrc), np.min(hlc))    # max backward (negative)

knee_stance_r = np.mean(krc[:len(krc)//2])  # first half ≈ stance
knee_stance_l = np.mean(klc[:len(klc)//2])

# Symmetry: difference between R and L ROM
hip_symmetry = abs(hip_rom_r - hip_rom_l)

# Step period (half cycle)
step_period = cycle_dur / 2

# Trunk stability
roll_range = np.max(rollc) - np.min(rollc)
pitch_range = np.max(pitchc) - np.min(pitchc)
yaw_range = np.max(yawc) - np.min(yawc)

# Cadence (steps/min) from step period
cadence = 60.0 / step_period

# Gait cycle time
stance_pct = 77.0   # from our phase analysis (DS1+Stance = 0-77%)
swing_pct  = 100 - 85.0  # swing starts at 85%
ds_pct     = 14.0 + (85.0 - 77.0)  # DS1 + DS2

# ── Table data ──
# (Parameter, Human Value, PaDy Value, Rating)
# Rating: 'green' = good match, 'yellow' = partial, 'red' = poor
data = [
    ('Gait Cycle Duration',    '1.0 – 1.2 s',       f'{cycle_dur:.2f} s',       'yellow'),
    ('Step Period',            '0.5 – 0.6 s',       f'{step_period:.2f} s',     'yellow'),
    ('Cadence',                '100 – 120 steps/min', f'{cadence:.0f} steps/min', 'red'),
    ('Hip ROM (avg)',          '35 – 45°',           f'{hip_rom_avg:.1f}°',      'yellow'),
    ('Hip Peak Flexion',      '25 – 35°',           f'{hip_peak_flex:.1f}°',    'red'),
    ('Hip Peak Extension',    '−10 to −20°',        f'{hip_peak_ext:.1f}°',     'yellow'),
    ('Knee Peak Flexion',     '55 – 65° (swing)',   f'{knee_peak_avg:.1f}°',    'yellow'),
    ('Knee Stance Flexion',   '10 – 20°',           f'{(knee_stance_r+knee_stance_l)/2:.1f}°', 'red'),
    ('Hip L–R Symmetry',      '< 3° diff',          f'{hip_symmetry:.1f}° diff', 'yellow'),
    ('Stance Phase',          '60 – 62%',           f'{stance_pct:.0f}%',       'red'),
    ('Swing Phase',           '38 – 40%',           f'{swing_pct:.0f}%',        'red'),
    ('Double Support',        '20 – 22%',           f'{ds_pct:.0f}%',           'yellow'),
    ('Pelvis Roll Range',     '6 – 10°',            f'{roll_range:.1f}°',       'green' if 4 < roll_range < 14 else 'yellow'),
    ('Pelvis Pitch Range',    '4 – 8°',             f'{pitch_range:.1f}°',      'green' if 2 < pitch_range < 12 else 'yellow'),
    ('Pelvis Yaw Range',      '< 10°',              f'{yaw_range:.1f}°',        'red' if yaw_range > 20 else ('yellow' if yaw_range > 10 else 'green')),
]

n_rows = len(data)
n_cols = 4  # Parameter | Human | PaDy | Rating

# ── Plot ──
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, n_rows + 1.5)
ax.axis('off')

# Column positions
col_x = [0.2, 3.5, 6.2, 8.8]
col_labels = ['Parameter', 'Human', 'PaDy (Step 2)', 'Rating']
col_align = ['left', 'center', 'center', 'center']

# Header
header_y = n_rows + 0.8
ax.add_patch(FancyBboxPatch((0, header_y - 0.35), 10, 0.7,
             boxstyle='round,pad=0.05', facecolor='#2c3e50', edgecolor='none'))
for j, (x, label) in enumerate(zip(col_x, col_labels)):
    ax.text(x, header_y, label, ha=col_align[j], va='center',
            fontsize=10, fontweight='bold', color='white')

# Rating colours
rating_colors = {'green': '#27ae60', 'yellow': '#f39c12', 'red': '#e74c3c'}
rating_labels = {'green': 'Good', 'yellow': 'Partial', 'red': 'Poor'}

# Data rows
for i, (param, human, pady, rating) in enumerate(data):
    y = n_rows - i - 0.2
    # Alternating row background
    if i % 2 == 0:
        ax.add_patch(FancyBboxPatch((0, y - 0.35), 10, 0.7,
                     boxstyle='round,pad=0.02', facecolor='#f8f9fa', edgecolor='none'))

    ax.text(col_x[0], y, param, ha='left', va='center', fontsize=9)
    ax.text(col_x[1], y, human, ha='center', va='center', fontsize=9, color='#2c3e50')
    ax.text(col_x[2], y, pady, ha='center', va='center', fontsize=9,
            fontweight='bold', color='#2c3e50')

    # Rating square
    sq_size = 0.22
    ax.add_patch(FancyBboxPatch((col_x[3] - sq_size/2, y - sq_size/2),
                 sq_size, sq_size, boxstyle='round,pad=0.02',
                 facecolor=rating_colors[rating], edgecolor='none', zorder=3))

# Legend below
legend_y = -0.8
legend_items = [('Good Match', '#27ae60'), ('Partial Match', '#f39c12'), ('Poor Match', '#e74c3c')]
total_width = len(legend_items) * 2.5
start_x = 5 - total_width / 2
for k, (label, color) in enumerate(legend_items):
    lx = start_x + k * 3.0
    ax.add_patch(FancyBboxPatch((lx, legend_y - 0.12), 0.25, 0.25,
                 boxstyle='round,pad=0.02', facecolor=color, edgecolor='none'))
    ax.text(lx + 0.4, legend_y, label, ha='left', va='center', fontsize=8.5)

fig.suptitle('Single Gait Cycle Comparison — PaDy vs Human Gait',
             fontsize=13, fontweight='bold', y=0.97)
ax.text(5, n_rows + 1.3, f'Cycle: t = {tc[0]:.2f} – {tc[-1]:.2f} s  (Duration: {cycle_dur:.2f} s)',
        ha='center', va='center', fontsize=9, color='#7f8c8d', style='italic')

fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved → {OUT}')
