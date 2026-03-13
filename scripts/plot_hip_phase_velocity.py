#!/usr/bin/env python3
"""Hip joint angle-velocity phase portrait for one gait cycle with gait phase annotations."""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

CSV = '/home/elec330-admin/ros2_ws/data/run_20260310_153542.csv'
OUT = '/home/elec330-admin/ros2_ws/src/pady_robot/prerequisite+docs/figures/hip_angle_velocity_phase.png'

with open(CSV) as f:
    rows = list(csv.DictReader(f))

t  = np.array([float(r['sim_time_s']) for r in rows])
hr = np.degrees(np.array([float(r['hip_right_rad']) for r in rows]))
hl = np.degrees(np.array([float(r['hip_left_rad']) for r in rows]))
fall = np.array([int(r['fall_detected']) for r in rows])

# ── Extract cycle (t≈1.279 to t≈3.253) ──
c_start = np.argmin(np.abs(t - 1.279))
c_end   = np.argmin(np.abs(t - 3.253))

tc = t[c_start:c_end+1]
hrc = hr[c_start:c_end+1]
hlc = hl[c_start:c_end+1]

# Also get full run for context (pre-fall)
fall_idx = np.argmax(fall) if fall.any() else len(fall)
t_full = t[:fall_idx]
hr_full = hr[:fall_idx]
hl_full = hl[:fall_idx]

# ── Compute angular velocity (finite differences + smoothing) ──
def compute_velocity(angle, time):
    dt = np.diff(time)
    dt[dt < 1e-6] = 1e-6
    vel = np.append(np.diff(angle) / dt, 0)
    # Smooth with moving average
    w = 5
    return np.convolve(vel, np.ones(w)/w, mode='same')

vel_r_cycle = compute_velocity(hrc, tc)
vel_l_cycle = compute_velocity(hlc, tc)

vel_r_full = compute_velocity(hr_full, t_full)
vel_l_full = compute_velocity(hl_full, t_full)

# ── Gait cycle percentage for colour coding ──
pct = (tc - tc[0]) / (tc[-1] - tc[0]) * 100

# Phase boundaries (% of gait cycle)
phase_bounds = [0, 14, 77, 85, 100]
phase_labels = ['DS₁', 'Stance', 'DS₂', 'Swing']
phase_colors_line = ['#3498db', '#27ae60', '#3498db', '#e67e22']

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for ax, angle_c, vel_c, angle_f, vel_f, side in zip(
        axes, [hrc, hlc], [vel_r_cycle, vel_l_cycle],
        [hr_full, hl_full], [vel_r_full, vel_l_full],
        ['Right', 'Left']):

    # Full run in light grey for context
    ax.plot(angle_f, vel_f, '-', color='#d5d5d5', lw=0.8, zorder=1, label='Full run')

    # Gait cycle colour-coded by phase
    for i in range(len(tc) - 1):
        p = pct[i]
        # Determine phase
        for k in range(len(phase_bounds) - 1):
            if phase_bounds[k] <= p < phase_bounds[k+1]:
                c = phase_colors_line[k]
                break
        ax.plot(angle_c[i:i+2], vel_c[i:i+2], '-', color=c, lw=2.2, zorder=2)

    # Start and end markers
    ax.plot(angle_c[0], vel_c[0], 'o', color='#27ae60', ms=10, zorder=5,
            markeredgecolor='white', markeredgewidth=1.5, label='Heel Strike (start)')
    ax.plot(angle_c[-1], vel_c[-1], 's', color='#e74c3c', ms=10, zorder=5,
            markeredgecolor='white', markeredgewidth=1.5, label='Cycle End')

    # Direction arrow at mid-cycle
    mid = len(angle_c) // 2
    ax.annotate('', xy=(angle_c[mid+1], vel_c[mid+1]),
                xytext=(angle_c[mid], vel_c[mid]),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))

    ax.set_xlabel('Hip Angle (°)', fontsize=11)
    ax.set_ylabel('Angular Velocity (°/s)', fontsize=11)
    ax.set_title(f'{side} Hip', fontsize=12, fontweight='bold')
    ax.axhline(0, color='#999', lw=0.5, ls='--')
    ax.axvline(0, color='#999', lw=0.5, ls='--')
    ax.grid(True, color='#d0d0d0', linewidth=0.4, alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

# Phase legend
from matplotlib.lines import Line2D
phase_handles = []
for label, color in zip(phase_labels, phase_colors_line):
    phase_handles.append(Line2D([0], [0], color=color, lw=2.5, label=label))
phase_handles.append(Line2D([0], [0], marker='o', color='#27ae60', lw=0, ms=8,
                     markeredgecolor='white', markeredgewidth=1, label='Heel Strike'))
phase_handles.append(Line2D([0], [0], marker='s', color='#e74c3c', lw=0, ms=8,
                     markeredgecolor='white', markeredgewidth=1, label='Cycle End'))
phase_handles.append(Line2D([0], [0], color='#d5d5d5', lw=1.5, label='Full Run'))

fig.legend(handles=phase_handles, loc='lower center', bbox_to_anchor=(0.5, -0.02),
           ncol=4, fontsize=8.5, framealpha=0.9)

fig.suptitle('Hip Angle–Velocity Phase Portrait — Single Gait Cycle',
             fontsize=13, fontweight='bold', y=1.02)
fig.subplots_adjust(bottom=0.18, top=0.90, wspace=0.30)
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved → {OUT}')
