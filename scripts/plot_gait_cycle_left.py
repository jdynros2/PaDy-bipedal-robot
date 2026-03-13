#!/usr/bin/env python3
"""Left hip + knee angles for one gait cycle — centred on peak knee flexion.
Convention: hip flexion (+), knee flexion (−).
Cycle extracted so knee trough aligns at 50% of gait cycle."""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
import matplotlib.transforms as mtransforms

CSV = '/home/elec330-admin/ros2_ws/data/run_20260310_153542.csv'
OUT = '/home/elec330-admin/ros2_ws/src/pady_robot/prerequisite+docs/figures/gait_cycle_left_joint_angles.png'

with open(CSV) as f:
    rows = list(csv.DictReader(f))

t  = np.array([float(r['sim_time_s']) for r in rows])
hl = np.degrees(np.array([float(r['hip_left_rad']) for r in rows]))
kl = np.degrees(np.array([float(r['knee_left_rad']) for r in rows]))

# ── Extract one cycle centred on knee peak ──
# Original cycle: t=0.635–2.527 (1.892s), knee peak at t=1.174
# Re-centre: start = 1.174 − 0.946 = 0.228, end = 1.174 + 0.946 = 2.120
c_start = np.argmin(np.abs(t - 0.228))
c_end   = np.argmin(np.abs(t - 2.120))
tc  = t[c_start:c_end+1]

# URDF hip joint origins have built-in pitch offsets:
#   hip_joint_right: rpy="0  0.28 0" → actual = joint + 0.28
#   hip_joint_left:  rpy="0 -0.28 0" → actual = joint − 0.28
# CSV reports raw joint angles — correct to body-relative sagittal angles.
HIP_LEFT_OFFSET = -0.28  # rad (from URDF joint origin rpy)
hip_plot  = hl[c_start:c_end+1] + np.degrees(HIP_LEFT_OFFSET)  # corrected, degrees
knee_plot = -kl[c_start:c_end+1]         # knee flexion = NEGATIVE (no offset needed)

pct = (tc - tc[0]) / (tc[-1] - tc[0]) * 100

# ── PaDy phase boundaries (from corrected hip kinematics, re-centred) ──
# 0-22%:  Swing (hip flexing from −14° → +15°, knee locked at 0°)
# 22-28%: DS (heel strike, initial contact, small knee flex)
# 28-62%: Stance (knee loading response, peak −35° at 50%)
# 62-68%: DS (pre-swing, hip crossing zero)
# 68-100%: Swing (hip extending to −18°, knee locked at 0°)
pady_phases = [
    (0,   22,   'Swing',   '#ffcc80'),
    (22,  28,   'DS',      '#b0bec5'),
    (28,  62,   'Stance',  '#a5d6a7'),
    (62,  68,   'DS',      '#b0bec5'),
    (68, 100,   'Swing',   '#ffcc80'),
]

# ── Human reference curves (Winter 2009, shifted to align knee troughs) ──
# Standard human knee trough at ~72% cycle. We want it at 50%.
# Phase shift = −22% of cycle.
pct_h = np.linspace(0, 100, 101)
shift = -22  # shift human curves left by 22% to align knee troughs
pct_shifted = (pct_h - shift) % 100
# Sort for clean plotting
order = np.argsort(pct_shifted)
pct_shifted_sorted = pct_shifted[order]

human_hip_raw = 20 * np.cos(2 * np.pi * pct_h / 100 + np.radians(10)) + 5
human_knee_load  = -15 * np.exp(-((pct_h - 12)**2) / (2 * 5**2))
human_knee_swing = -60 * np.exp(-((pct_h - 72)**2) / (2 * 8**2))
human_knee_raw = human_knee_load + human_knee_swing - 5

# Resample shifted curves at uniform spacing
human_hip  = np.interp(np.linspace(0, 100, 101), pct_shifted_sorted, human_hip_raw[order])
human_knee = np.interp(np.linspace(0, 100, 101), pct_shifted_sorted, human_knee_raw[order])
pct_h = np.linspace(0, 100, 101)

# Human phases (shifted by −22%): original DS=0-12, Stance=12-50, DS=50-62, Swing=62-100
# After shift: Swing wraps, DS=0-( ), Stance, DS, Swing
human_phases = [
    (0,   40,  'Swing',   '#ffcc80'),
    (40,  50,  'DS',      '#b0bec5'),
    (50,  72,  'Stance',  '#a5d6a7'),   # was 12-50 + shift → but better:
]
# Actually compute shifted phases properly
# Original: DS 0-12, Stance 12-50, DS 50-62, Swing 62-100
# Shift each by -22: DS -22–-10 → 78-90, Stance -10–28 → wraps, DS 28-40, Swing 40-78
# Reorder for 0-100:
human_phases = [
    (0,   28,  'Stance',  '#a5d6a7'),
    (28,  40,  'DS',      '#b0bec5'),
    (40,  78,  'Swing',   '#ffcc80'),
    (78,  90,  'DS',      '#b0bec5'),
    (90, 100,  'Stance',  '#a5d6a7'),
]

# ── Plot ──
fig, ax = plt.subplots(figsize=(10, 5.5))

# PaDy phase dividers — solid dark lines
for x0, x1, label, color in pady_phases:
    if x0 > 0:
        ax.axvline(x0, color='#2c3e50', lw=1.0, ls='-', zorder=3)

# Human phase dividers — dashed grey lines
for x0, x1, label, color in human_phases:
    if x0 > 0:
        ax.axvline(x0, color='#aaaaaa', lw=0.8, ls='--', zorder=2, alpha=0.5)

# PaDy measured data — solid
ax.plot(pct, hip_plot, '-', color='#2196F3', lw=2.2, label='PaDy Hip (left)')
ax.plot(pct, knee_plot, '-', color='#e74c3c', lw=2.2, label='PaDy Knee (left)')

# Human reference — dashed, transparent
ax.plot(pct_h, human_hip, '--', color='#2196F3', lw=1.3, alpha=0.5,
        label='Human Hip (typical)')
ax.fill_between(pct_h, human_hip - 5, human_hip + 5, color='#2196F3', alpha=0.07)
ax.plot(pct_h, human_knee, '--', color='#e74c3c', lw=1.3, alpha=0.5,
        label='Human Knee (typical)')
ax.fill_between(pct_h, human_knee - 5, human_knee + 5, color='#e74c3c', alpha=0.07)

# Zero line
ax.axhline(0, color='#999', lw=0.5, ls='-')

# Axes
ax.set_ylabel('Joint Angle (°)', fontsize=11)
ax.set_xlabel('Gait Cycle (%)', fontsize=11)
ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.set_xlim(0, 100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, color='#d0d0d0', linewidth=0.4, alpha=0.8)

# ── Coloured phase bars ──
bar_h = 0.04

# PaDy phase bar (above plot) — solid colours
pady_bar_y = 1.02
for x0, x1, label, color in pady_phases:
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    rect = plt.Rectangle((x0, pady_bar_y), x1 - x0, bar_h, transform=trans,
                          facecolor=color, edgecolor='white', lw=0.8,
                          clip_on=False, zorder=5)
    ax.add_patch(rect)
    if (x1 - x0) > 8:
        ax.text((x0 + x1) / 2, pady_bar_y + bar_h / 2, label, transform=trans,
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='#2c3e50', clip_on=False, zorder=6)
ax.text(-2, pady_bar_y + bar_h / 2, 'PaDy',
        transform=mtransforms.blended_transform_factory(ax.transData, ax.transAxes),
        ha='right', va='center', fontsize=8, fontweight='bold',
        color='#2c3e50', clip_on=False)

# Human phase bar (below plot) — lighter/transparent
bar_y = -0.12
for x0, x1, label, color in human_phases:
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    rect = plt.Rectangle((x0, bar_y), x1 - x0, bar_h, transform=trans,
                          facecolor=color, edgecolor='white', lw=0.8,
                          alpha=0.5, clip_on=False, zorder=5)
    ax.add_patch(rect)
    if (x1 - x0) > 8:
        ax.text((x0 + x1) / 2, bar_y + bar_h / 2, label, transform=trans,
                ha='center', va='center', fontsize=8, color='#666666',
                style='italic', clip_on=False, zorder=6)
ax.text(-2, bar_y + bar_h / 2, 'Human',
        transform=mtransforms.blended_transform_factory(ax.transData, ax.transAxes),
        ha='right', va='center', fontsize=8, color='#666666',
        style='italic', clip_on=False)

# Sign convention annotation
ax.annotate('(+) hip flexion\n(−) knee flexion', xy=(0.98, 0.02),
            xycoords='axes fraction', ha='right', va='bottom',
            fontsize=7.5, color='#7f8c8d', style='italic')

# Legend below chart
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=8.5,
          framealpha=0.9)

fig.suptitle('Comparison of Mean Measured Joint Angles — Left Hip, Knee',
             fontsize=13, fontweight='bold', y=1.02)

fig.subplots_adjust(bottom=0.22)
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved → {OUT}')
