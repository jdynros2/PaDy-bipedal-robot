#!/usr/bin/env python3
"""Plot 1: Hip flexion/extension angles vs time with human gait reference band."""

import csv
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

CSV = '/home/elec330-admin/ros2_ws/data/run_20260310_153542.csv'
OUT = '/home/elec330-admin/ros2_ws/src/pady_robot/prerequisite+docs/figures/hip_angles_vs_time.png'

# ── Load CSV ──────────────────────────────────────────────────────────────────
with open(CSV) as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader]

t     = np.array([float(r['sim_time_s']) for r in rows])
hip_r = np.array([float(r['hip_right_rad']) for r in rows])
hip_l = np.array([float(r['hip_left_rad']) for r in rows])
steps = np.array([int(r['step_count']) for r in rows])
fall  = np.array([int(r['fall_detected']) for r in rows])

# Convert to degrees
hip_r_deg = np.degrees(hip_r)
hip_l_deg = np.degrees(hip_l)

# Trim to pre-fall only
fall_idx = np.argmax(fall) if fall.any() else len(fall)
t = t[:fall_idx]
hip_r_deg = hip_r_deg[:fall_idx]
hip_l_deg = hip_l_deg[:fall_idx]
steps = steps[:fall_idx]

# Real stride boundaries — a full stride = 2 hip zero-crossings.
# CSV step_count over-counts; the robot achieves 2 full steps then falls on 3rd.
all_transitions = []
for i in range(1, len(steps)):
    if steps[i] != steps[i-1]:
        all_transitions.append((t[i], steps[i]))

# Full strides complete at even step_counts (2, 4) — each pair of zero-crossings = 1 stride
stride_labels = []
stride_num = 0
for trans_t, sc in all_transitions:
    if sc % 2 == 0:  # full stride boundary
        stride_num += 1
        stride_labels.append((trans_t, f'Step {stride_num}'))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

# Human reference band: ±20° from neutral (typical healthy gait hip ROM)
ax.axhspan(-20, 20, color='#2ecc71', alpha=0.12, label='Human hip ROM envelope (±20°)')
ax.axhline(0, color='grey', lw=0.5, ls='--')

# PaDy hip joint limits
ax.axhline(31.05, color='#e74c3c', lw=0.8, ls=':', alpha=0.6, label='PaDy joint limit (±31°)')
ax.axhline(-31.05, color='#e74c3c', lw=0.8, ls=':', alpha=0.6)

# Hip angles
ax.plot(t, hip_r_deg, color='#3498db', lw=1.8, label='Right hip (stance first)')
ax.plot(t, hip_l_deg, color='#e67e22', lw=1.8, label='Left hip (swing first)')

# Step markers — only full strides
for st, lbl in stride_labels:
    ax.axvline(st, color='#2c3e50', lw=1.2, ls='--', alpha=0.7)
    ax.text(st, 33, lbl, ha='center', va='bottom', fontsize=9, color='#2c3e50',
            fontweight='bold')

ax.set_xlabel('Simulation Time (s)', fontsize=11)
ax.set_ylabel('Hip Angle (degrees)', fontsize=11)
ax.set_title('PaDy Hip Flexion/Extension vs Human Reference', fontsize=13, fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=9, framealpha=0.9)
ax.set_xlim(t[0], t[-1])
ax.grid(True, which='major', axis='both', color='#d0d0d0', lw=0.4, ls='-')
ax.minorticks_off()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved → {OUT}')
