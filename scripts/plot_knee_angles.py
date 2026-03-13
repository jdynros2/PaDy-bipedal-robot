#!/usr/bin/env python3
"""Plot 2: Knee flexion angles vs time with human gait reference band."""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = '/home/elec330-admin/ros2_ws/data/run_20260310_153542.csv'
OUT = '/home/elec330-admin/ros2_ws/src/pady_robot/prerequisite+docs/figures/knee_angles_vs_time.png'

# ── Load CSV ──────────────────────────────────────────────────────────────────
with open(CSV) as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader]

t      = np.array([float(r['sim_time_s']) for r in rows])
knee_r = np.array([float(r['knee_right_rad']) for r in rows])
knee_l = np.array([float(r['knee_left_rad']) for r in rows])
steps  = np.array([int(r['step_count']) for r in rows])
fall   = np.array([int(r['fall_detected']) for r in rows])

knee_r_deg = np.degrees(knee_r)
knee_l_deg = np.degrees(knee_l)

# Trim to pre-fall
fall_idx = np.argmax(fall) if fall.any() else len(fall)
t = t[:fall_idx]; knee_r_deg = knee_r_deg[:fall_idx]; knee_l_deg = knee_l_deg[:fall_idx]
steps = steps[:fall_idx]

# Real stride boundaries — full stride = 2 hip zero-crossings
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
fig, ax = plt.subplots(figsize=(10, 5))

# Human reference: knee flexes 0-60° during swing, ~5-15° during stance
ax.axhspan(0, 60, color='#2ecc71', alpha=0.12, label='Human knee swing ROM (0\u201360\u00b0)')
ax.axhline(0, color='grey', lw=0.5, ls='--')

# PaDy knee joint limit
ax.axhline(37.3, color='#e74c3c', lw=0.8, ls=':', alpha=0.6, label='PaDy joint limit (37.3\u00b0)')

# Knee angles
ax.plot(t, knee_r_deg, color='#3498db', lw=1.8, label='Right knee')
ax.plot(t, knee_l_deg, color='#e67e22', lw=1.8, label='Left knee')

# Step markers — only full strides
for st, lbl in stride_labels:
    ax.axvline(st, color='#2c3e50', lw=1.2, ls='--', alpha=0.7)
    ax.text(st, 62, lbl, ha='center', va='bottom', fontsize=9, color='#2c3e50',
            fontweight='bold')

ax.set_xlabel('Simulation Time (s)', fontsize=11)
ax.set_ylabel('Knee Flexion (degrees)', fontsize=11)
ax.set_title('PaDy Knee Flexion vs Human Reference', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.set_xlim(t[0], t[-1])
ax.set_ylim(-2, 65)
ax.grid(True, which='major', axis='both', color='#d0d0d0', lw=0.4, ls='-')
ax.minorticks_off()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved → {OUT}')
