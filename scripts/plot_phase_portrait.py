#!/usr/bin/env python3
"""Plot 5: Hip phase portrait (angle vs angular velocity) — shows limit cycle stability."""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = '/home/elec330-admin/ros2_ws/data/run_20260310_153542.csv'
OUT = '/home/elec330-admin/ros2_ws/src/pady_robot/prerequisite+docs/figures/hip_phase_portrait.png'

with open(CSV) as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader]

t     = np.array([float(r['sim_time_s']) for r in rows])
hip_r = np.degrees(np.array([float(r['hip_right_rad']) for r in rows]))
hip_l = np.degrees(np.array([float(r['hip_left_rad']) for r in rows]))
fall  = np.array([int(r['fall_detected']) for r in rows])

# Trim to pre-fall
fall_idx = np.argmax(fall) if fall.any() else len(fall)
t = t[:fall_idx]; hip_r = hip_r[:fall_idx]; hip_l = hip_l[:fall_idx]

# Compute angular velocity (finite differences)
dt = np.diff(t)
dt[dt < 1e-6] = 1e-6  # avoid division by zero
vel_r = np.append(np.diff(hip_r) / dt, 0)
vel_l = np.append(np.diff(hip_l) / dt, 0)

# Smooth velocity with a small moving average
def smooth(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode='same')

vel_r = smooth(vel_r)
vel_l = smooth(vel_l)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Colour by time progression
cmap = plt.cm.viridis
norm = plt.Normalize(t[0], t[-1])

# Right hip
for i in range(len(t) - 1):
    ax1.plot(hip_r[i:i+2], vel_r[i:i+2], color=cmap(norm(t[i])), lw=1.2)
ax1.set_xlabel('Hip Angle (°)', fontsize=11)
ax1.set_ylabel('Angular Velocity (°/s)', fontsize=11)
ax1.set_title('Right Hip Phase Portrait', fontsize=12, fontweight='bold')
ax1.axhline(0, color='grey', lw=0.5, ls='--')
ax1.axvline(0, color='grey', lw=0.5, ls='--')
ax1.grid(True, alpha=0.3)

# Annotate start/end
ax1.plot(hip_r[0], vel_r[0], 'o', color='green', ms=8, zorder=5, label='Start')
ax1.plot(hip_r[-1], vel_r[-1], 's', color='red', ms=8, zorder=5, label='Pre-fall')

# Left hip
for i in range(len(t) - 1):
    ax2.plot(hip_l[i:i+2], vel_l[i:i+2], color=cmap(norm(t[i])), lw=1.2)
ax2.set_xlabel('Hip Angle (°)', fontsize=11)
ax2.set_ylabel('Angular Velocity (°/s)', fontsize=11)
ax2.set_title('Left Hip Phase Portrait', fontsize=12, fontweight='bold')
ax2.axhline(0, color='grey', lw=0.5, ls='--')
ax2.axvline(0, color='grey', lw=0.5, ls='--')
ax2.grid(True, alpha=0.3)

ax2.plot(hip_l[0], vel_l[0], 'o', color='green', ms=8, zorder=5, label='Start')
ax2.plot(hip_l[-1], vel_l[-1], 's', color='red', ms=8, zorder=5, label='Pre-fall')

# Shared legend below both plots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
           ncol=2, fontsize=9, framealpha=0.9)

# Colourbar below legend
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.25, -0.06, 0.50, 0.025])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Simulation Time (s)', fontsize=10)

fig.suptitle('Hip Phase Portraits — Stable Gait Shows Closed Loops',
             fontsize=13, fontweight='bold')
fig.subplots_adjust(bottom=0.15, top=0.90, wspace=0.30)
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved → {OUT}')
