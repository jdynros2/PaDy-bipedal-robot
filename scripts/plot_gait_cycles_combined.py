#!/usr/bin/env python3
"""Combined gait cycle plot: left and right body side-by-side.
Full walking period (pre-first-step to fall), hip offset corrected.
Convention: hip flexion (+), knee flexion (−).
Swing labels aligned with peak knee flexion.
Human overlay at PaDy's cadence, knee peaks aligned."""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
import matplotlib.transforms as mtransforms

CSV = '/home/elec330-admin/ros2_ws/data/run_20260310_153542.csv'
OUT = '/home/elec330-admin/ros2_ws/src/pady_robot/prerequisite+docs/figures/gait_cycles_combined.png'

with open(CSV) as f:
    rows = list(csv.DictReader(f))

t    = np.array([float(r['sim_time_s']) for r in rows])
hl   = np.degrees(np.array([float(r['hip_left_rad'])  for r in rows]))
kl   = np.degrees(np.array([float(r['knee_left_rad']) for r in rows]))
hr   = np.degrees(np.array([float(r['hip_right_rad'])  for r in rows]))
kr   = np.degrees(np.array([float(r['knee_right_rad']) for r in rows]))
fall = np.array([int(r['fall_detected']) for r in rows])

# ── URDF hip offset correction ──
hl_corr = hl + np.degrees(-0.28)   # left hip
hr_corr = hr + np.degrees(+0.28)   # right hip

# ── Trim to pre-fall ──
fall_idx = np.argmax(fall) if fall.any() else len(fall)
t_pf  = t[:fall_idx+1]
hl_pf = hl_corr[:fall_idx+1]
kl_pf = kl[:fall_idx+1]
hr_pf = hr_corr[:fall_idx+1]
kr_pf = kr[:fall_idx+1]

# Normalise to 0–100 %
pct = (t_pf - t_pf[0]) / (t_pf[-1] - t_pf[0]) * 100

# ── Identify knee flexion peaks ──
def find_knee_peaks(k, min_height=8, min_sep=10):
    """Find indices of knee flexion peaks above min_height (degrees)."""
    peaks = []
    for i in range(1, len(k)-1):
        if k[i] > min_height and k[i] >= k[i-1] and k[i] >= k[i+1]:
            if not peaks or (i - peaks[-1]) > min_sep:
                peaks.append(i)
            elif k[i] > k[peaks[-1]]:
                peaks[-1] = i
    return peaks

kr_peaks = find_knee_peaks(kr_pf)  # right knee peak indices
kl_peaks = find_knee_peaks(kl_pf)  # left knee peak indices

# ── Define PaDy gait phases per leg ──
# Swing = centred on knee flexion peak; DS = short transitions; Stance = rest
def build_phases(knee_data, pct_arr, peak_indices, knee_thresh=2.0):
    """Build phase regions: identify swing windows from knee flex, add DS buffers."""
    phases = []
    swing_windows = []

    for pk in peak_indices:
        # Find swing boundaries: where knee > threshold around peak
        start = pk
        while start > 0 and knee_data[start-1] > knee_thresh:
            start -= 1
        end = pk
        while end < len(knee_data)-1 and knee_data[end+1] > knee_thresh:
            end += 1
        swing_windows.append((pct_arr[start], pct_arr[end]))

    # Build phase list
    ds_width = 3.0  # % width of DS bands
    cursor = 0.0
    for sw_start, sw_end in swing_windows:
        if sw_start - ds_width > cursor:
            phases.append((cursor, sw_start - ds_width, 'Stance', '#a5d6a7'))
        if sw_start - ds_width > cursor:
            phases.append((sw_start - ds_width, sw_start, 'DS', '#b0bec5'))
        elif sw_start > cursor:
            phases.append((cursor, sw_start, 'DS', '#b0bec5'))
        phases.append((sw_start, sw_end, 'Swing', '#ffcc80'))
        if sw_end + ds_width < 100:
            phases.append((sw_end, sw_end + ds_width, 'DS', '#b0bec5'))
            cursor = sw_end + ds_width
        else:
            cursor = sw_end

    if cursor < 100:
        phases.append((cursor, 100, 'Stance', '#a5d6a7'))

    return phases

right_phases = build_phases(kr_pf, pct, kr_peaks)
left_phases  = build_phases(kl_pf, pct, kl_peaks)

# ── Human reference curves aligned to PaDy's cadence & knee peaks ──
def make_human_overlay(pct_arr, peak_pcts):
    """Generate human gait curves at PaDy's cadence, knee peaks aligned.
    peak_pcts: list of % positions of PaDy knee peaks."""
    # Determine PaDy cycle period from peak spacing (or use full range if 1 peak)
    if len(peak_pcts) >= 2:
        period = peak_pcts[1] - peak_pcts[0]
    else:
        period = 50.0  # fallback

    # Human swing knee peak is at 72% of standard cycle.
    # Offset so human phase = 72 when pct = first PaDy peak.
    offset = peak_pcts[0] - 0.72 * period

    # Map run percentage → human gait phase (mod 100)
    human_phase = ((pct_arr - offset) / period * 100) % 100

    # Standard Winter 2009 curves
    h_hip = 20 * np.cos(2 * np.pi * human_phase / 100 + np.radians(10)) + 5
    h_knee_load  = -15 * np.exp(-((human_phase - 12)**2) / (2 * 5**2))
    h_knee_swing = -60 * np.exp(-((human_phase - 72)**2) / (2 * 8**2))
    h_knee = h_knee_load + h_knee_swing - 5

    # Human phase regions mapped to run timeline
    human_std_phases = [
        (0,   12,  'DS',      '#b0bec5'),
        (12,  50,  'Stance',  '#a5d6a7'),
        (50,  62,  'DS',      '#b0bec5'),
        (62, 100,  'Swing',   '#ffcc80'),
    ]

    h_phases = []
    # Walk through the run and assign human phases
    for i in range(len(pct_arr)):
        hp = human_phase[i]
        for p0, p1, label, color in human_std_phases:
            if p0 <= hp < p1:
                h_phases.append((label, color))
                break
        else:
            h_phases.append(('Swing', '#ffcc80'))

    # Convert to contiguous regions
    regions = []
    curr_label = h_phases[0][0]
    curr_color = h_phases[0][1]
    start_pct = pct_arr[0]
    for i in range(1, len(h_phases)):
        if h_phases[i][0] != curr_label:
            regions.append((start_pct, pct_arr[i], curr_label, curr_color))
            curr_label = h_phases[i][0]
            curr_color = h_phases[i][1]
            start_pct = pct_arr[i]
    regions.append((start_pct, pct_arr[-1], curr_label, curr_color))

    return h_hip, h_knee, regions

# Compute human overlays
pct_dense = np.linspace(0, 100, 501)

right_peak_pcts = [pct[p] for p in kr_peaks]
left_peak_pcts  = [pct[p] for p in kl_peaks]

h_hip_r, h_knee_r, h_phases_r = make_human_overlay(pct_dense, right_peak_pcts)
h_hip_l, h_knee_l, h_phases_l = make_human_overlay(pct_dense, left_peak_pcts)

# ── Plot ──
fig, (ax_l, ax_r) = plt.subplots(2, 1, figsize=(12, 9), sharey=True, sharex=True)
bar_h = 0.04

for ax, hip, knee, pady_ph, h_hip, h_knee, h_ph, pct_d, side in [
    (ax_l, hl_pf, -kl_pf, left_phases,  h_hip_l, h_knee_l, h_phases_l, pct_dense, 'Left'),
    (ax_r, hr_pf, -kr_pf, right_phases, h_hip_r, h_knee_r, h_phases_r, pct_dense, 'Right'),
]:
    # PaDy phase dividers — solid
    for x0, x1, label, color in pady_ph:
        if x0 > 0:
            ax.axvline(x0, color='#2c3e50', lw=0.8, ls='-', zorder=3, alpha=0.5)

    # Human phase dividers — dashed
    for x0, x1, label, color in h_ph:
        if x0 > 0.5:
            ax.axvline(x0, color='#aaaaaa', lw=0.6, ls='--', zorder=2, alpha=0.4)

    # PaDy data — solid lines
    ax.plot(pct, hip,  '-', color='#2196F3', lw=2.2, label=f'PaDy Hip ({side.lower()})')
    ax.plot(pct, knee, '-', color='#e74c3c', lw=2.2, label=f'PaDy Knee ({side.lower()})')

    # Human reference — dashed
    ax.plot(pct_d, h_hip,  '--', color='#2196F3', lw=1.3, alpha=0.45,
            label='Human Hip (typical)')
    ax.fill_between(pct_d, h_hip - 5, h_hip + 5, color='#2196F3', alpha=0.06)
    ax.plot(pct_d, h_knee, '--', color='#e74c3c', lw=1.3, alpha=0.45,
            label='Human Knee (typical)')
    ax.fill_between(pct_d, h_knee - 5, h_knee + 5, color='#e74c3c', alpha=0.06)

    # Zero line
    ax.axhline(0, color='#999', lw=0.5, ls='-')

    # Axes
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, color='#d0d0d0', linewidth=0.4, alpha=0.8)
    # ── Title above phase bar ──
    ax.set_title(f'PaDy {side} Body', fontsize=11, fontweight='bold', pad=18)

    # ── PaDy phase bar (above plot) ──
    pady_bar_y = 1.02
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for x0, x1, label, color in pady_ph:
        rect = plt.Rectangle((x0, pady_bar_y), x1 - x0, bar_h, transform=trans,
                              facecolor=color, edgecolor='white', lw=0.8,
                              clip_on=False, zorder=5)
        ax.add_patch(rect)
        if (x1 - x0) > 6:
            ax.text((x0 + x1) / 2, pady_bar_y + bar_h / 2, label, transform=trans,
                    ha='center', va='center', fontsize=7, fontweight='bold',
                    color='#2c3e50', clip_on=False, zorder=6)

    # ── Human phase bar (below plot) ──
    h_bar_y = -0.15
    for x0, x1, label, color in h_ph:
        rect = plt.Rectangle((x0, h_bar_y), x1 - x0, bar_h, transform=trans,
                              facecolor=color, edgecolor='white', lw=0.8,
                              alpha=0.5, clip_on=False, zorder=5)
        ax.add_patch(rect)
        if (x1 - x0) > 6:
            ax.text((x0 + x1) / 2, h_bar_y + bar_h / 2, label, transform=trans,
                    ha='center', va='center', fontsize=7, color='#666666',
                    style='italic', clip_on=False, zorder=6)
    ax.text(-2, h_bar_y + bar_h / 2, 'Human',
            transform=trans, ha='right', va='center', fontsize=7,
            color='#666666', style='italic', clip_on=False)

ax_l.set_ylabel('Joint Angle (°)', fontsize=10)
ax_r.set_ylabel('Joint Angle (°)', fontsize=10)
ax_r.set_xlabel('Walking Period (%)', fontsize=10, labelpad=20)
ax_l.yaxis.set_major_locator(MaxNLocator(nbins=8))
plt.setp(ax_l.get_xticklabels(), visible=False)

# Sign convention annotation
ax_r.annotate('(+) hip flexion  (−) knee flexion', xy=(0.98, 0.02),
              xycoords='axes fraction', ha='right', va='bottom',
              fontsize=7, color='#7f8c8d', style='italic')

# Shared legend below both plots
handles, labels = ax_l.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.04),
           ncol=4, fontsize=8.5, framealpha=0.9)

fig.suptitle('Comparison of Measured Joint Angles — Full Walking Period',
             fontsize=13, fontweight='bold', y=1.04)

fig.subplots_adjust(bottom=0.14, hspace=0.50)
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved → {OUT}')
