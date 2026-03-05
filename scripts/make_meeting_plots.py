#!/usr/bin/env python3
"""Generate meeting-ready benchmark plots directly from run_*.csv files.

This script uses only CSV contents (plus filename ordering for chronology).
No rosbag parsing or external dependencies beyond matplotlib/numpy.

Outputs (default):
  ~/ros2_ws/data/figures/meeting_benchmark1_step_distribution.png
  ~/ros2_ws/data/figures/meeting_benchmark2_progress_trend.png
  ~/ros2_ws/data/figures/meeting_benchmark3_best_run_dynamics.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class RunSummary:
    file_name: str
    max_step_count: int
    fall_time_s: float


def _to_float(value: str | None, default: float = float("nan")) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _summarize_run(path: str) -> RunSummary | None:
    rows = _read_rows(path)
    if not rows:
        return None

    step_values = [int(_to_float(row.get("step_count"), 0.0)) for row in rows]
    max_steps = max(step_values) if step_values else 0

    fall_time = float("nan")
    for row in rows:
        if _to_float(row.get("fall_detected"), 0.0) >= 1.0:
            fall_time = _to_float(row.get("sim_time_s"))
            break

    return RunSummary(file_name=os.path.basename(path), max_step_count=max_steps, fall_time_s=fall_time)


def plot_step_distribution(runs: list[RunSummary], out_path: str) -> None:
    vals = [r.max_step_count for r in runs]
    minimum = min(vals)
    maximum = max(vals)

    plt.figure(figsize=(7, 4.2))
    plt.hist(vals, bins=list(range(minimum, maximum + 2)), align="left", rwidth=0.75, color="#2563eb")
    plt.xticks(range(minimum, maximum + 1))
    plt.xlabel("CSV max step_count per run")
    plt.ylabel("Run count")
    plt.title("Benchmark 1: Step-count distribution (from CSV)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_progress_trend(runs: list[RunSummary], out_path: str) -> None:
    y = [r.max_step_count for r in runs]
    x = list(range(len(y)))
    best_so_far = []
    current_best = 0
    for value in y:
        current_best = max(current_best, value)
        best_so_far.append(current_best)

    plt.figure(figsize=(9, 4.6))
    plt.plot(x, y, marker="o", markersize=3, linewidth=1, color="#94a3b8", label="CSV max step_count")
    plt.plot(x, best_so_far, linewidth=2.2, color="#16a34a", label="Best-so-far envelope")
    plt.axhline(5, color="#ef4444", linestyle="--", linewidth=1.6, label="Target: 5 consecutive passive steps")
    plt.xlabel("Run index (chronological by filename)")
    plt.ylabel("CSV max step_count")
    plt.title("Benchmark 2: Progress trend (from CSV)")
    plt.ylim(bottom=0)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_best_run_dynamics(best_csv_path: str, out_path: str) -> None:
    rows = _read_rows(best_csv_path)
    if not rows:
        return

    t = [_to_float(r.get("sim_time_s"), 0.0) for r in rows]
    hip_r = [_to_float(r.get("hip_right_rad")) for r in rows]
    hip_l = [_to_float(r.get("hip_left_rad")) for r in rows]
    knee_r = [_to_float(r.get("knee_right_rad")) for r in rows]
    knee_l = [_to_float(r.get("knee_left_rad")) for r in rows]

    height_col = "slope_height_m" if "slope_height_m" in rows[0] else "base_height_m"
    height = [_to_float(r.get(height_col)) for r in rows]

    max_steps = max(int(_to_float(r.get("step_count"), 0.0)) for r in rows)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(t, hip_r, label="hip_right", color="#2563eb")
    axes[0].plot(t, hip_l, label="hip_left", color="#60a5fa", linestyle="--")
    axes[0].set_ylabel("Hip angle (rad)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, knee_r, label="knee_right", color="#16a34a")
    axes[1].plot(t, knee_l, label="knee_left", color="#86efac", linestyle="--")
    axes[1].set_ylabel("Knee angle (rad)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, height, color="#9333ea")
    axes[2].set_ylabel("Height (m)")
    axes[2].set_xlabel("Simulation time (s)")
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        f"Benchmark 3: Best-run dynamics ({os.path.basename(best_csv_path)}, csv_step_count={max_steps})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate meeting plots from run CSV files")
    parser.add_argument(
        "--run-glob",
        default="/home/elec330-admin/ros2_ws/data/run_*.csv",
        help="Glob for input run CSV files",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/elec330-admin/ros2_ws/data/figures",
        help="Output directory for PNG files",
    )
    args = parser.parse_args()

    run_paths = sorted(glob.glob(os.path.expanduser(args.run_glob)))
    if not run_paths:
        print("No run CSV files found.")
        return 1

    os.makedirs(os.path.expanduser(args.out_dir), exist_ok=True)

    summaries: list[RunSummary] = []
    for path in run_paths:
        summary = _summarize_run(path)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        print("No non-empty run CSV files found.")
        return 1

    out1 = os.path.join(args.out_dir, "meeting_benchmark1_step_distribution.png")
    out2 = os.path.join(args.out_dir, "meeting_benchmark2_progress_trend.png")
    out3 = os.path.join(args.out_dir, "meeting_benchmark3_best_run_dynamics.png")

    plot_step_distribution(summaries, out1)
    plot_progress_trend(summaries, out2)

    best_summary = max(summaries, key=lambda r: r.max_step_count)
    best_csv_path = os.path.join("/home/elec330-admin/ros2_ws/data", best_summary.file_name)
    plot_best_run_dynamics(best_csv_path, out3)

    print("Created:")
    print(out1)
    print(out2)
    print(out3)
    print(f"Best CSV by step_count: {best_summary.file_name} (step_count={best_summary.max_step_count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
