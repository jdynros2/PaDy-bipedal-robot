#!/usr/bin/env python3
"""Plot three-run comparison triplets from run_*.csv files.

For each triplet, this script creates one figure with two subplots:
1) step_count vs time
2) base/slope height vs time

Each of the three runs is drawn in a different colour.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2563eb", "#dc2626", "#16a34a"]


def _to_float(value: str | None, default: float = float("nan")) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_run(path: str):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    t = [_to_float(r.get("sim_time_s"), 0.0) for r in rows]
    steps = [_to_float(r.get("step_count"), 0.0) for r in rows]

    height_col = "slope_height_m" if "slope_height_m" in rows[0] else "base_height_m"
    height = [_to_float(r.get(height_col)) for r in rows]

    return {
        "file": os.path.basename(path),
        "t": t,
        "steps": steps,
        "height": height,
        "height_col": height_col,
    }


def _parse_triplet(value: str) -> list[str]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if len(items) != 3:
        raise argparse.ArgumentTypeError("Triplet must contain exactly 3 CSV paths, comma-separated.")
    return items


def _parse_labels(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    if len(items) != 3:
        raise argparse.ArgumentTypeError("Labels must contain exactly 3 items, comma-separated.")
    return items


def plot_triplet(name: str, csv_paths: list[str], out_dir: str, labels: list[str] | None = None) -> str:
    runs = []
    for p in csv_paths:
        data = _read_run(os.path.expanduser(p))
        if data is None:
            raise ValueError(f"Run CSV is empty or invalid: {p}")
        runs.append(data)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for idx, run in enumerate(runs):
        color = COLORS[idx % len(COLORS)]
        label = labels[idx] if labels is not None else run["file"]
        axes[0].plot(run["t"], run["steps"], color=color, linewidth=2.0, label=label)
        axes[1].plot(run["t"], run["height"], color=color, linewidth=1.8, label=label)

        if run["t"] and run["steps"]:
            final_t = run["t"][-1]
            max_step = max(run["steps"])
            axes[0].annotate(
                f"max={int(max_step)}",
                xy=(final_t, run["steps"][-1]),
                xytext=(6, 0),
                textcoords="offset points",
                color=color,
                fontsize=8,
                va="center",
            )

    winner_idx = max(range(len(runs)), key=lambda i: max(runs[i]["steps"]) if runs[i]["steps"] else float("-inf"))
    winner_label = labels[winner_idx] if labels is not None else runs[winner_idx]["file"]
    winner_max = int(max(runs[winner_idx]["steps"])) if runs[winner_idx]["steps"] else 0

    axes[0].set_title(f"{name}: step_count progression")
    axes[0].set_ylabel("step_count (CSV signal)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper left")

    height_label = runs[0]["height_col"]
    axes[1].set_title(f"{name}: {height_label} stability")
    axes[1].set_ylabel("height (m)")
    axes[1].set_xlabel("simulation time (s)")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"Winner: {winner_label} (max step_count={winner_max})", fontsize=11, y=0.99)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"triplet_{name.lower().replace(' ', '_')}.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot 3-run CSV triplet comparisons")
    parser.add_argument(
        "--slope",
        type=_parse_triplet,
        required=True,
        help="Three CSV files for slope benchmark (comma-separated)",
    )
    parser.add_argument(
        "--pitch",
        type=_parse_triplet,
        required=True,
        help="Three CSV files for spawn_pitch benchmark (comma-separated)",
    )
    parser.add_argument(
        "--param",
        type=_parse_triplet,
        required=True,
        help="Three CSV files for selected-parameter benchmark (comma-separated)",
    )
    parser.add_argument(
        "--slope-labels",
        type=_parse_labels,
        default=None,
        help="Three legend labels for slope runs (comma-separated)",
    )
    parser.add_argument(
        "--pitch-labels",
        type=_parse_labels,
        default=None,
        help="Three legend labels for spawn_pitch runs (comma-separated)",
    )
    parser.add_argument(
        "--param-labels",
        type=_parse_labels,
        default=None,
        help="Three legend labels for selected-parameter runs (comma-separated)",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/elec330-admin/ros2_ws/data/figures",
        help="Directory to save PNG outputs",
    )
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    out1 = plot_triplet("Slope", args.slope, out_dir, args.slope_labels)
    out2 = plot_triplet("Spawn Pitch", args.pitch, out_dir, args.pitch_labels)
    out3 = plot_triplet("Selected Param", args.param, out_dir, args.param_labels)

    print("Created:")
    print(out1)
    print(out2)
    print(out3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
