# PaDy Meeting Brief (6 Mar 2026)

## Project Status Snapshot

- Current objective: **5 consecutive passive steps**.
- Current workspace evidence:
  - Best observed physical performance: **3 steps max** (`run_20260305_132929.csv`).
  - Current recent performance: last 10 runs average **~1 physical step**.
- Important metric note for supervisor: `gait_analyser` `step_count` is a zero-crossing event count (signal-level), not direct footfall count. In this brief, physical steps are estimated from that signal and capped by visual validation.
- Interpretation to present: the system is producing repeatable early-step motion, but must now be pushed from 2–3 steps to stable 5-step consistency.

---

## Which Launch Files to Use (and Why)

### 1) `launch/spawn_pady.launch.py` (manual tuning)
Use for real-time iterative tuning in Gazebo + RViz.

- Best for quickly checking if a parameter change helps/hurts gait.
- Main tuning knobs (CLI):
  - `kick_torque`
  - `hip_push_torque`
  - `body_force`
  - `hip_push_start_time`
  - `hip_push_stop_time`
  - `spawn_pitch`
  - `spawn_x`

### 2) `launch/analysis.launch.py` (formal data collection)
Use for publishable/traceable test runs.

- Adds `gait_analyser.py` and rosbag recording.
- Produces:
  - `data/run_*.csv` (metrics)
  - `data/bag_*/` (topic history)

### 3) `launch/headless.launch.py` (batch/sweep)
Use for repeated benchmark trials and objective comparison.

- Same gait timing as spawn launch.
- No GUI, faster throughput.
- Primary launch target for `param_sweep.py`.

---

## Which Scripts to Use (and Why)

### 1) `scripts/continuous_hip_push.py`
- Runtime assist publisher during configured window.
- Supports gait initiation/energy injection tests.

### 2) `scripts/gait_analyser.py`
- Core metric extraction from each run.
- Outputs step count, gait period, symmetry, fall detection to CSV.

### 3) `scripts/param_sweep.py`
- Automated benchmark execution over parameter grid.
- Use to replace trial-and-error with structured search.
- New quick check mode:
  - `--smoke-csv PATH` for fast metric validation without launching Gazebo.

### 4) `scripts/plot_results.py`
- Converts sweep summary into figures and ranking tables.
- New quick check mode:
  - `--smoke` for parse+summary only.

---

## Three Main Benchmark Tests (for Optimal Gait Parameters)

## Benchmark 1 — Step Count Performance
**Question:** How often do settings achieve high consecutive steps?

- Metric: **estimated physical steps** per run (derived from run CSV and checked against observed behavior).
- Pass trend: distribution shifts from 2–3 toward 4–5.
- Current figure: `data/figures/meeting_benchmark1_step_distribution.png`.

## Benchmark 2 — Progress and Repeatability
**Question:** Are we improving over time and maintaining gains?

- Metric: chronological estimated-step trend + best-so-far envelope.
- Pass trend: recent runs should climb toward and hold 5.
- Current figure: `data/figures/meeting_benchmark2_progress_trend.png`.

## Benchmark 3 — Best-Run Dynamics Quality
**Question:** What joint/height behavior produced the best run?

- Metrics shown: hip angles, knee angles, projected/base height.
- Use as a reference gait signature to match in new runs.
- Current figure: `data/figures/meeting_benchmark3_best_run_dynamics.png`.

---

## Data Products Ready for Meeting

- Figures:
  - `data/figures/meeting_benchmark1_step_distribution.png`
  - `data/figures/meeting_benchmark2_progress_trend.png`
  - `data/figures/meeting_benchmark3_best_run_dynamics.png`
- Summary table source:
  - `data/meeting_run_summary.csv`

Plot generation source (CSV-only):

```bash
python3 src/pady_robot/scripts/make_meeting_plots.py
```

This command reads `data/run_*.csv` directly and regenerates the three meeting figures.

---

## 1.5-Week Completion Plan to 5 Reliable Passive Steps

### Phase A (Days 1–3): Recover from regression
- Lock baseline run command and world/URDF/launch versions.
- Execute 20-run controlled benchmark (fixed settings) to confirm current ceiling.
- Compare with current best-run signature (`run_20260305_132929.csv`, 3-step case).

### Phase B (Days 4–7): Parameter convergence
- Run targeted sweeps around current best neighborhood:
  - Narrow ranges around `kick_torque`, `hip_push_torque`, `spawn_pitch`, `body_force`.
- Use mean step count + variance as selection criteria.
- Keep top 3 candidates only.

### Phase C (Days 8–10): Robustness validation
- For each top candidate, run ≥10 repeats.
- Select final setting by:
  1. highest mean steps,
  2. lowest run-to-run variance,
  3. no immediate side-fall behavior.

### Phase D (Days 11–12): Final evidence package
- Produce final plots + summary table.
- Record one final analysis launch bag and run CSV set for supervisor review.
- Prepare concise final presentation:
  - baseline vs tuned performance,
  - benchmark outcomes,
  - final recommendation.

---

## Suggested Live Demo Flow (Tomorrow)

1. Show current benchmark figures (distribution, trend, best-run dynamics).
2. Explain that earlier CSV `step_count` values were signal counts, and corrected physical-step interpretation shows a max of 3.
3. Show structured 3-benchmark approach and 1.5-week plan.
4. Run one quick smoke check command to show reproducible pipeline:

```bash
python3 src/pady_robot/scripts/param_sweep.py --smoke-csv data/run_20260305_132929.csv --smoke-kick-torque 30
```

5. If time permits, launch one fresh measured run:

```bash
source install/setup.bash
ros2 launch pady_robot analysis.launch.py
```
