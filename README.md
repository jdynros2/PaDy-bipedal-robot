# PaDy - Passive Dynamic Bipedal Walker

PaDy is a passive-dynamic biped for slope walking experiments in ROS 2 + Gazebo.
This workspace is set up for fast iteration: manual tuning runs, reproducible analysis runs, and automated parameter sweeps.

## Design Parameters

| Parameter | Value |
|-----------|-------|
| Total height | ~1.017 m |
| Thigh length | 500 mm |
| Shin length | 500 mm |
| Hip width (bearing to bearing) | 215.72 mm |
| Total mass | 3.524 kg |
| Target slope | 3 degrees |

## Arm Behavior (Current)

- Arms are fully passive during simulation.
- No arm force controllers are launched.
- Arms only receive initial offsets at spawn to match contralateral leg offsets.

## Quick Start

```bash
cd ~/ros2_ws
colcon build --packages-select pady_robot
source install/setup.bash

# Best run parameters I found
ros2 launch pady_robot analysis.launch.py   kick_torque:=0.2   kick_torque_right:=-0.2   kick_follow_torque:=0   body_force:=0   hip_push_torque:=0   hip_push_start_time:=0   hip_push_stop_time:=12.0   spawn_x:=0.34   spawn_pitch:=0.42   spawn_roll:=-0.2385   spawn_yaw:=0.1


# Interactive simulation (Gazebo)
ros2 launch pady_robot spawn_pady.launch.py

# Data collection run (adds gait_analyser + rosbag)
ros2 launch pady_robot analysis.launch.py

# Headless run (for sweeps / batch testing)
ros2 launch pady_robot headless.launch.py
```

## Launch Files and When to Use Them

- `spawn_pady.launch.py`: Main manual tuning launch (Gazebo + RViz + gait kick timeline + hip assist).
- `analysis.launch.py`: Wraps `spawn_pady` and adds `gait_analyser` + rosbag recording.
- `headless.launch.py`: Server-only Gazebo for repeatable automated trials.
- `view_urdf.launch.py`: URDF/joint visualization only (no physics).

## Runtime Tuning Parameters (Launch Arguments)

| Parameter | Default | Units | Meaning |
|-----------|---------|-------|---------|
| `kick_torque` | 30.0 | N·m | Burst torque used for left and right hip kick events |
| `hip_push_torque` | 5.0 | N·m | Continuous assist torque during configured window |
| `body_force` | 20.0 | N | Forward force command on base link |
| `hip_push_start_time` | 7.0 | s | Assist activation start threshold (sim time) |
| `hip_push_stop_time` | 12.0 | s | Assist deactivation time (sim time) |
| `spawn_pitch` | 0.275 | rad | Initial forward lean |
| `spawn_x` | 0.45 | m | Initial x-position on slope |
| `use_sim_time` | true | bool | Use simulation clock |

Example:

```bash
ros2 launch pady_robot spawn_pady.launch.py \
  kick_torque:=32.0 \
  hip_push_torque:=6.0 \
  body_force:=22.0 \
  hip_push_start_time:=7.0 \
  hip_push_stop_time:=12.5 \
  spawn_pitch:=0.29
```

## Current Gait Timeline (spawn/headless aligned)

- T=5.0s: robot spawn while world is paused
- T=8.0s: world unpause (physics starts)
- T=8.3s: left hip kick start
- T=8.6s: left hip kick stop
- T=9.0s: `continuous_hip_push.py` node starts
- T=9.8s: right hip kick start
- T=10.1s: right hip kick stop
- T=12.0s: safety zero command to both hip kick topics

## Where to Tune What

### 1) Most gait tuning (first place to change)
- Launch arguments in `spawn_pady.launch.py` / `analysis.launch.py` / `headless.launch.py`.

### 2) Contact and physical behavior
- `urdf/pady.urdf` for:
  - foot cant (`foot_joint_*` roll),
  - contact grip/compliance (`mu1`, `mu2`, `kp`, `kd`),
  - joint damping/friction,
  - mass/inertia values.

### 3) Metrics and scoring logic
- `scripts/gait_analyser.py` constants and logic:
  - fall logic (`FALL_HEIGHT_FRACTION`),
  - step detection (`STEP_DEBOUNCE_S`, `STEP_MIN_AMPLITUDE`),
  - logging rate (`CSV_INTERVAL`).

### 4) Topic wiring (usually leave unchanged)
- `config/bridge.yaml` only if topic names/message types change.

## Data and Analysis Workflow

### Manual + recorded run

```bash
ros2 launch pady_robot analysis.launch.py kick_torque:=30 hip_push_torque:=5
```

Outputs:
- `~/ros2_ws/data/run_YYYYMMDD_HHMMSS.csv` from `gait_analyser`
- `~/ros2_ws/data/bag_YYYYMMDD_HHMMSS/` rosbag

### Parameter sweep

```bash
python3 ~/ros2_ws/src/pady_robot/scripts/param_sweep.py
```

Then plot:

```bash
python3 ~/ros2_ws/src/pady_robot/scripts/plot_results.py ~/ros2_ws/data/sweep_TIMESTAMP.csv
```

## Practical Troubleshooting

| Symptom | Typical adjustment | Location |
|---------|--------------------|----------|
| Falls backward quickly | Increase `kick_torque` slightly | Launch arg |
| Falls forward quickly | Decrease `kick_torque` | Launch arg |
| Yaw / lateral slip | Increase `mu1/mu2` or revisit foot cant | URDF contact/joint blocks |
| Not enough steps | Increase `hip_push_torque` or extend `hip_push_stop_time` | Launch args |
| Knee instability | Check knee damping/friction and limits | URDF knee joint dynamics |
