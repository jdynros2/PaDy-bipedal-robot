# PaDy — Passive Dynamic Bipedal Walker

3D Passive Dynamic Walker, simulated in ROS 2 Jazzy + Gazebo Harmonic (DART physics)/

Video medley
https://github.com/user-attachments/assets/74391a78-1923-4f4a-97c0-19f8769bd9da


## Prerequisites

- **Ubuntu 24.04** (or compatible Linux distribution)
- **ROS 2 Jazzy** — [installation guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
- **Gazebo Harmonic** — [installation guide](https://gazebosim.org/docs/harmonic/install_ubuntu)
- **ros_gz bridge** — `sudo apt install ros-jazzy-ros-gz`
- **colcon** — `sudo apt install python3-colcon-common-extensions`
- **rqt-plot** (for analysis launch) — `sudo apt install ros-jazzy-rqt-plot`

## Setup

```bash
# Clone the repository
cd ~/ros2_ws/src
git clone https://github.com/jdynros2/PaDy-bipedal-robot.git pady_robot

# Build
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select pady_robot
source install/setup.bash
```

## Running

```bash
# Standard simulation with built in arguments
ros2 launch pady_robot spawn_pady.launch.py

# Data collection (adds gait_analyser + rosbag recording)
ros2 launch pady_robot analysis.launch.py

# Best tuned passive run (CLI arguments for quick tweaking)
ros2 launch pady_robot analysis.launch.py \
  kick_torque:=0.2 kick_torque_right:=-0.2 \
  kick_follow_torque:=0 body_force:=0 hip_push_torque:=0 \
  hip_push_start_time:=0 hip_push_stop_time:=12.0 \
  spawn_x:=0.34 spawn_pitch:=0.42 spawn_roll:=-0.2385 spawn_yaw:=0.1

# Different slope (default: 3.50°, raanges from 2.85° -> 4.00°)
ros2 launch pady_robot analysis.launch.py world:=slope_3p00deg.sdf
```

## Robot Specifications

| Parameter | Value |
|-----------|-------|
| Total mass | 4.124 kg |
| Hip (base_link) mass | 1.600 kg |
| Leg mass (per side) | 0.962 kg |
| Arm mass (per side) | 0.300 kg |
| Thigh / shin length | 500 mm each |
| Hip width | 215.72 mm |
| Hip range | ±15° |
| Knee range | 0–65° |
| Optimal slope range | 2.95°–3.55° |

## Launch Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kick_torque` | 0.2 | Left hip kick (N·m) |
| `kick_torque_right` | -0.2 | Right hip kick (N·m) |
| `body_force` | 0 | Forward force on base link (N) |
| `hip_push_torque` | 0 | Hip bias torque during swing (N·m) |
| `spawn_x` | 0.34 | Initial x on slope (m) |
| `spawn_pitch` | 0.42 | Initial forward lean (rad) |
| `spawn_roll` | -0.2385 | Initial lateral roll (rad) |
| `spawn_yaw` | 0.1 | Initial yaw (rad) |
| `world` | slope_3p50deg.sdf | World file |


## Data Output

- CSV: `~/ros2_ws/data/run_YYYYMMDD_HHMMSS.csv`
- Rosbag: `~/ros2_ws/data/bag_YYYYMMDD_HHMMSS/`
- Slope test CSVs: `~/ros2_ws/data/slope_tests/`

## Key Files

| File | Purpose |
|------|---------|
| `urdf/pady.urdf` | Robot description (masses, joints, contacts) |
| `worlds/slope_*.sdf` | Slope world files |
| `config/bridge.yaml` | ROS↔Gazebo topic bridge |
| `scripts/gait_analyser.py` | Gait metrics + CSV logging |
| `scripts/continuous_hip_push.py` | Hip torque control |
| `scripts/knee_lock.py` | Knee locking controller |
| `scripts/yaw_corrector.py` | Yaw correction |
| `scripts/plot_*.py` | Analysis plotting scripts |

## Tuning Guide

| What to change | Where |
|----------------|-------|
| Gait initiation / forces | Launch arguments |
| Slope angle | `world:=slope_Xdeg.sdf` launch arg |
| Contact friction, joint limits, mass | `urdf/pady.urdf` |
| Fall detection, step counting | `scripts/gait_analyser.py` |
| Topic wiring | `config/bridge.yaml` |

## URDF Notes

- Hip joints have **built-in pitch offsets** (±0.28 rad) in the joint origin `rpy`. Raw joint angles from `/joint_states` must be corrected: `hip_left_actual = joint_angle - 0.28 rad`, `hip_right_actual = joint_angle + 0.28 rad`

## Passivity Audit

| Component | With passive CLI args | Truly passive? |
|-----------|----------------------|----------------|
| Hip push torque | 0 N·m | Yes (disabled) |
| Body forward force | 0 N | Yes (disabled) |
| Initial kick | 0.2 N·m for 0.2s | No — small but active |
| Knee lock/unlock | PD: 150 Kp, 8 swing assist | No — active controller |
| Yaw correction | PD: 15 Kp, up to 8 N·m | No — active controller |

**Both active controllers are necessary due to simulation limitations.**

- No rotational torque friction - robot turns due to yaw forces

- Knee lock would be suction cup in real life - simulation cant replicate this physics


