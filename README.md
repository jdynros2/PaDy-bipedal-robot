# PaDy - Passive Dynamic Bipedal Walker
"PaDy" is a passive dynamic slope walking robot, which I am creating for a university final year project. This repo will contain my workflow and updates to manage/track my progress.

---

## Design Parameters

| Parameter | Value |
|-----------|-------|
| Total height | ~1.017m |
| Thigh length | 500mm |
| Shin length | 500mm |
| Hip width (bearing to bearing) | 215.72mm |
| Total mass | 3.524 kg |
| Target slope | 3 degrees |

---

## Mass Distribution 

Based on **Collins et al. 2001** and **Morales-Cruz 2014**:

| Link | Mass | COM Location | Why |
|------|------|-------------|-----|
| base_link (hip) | 1.600 kg | 15mm below hip joint | Stability, high centre |
| thigh_right/left | 0.437 kg each | 47.5mm from hip (9.5%) | Near-hip concentration |
| shin_right/left | 0.350 kg each | 32mm from knee (93.6% from ankle) | Near-knee concentration |
| foot_right/left | 0.175 kg each | 40mm forward | Ground contact mass |

**Key insight:** Mass concentrated near the KNEE (bottom of thigh + top of shin)
creates the limit cycle needed for stable passive gait.

---

## Joint Structure

```
base_link [free-floating root - Gazebo treats URDF root as 6 DOF]
    ├─ thigh_right [hip_joint_right - revolute ±30deg]
    │   └─ shin_right [knee_joint_right - revolute 0-65deg]
    │       └─ foot_right [foot_joint_right - FIXED, no ankle]
    └─ thigh_left [hip_joint_left - revolute ±30deg]
        └─ shin_left [knee_joint_left - revolute 0-65deg]
            └─ foot_left [foot_joint_left - FIXED, no ankle]
```

---

## Quick Start (University PC)

```bash
# Clone repository
cd ~/ros2_ws/src
git clone https://github.com/jdynros2/PaDy-bipedal-walker.git pady_robot

# Build
cd ~/ros2_ws
colcon build --packages-select pady_robot
source install/setup.bash

# Validate URDF
check_urdf src/pady_robot/urdf/pady.urdf

# Launch simulation (default parameters)
ros2 launch pady_robot spawn_pady.launch.py

# Launch with custom gait parameters
ros2 launch pady_robot spawn_pady.launch.py \
  kick_torque:=30.0 \
  hip_push_torque:=5.0 \
  body_force:=20.0 \
  hip_push_start_time:=7.0 \
  hip_push_stop_time:=12.0
```

---

## Gait Initiation Parameters

The robot uses Collins-style two-step gait initiation with continuous hip bias. All parameters are command-line tunable:

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `kick_torque` | 30.0 | N·m | Hip joint torque at push-off (left @ T=7.5s, right @ T=9.0s) |
| `hip_push_torque` | 5.0 | N·m | Continuous hip bias during swing phase |
| `body_force` | 20.0 | N | Forward force on base_link (COM momentum) |
| `hip_push_start_time` | 7.0 | s | Simulation time to start hip bias |
| `hip_push_stop_time` | 12.0 | s | Simulation time to stop hip bias |
| `spawn_pitch` | 0.28 | rad | Initial forward lean (~16°) to initiate step-1 |
| `spawn_x` | 0.45 | m | Initial x-position on slope |

**Physics Timeline:**
- T=5.0s: Robot spawned (paused)
- T=7.5s: Left hip kick (150 ms burst → 30 Nm)
- T=8.0s: Physics unpaused
- T=9.0s: Right hip kick (150 ms burst → 30 Nm)
- T=7.0–12.0s: Continuous hip bias + body force
- **Result:** 4-step walking (first 3 controlled, 4th passive)

---

## Tuning for Walking

### Current Working Configuration (4-step walking)

Foot contact parameters in `urdf/pady.urdf` (verified working):

| Parameter | Value | Effect |
|-----------|-------|--------|
| Foot friction (μ₁, μ₂) | 4.0 | Prevents yaw-inducing slip |
| Foot stiffness (kp) | 10e6 Pa | High contact stiffness for ground lock |
| Foot damping (kd) | 1000 | Contact damping |
| Knee joint damping | 0.0 | Zero damping for true passive swing |

### Adjustment Checklist

If robot falls or doesn't walk:

| Symptom | Adjustment | Location |
|---------|-----------|----------|
| Falls backward | Increase `kick_torque` | Launch parameter |
| Falls forward | Decrease `kick_torque` | Launch parameter |
| Sideways slip/yaw | Increase `body_force` or foot μ | Launch param or URDF |
| Only 2 steps | Extend `hip_push_stop_time` to 12.0+ | Launch parameter |
| Knee buckles | Ensure `knee_joint_left/right` damping = 0.0 | URDF `<dynamics>` |

---
