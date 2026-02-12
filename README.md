# PaDy - Passive Dynamic Bipedal Walker
"PaDy" is a passive dynamic slope walking robot, which i am creating for a university final year project. This repo will contain my workflow and updates to manage/track my progress.

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
world (fixed reference)
  └─ base_link [floating - 6DOF free movement]
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

# Launch simulation
ros2 launch pady_robot spawn_pady.launch.py
```

---


## Tuning for Walking

If robot falls immediately, adjust these values in `urdf/pady.urdf`:

| Parameter | Location in URDF | Effect |
|-----------|-----------------|--------|
| Hip COM Z | `base_link inertial origin z` | Raise/lower stability |
| Shin COM Z | `shin_* inertial origin z` | Change gait timing |
| Joint damping | `<dynamics damping="X"/>` | More=stiffer, Less=freer |
| Foot friction | `<mu1>X</mu1>` | More=less slip |
| Spawn height | launch file `-z` | Starting position |

---
