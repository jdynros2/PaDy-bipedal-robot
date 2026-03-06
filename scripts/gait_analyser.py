#!/usr/bin/env python3
"""Gait analysis node for the PaDy passive-dynamic walker.

Subscribes
----------
/joint_states       sensor_msgs/JointState   – hip + knee angles from Gazebo bridge
/pady/pose         geometry_msgs/Pose        – model world pose from PosePublisher

Publishes  (all Float64 unless noted, topic root = /gait/)
-----------------------------------------------------------
hip_right      – hip_joint_right position   (rad)
hip_left       – hip_joint_left  position   (rad)
knee_right     – knee_joint_right position  (rad)
knee_left      – knee_joint_left  position  (rad)
hip_symmetry   – hip_right + hip_left       (rad, ≈0 for symmetric gait)
base_height    – projected base height above slope plane (m)
hip_variance   – rolling population std of [hip_right, hip_left] over window
knee_variance  – rolling population std of [knee_right, knee_left] over window
gait_period    – time between detected steps (first non-zero value = time-to-first-step, then step-to-step period) (s)
step_count     – Int32, cumulative zero-crossing count
fall_detected  – Bool, True once projected height < (0.25 × initial projected height)
                 or base attitude exceeds roll/pitch limits

On shutdown (or first fall): flushes CSV to ~/ros2_ws/data/run_<timestamp>.csv

Usage
-----
ros2 run pady_robot gait_analyser.py
  or launched from launch/analysis.launch.py (preferred).

Parameters
----------
use_sim_time         bool   default True
variance_window      int    default 60  samples

Where to tune analysis behavior
-------------------------------
- Runtime: `variance_window` (launch parameter).
- Code-level thresholds: constants near top of this file
    (`FALL_HEIGHT_FRACTION`, `STEP_DEBOUNCE_S`, `STEP_MIN_AMPLITUDE`, etc.).
"""

import csv
import math
import os
import datetime
import statistics
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Int32

# ── quaternion → euler (ZYX intrinsic) ──────────────────────────────────────
def _quat_to_rpy(q):
    """Convert quaternion (x,y,z,w) to roll, pitch, yaw (rad)."""
    x, y, z, w = q
    # Roll (X)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    # Pitch (Y)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    # Yaw (Z)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


# ── tuneable analysis constants (edit here for global behavior changes) ─────
FALL_HEIGHT_FRACTION = 0.25     # fall if projected slope-height drops below this fraction of initial
FALL_ROLL_RAD = 1.20            # fall if |roll| exceeds this (rad)
FALL_PITCH_RAD = 1.30           # fall if |pitch| exceeds this (rad)
VARIANCE_WINDOW = 60            # default samples per joint for rolling std (can override by param)
STEP_DEBOUNCE_S = 0.25          # min time between counted zero-crossings
STEP_MIN_AMPLITUDE = 0.15       # min |hip| peak since last crossing to count a step
STEP_COUNT_MAX_ROLL_RAD = 0.70  # do not count steps when roll exceeds this (near-fall posture)
STEP_COUNT_MAX_PITCH_RAD = 1.00 # do not count steps when pitch exceeds this (near-fall posture)
KNEE_STEP_ARM_RAD = 0.35        # arm a knee-step once flexion exceeds this angle
KNEE_STEP_FIRE_RAD = 0.08       # count touchdown when armed knee extends below this angle
SLOPE_ANGLE = 0.0611            # world slope angle used for height projection
CSV_INTERVAL = 1.0 / 30.0       # CSV write rate (seconds)
DATA_DIR = os.path.expanduser('~/ros2_ws/data')


class GaitAnalyser(Node):
    def __init__(self):
        super().__init__('gait_analyser')

        self.declare_parameter('variance_window', VARIANCE_WINDOW)

        var_win = int(self.get_parameter('variance_window').value)

        # ── Pose state cache (updated by /pady/pose) ─────────────────────────
        self._base_pose_data = None  # (x, y, z, roll, pitch, yaw) or None

        # ── Publishers ───────────────────────────────────────────────────────
        def _fp(name):
            return self.create_publisher(Float64, f'/gait/{name}', 10)

        self._pub_hip_r    = _fp('hip_right')
        self._pub_hip_l    = _fp('hip_left')
        self._pub_knee_r   = _fp('knee_right')
        self._pub_knee_l   = _fp('knee_left')
        self._pub_sym      = _fp('hip_symmetry')
        self._pub_height   = _fp('base_height')
        self._pub_hip_var  = _fp('hip_variance')
        self._pub_knee_var = _fp('knee_variance')
        self._pub_period   = _fp('gait_period')
        self._pub_fall     = self.create_publisher(Bool,  '/gait/fall_detected', 10)
        self._pub_steps    = self.create_publisher(Int32, '/gait/step_count',    10)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)
        self.create_subscription(Pose, '/pady/pose', self._pose_cb, 10)

        # ── Runtime state ─────────────────────────────────────────────────────
        self._hip_win    = deque(maxlen=var_win)
        self._knee_win   = deque(maxlen=var_win)
        self._prev_sign_r = None         # previous sign of hip_right
        self._prev_sign_l = None         # previous sign of hip_left
        self._peak_r = 0.0               # peak |hip_r| since last right zero-crossing
        self._peak_l = 0.0               # peak |hip_l| since last left zero-crossing
        self._knee_r_armed = False       # true after right knee enters swing flexion zone
        self._knee_l_armed = False       # true after left knee enters swing flexion zone
        self._step_count = 0
        self._run_start_t = None         # first valid sim timestamp seen by analyser
        self._last_step_t = 0.0
        self._gait_period = 0.0
        self._fallen      = False
        # Fall detection threshold is computed from first valid pose sample.
        self._initial_slope_h = None  # captured from first valid pose
        self._fall_threshold = None   # FALL_HEIGHT_FRACTION × initial_slope_h
        self._last_csv_t = 0.0        # last sim_time a CSV row was written

        # ── CSV logging ──────────────────────────────────────────────────────
        os.makedirs(DATA_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._csv_path = os.path.join(DATA_DIR, f'run_{ts}.csv')
        self._csv_file = open(self._csv_path, 'w', newline='')
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow([
            'sim_time_s',
            'hip_right_rad', 'hip_left_rad',
            'knee_right_rad', 'knee_left_rad',
            'base_x_m', 'base_y_m',
            'base_roll_rad', 'base_pitch_rad', 'base_yaw_rad',
            'slope_height_m',
            'arm_right_rad', 'arm_left_rad',
            'hip_symmetry_rad',
            'hip_variance_rad', 'knee_variance_rad',
            'step_count', 'gait_period_s',
            'fall_detected',
        ])

        self.get_logger().info(
            f'gait_analyser started — logging to {self._csv_path}\n'
            f'  fall at {FALL_HEIGHT_FRACTION:.0%} of initial height, variance window = {var_win} samples'
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _pose_cb(self, msg: Pose):
        """Cache latest model world-pose from PosePublisher plugin."""
        p = msg.position
        q = msg.orientation
        roll, pitch, yaw = _quat_to_rpy((q.x, q.y, q.z, q.w))
        self._base_pose_data = (p.x, p.y, p.z, roll, pitch, yaw)

    @staticmethod
    def _slope_height(x, z):
        """Height of point (world x, world z) above the 3.5° slope plane."""
        # Slope descends in +X: plane normal = (-sin θ, 0, cos θ)
        # Height above plane = z·cos(θ) + x·sin(θ)  (offset absorbed into relative change)
        return z * math.cos(SLOPE_ANGLE) + x * math.sin(SLOPE_ANGLE)

    @staticmethod
    def _pstdev(window) -> float:
        return statistics.pstdev(window) if len(window) > 1 else 0.0

    # ── main callback ────────────────────────────────────────────────────────

    def _js_cb(self, msg: JointState):
        if not msg.name:
            return

        pos = dict(zip(msg.name, msg.position))
        hip_r  = pos.get('hip_joint_right',  float('nan'))
        hip_l  = pos.get('hip_joint_left',   float('nan'))
        knee_r = pos.get('knee_joint_right', float('nan'))
        knee_l = pos.get('knee_joint_left',  float('nan'))
        arm_r  = pos.get('arm_joint_right',  float('nan'))
        arm_l  = pos.get('arm_joint_left',   float('nan'))

        if any(math.isnan(v) for v in (hip_r, hip_l, knee_r, knee_l)):
            return

        sim_t = self.get_clock().now().nanoseconds / 1e9
        if self._run_start_t is None:
            self._run_start_t = sim_t

        # ── latest world pose (from /pady/pose subscriber) ───────────────────
        nan = float('nan')
        if self._base_pose_data is not None:
            bx, by, bz, b_roll, b_pitch, b_yaw = self._base_pose_data
            has_pose = True
            slope_h = self._slope_height(bx, bz)
        else:
            bx, by, bz, b_roll, b_pitch, b_yaw = nan, nan, nan, nan, nan, nan
            has_pose = False
            slope_h = nan

        # ── rolling variance ─────────────────────────────────────────────────
        self._hip_win.append(hip_r)
        self._hip_win.append(hip_l)
        self._knee_win.append(knee_r)
        self._knee_win.append(knee_l)
        hip_var  = self._pstdev(self._hip_win)
        knee_var = self._pstdev(self._knee_win)

        # ── fall detection: projected height + attitude guardrails ─
        if has_pose and not self._fallen:
            if self._initial_slope_h is None:
                self._initial_slope_h = slope_h
                self._fall_threshold = slope_h * FALL_HEIGHT_FRACTION
                self.get_logger().info(
                    f'Initial slope height = {slope_h:.3f}m, '
                    f'fall threshold = {self._fall_threshold:.3f}m, '
                    f'|roll|>{FALL_ROLL_RAD:.2f}rad or |pitch|>{FALL_PITCH_RAD:.2f}rad'
                )
            else:
                fell_by_height = slope_h < self._fall_threshold
                fell_by_attitude = (abs(b_roll) > FALL_ROLL_RAD) or (abs(b_pitch) > FALL_PITCH_RAD)
                if not (fell_by_height or fell_by_attitude):
                    pass
                else:
                    self._fallen = True
                    reason = 'height' if fell_by_height else 'attitude'
                    self.get_logger().warn(
                        f'FALL DETECTED ({reason})  t={sim_t:.2f}s  '
                        f'slope_h={slope_h:.3f}m  threshold={self._fall_threshold:.3f}m  '
                        f'roll={b_roll:.3f}rad  pitch={b_pitch:.3f}rad  steps={self._step_count}'
                    )
                    self._csv_file.flush()

        # ── step counting (frozen once fallen) ─────────────────────────────────
        # Primary detector: hip zero-crossings with amplitude and debounce.
        # Backup detector: knee touchdown (flexed knee extends back near 0 rad).
        sign_r = 1 if hip_r >= 0.0 else -1
        sign_l = 1 if hip_l >= 0.0 else -1
        self._peak_r = max(self._peak_r, abs(hip_r))
        self._peak_l = max(self._peak_l, abs(hip_l))
        stepped = False
        pose_ok_for_count = (not has_pose) or (
            abs(b_roll) <= STEP_COUNT_MAX_ROLL_RAD and
            abs(b_pitch) <= STEP_COUNT_MAX_PITCH_RAD
        )
        if not self._fallen:
            if self._prev_sign_r is not None and sign_r != self._prev_sign_r:
                dt = sim_t - self._last_step_t
                if dt >= STEP_DEBOUNCE_S and self._peak_r >= STEP_MIN_AMPLITUDE and pose_ok_for_count:
                    self._step_count += 1
                    stepped = True
                    if self._last_step_t > 0.0:
                        self._gait_period = dt
                    elif self._run_start_t is not None:
                        self._gait_period = max(0.0, sim_t - self._run_start_t)
                    self._last_step_t = sim_t
                self._peak_r = 0.0
            if self._prev_sign_l is not None and sign_l != self._prev_sign_l:
                dt = sim_t - self._last_step_t
                if dt >= STEP_DEBOUNCE_S and not stepped and self._peak_l >= STEP_MIN_AMPLITUDE and pose_ok_for_count:
                    self._step_count += 1
                    if self._last_step_t > 0.0:
                        self._gait_period = dt
                    elif self._run_start_t is not None:
                        self._gait_period = max(0.0, sim_t - self._run_start_t)
                    self._last_step_t = sim_t
                    stepped = True
                self._peak_l = 0.0

            # Knee touchdown step detector (helps count first physical step when
            # hip sign does not cross zero before fall).
            if knee_r >= KNEE_STEP_ARM_RAD:
                self._knee_r_armed = True
            if knee_l >= KNEE_STEP_ARM_RAD:
                self._knee_l_armed = True

            if self._knee_r_armed and knee_r <= KNEE_STEP_FIRE_RAD and not stepped:
                dt = sim_t - self._last_step_t
                if dt >= STEP_DEBOUNCE_S and pose_ok_for_count:
                    self._step_count += 1
                    if self._last_step_t > 0.0:
                        self._gait_period = dt
                    elif self._run_start_t is not None:
                        self._gait_period = max(0.0, sim_t - self._run_start_t)
                    self._last_step_t = sim_t
                    stepped = True
                self._knee_r_armed = False

            if self._knee_l_armed and knee_l <= KNEE_STEP_FIRE_RAD and not stepped:
                dt = sim_t - self._last_step_t
                if dt >= STEP_DEBOUNCE_S and pose_ok_for_count:
                    self._step_count += 1
                    if self._last_step_t > 0.0:
                        self._gait_period = dt
                    elif self._run_start_t is not None:
                        self._gait_period = max(0.0, sim_t - self._run_start_t)
                    self._last_step_t = sim_t
                    stepped = True
                self._knee_l_armed = False
        self._prev_sign_r = sign_r
        self._prev_sign_l = sign_l

        # ── publish ───────────────────────────────────────────────────────────
        def _pf(pub, v):
            pub.publish(Float64(data=float(v)))

        _pf(self._pub_hip_r,    hip_r)
        _pf(self._pub_hip_l,    hip_l)
        _pf(self._pub_knee_r,   knee_r)
        _pf(self._pub_knee_l,   knee_l)
        _pf(self._pub_sym,      hip_r + hip_l)
        _pf(self._pub_hip_var,  hip_var)
        _pf(self._pub_knee_var, knee_var)
        _pf(self._pub_period,   self._gait_period)
        if has_pose:
            _pf(self._pub_height, slope_h)

        self._pub_fall.publish(Bool(data=self._fallen))
        self._pub_steps.publish(Int32(data=self._step_count))

        # ── CSV row (throttled to ~30 Hz) ────────────────────────────────────
        if sim_t - self._last_csv_t >= CSV_INTERVAL:
            self._last_csv_t = sim_t

            def _fv(v, fmt='.4f'):
                return f'{v:{fmt}}' if not math.isnan(v) else ''

            def _fj(v):
                return f'{v:.6f}' if not math.isnan(v) else ''

            self._csv.writerow([
                f'{sim_t:.4f}',
                f'{hip_r:.6f}',  f'{hip_l:.6f}',
                f'{knee_r:.6f}', f'{knee_l:.6f}',
                _fv(bx), _fv(by),
                _fv(b_roll, '.6f'), _fv(b_pitch, '.6f'), _fv(b_yaw, '.6f'),
                _fv(slope_h),
                _fj(arm_r), _fj(arm_l),
                f'{hip_r + hip_l:.6f}',
                f'{hip_var:.6f}', f'{knee_var:.6f}',
                self._step_count,
                f'{self._gait_period:.4f}',
                int(self._fallen),
            ])

    # ── cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        self._csv_file.flush()
        self._csv_file.close()
        self.get_logger().info(f'Data saved → {self._csv_path}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GaitAnalyser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
