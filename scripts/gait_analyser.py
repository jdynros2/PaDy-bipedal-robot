#!/usr/bin/env python3
"""Gait analysis node for the PaDy passive-dynamic walker.

Subscribes
----------
/joint_states       sensor_msgs/JointState   – hip + knee angles from Gazebo bridge
/tf                                          – world-frame transforms (via Pose_V bridge)

Publishes  (all Float64 unless noted, topic root = /gait/)
-----------------------------------------------------------
hip_right      – hip_joint_right position   (rad)
hip_left       – hip_joint_left  position   (rad)
knee_right     – knee_joint_right position  (rad)
knee_left      – knee_joint_left  position  (rad)
hip_symmetry   – hip_right + hip_left       (rad, ≈0 for symmetric gait)
base_height    – world-z of base_link       (m)
hip_variance   – rolling population std of [hip_right, hip_left] over window
knee_variance  – rolling population std of [knee_right, knee_left] over window
gait_period    – time between consecutive hip_right zero-crossings (s)
step_count     – Int32, cumulative zero-crossing count
fall_detected  – Bool, True once base_height < threshold

On shutdown (or first fall): flushes CSV to ~/ros2_ws/data/run_<timestamp>.csv

Usage
-----
ros2 run pady_robot gait_analyser.py
  or launched from launch/analysis.launch.py (preferred).

Parameters
----------
use_sim_time         bool   default True
fall_height_threshold float  default 0.35 m
variance_window      int    default 60  samples
"""

import csv
import math
import os
import datetime
import statistics
from collections import deque

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Int32
import tf2_ros


# ── tuneable defaults ────────────────────────────────────────────────────────
FALL_HEIGHT_THRESHOLD = 0.35   # m — base_link below this ⇒ fallen
VARIANCE_WINDOW = 60            # samples per joint for rolling std
STEP_DEBOUNCE_S = 0.25          # min seconds between counted zero-crossings
DATA_DIR = os.path.expanduser('~/ros2_ws/data')


class GaitAnalyser(Node):
    def __init__(self):
        super().__init__('gait_analyser')

        self.declare_parameter('fall_height_threshold', FALL_HEIGHT_THRESHOLD)
        self.declare_parameter('variance_window', VARIANCE_WINDOW)

        fall_thresh = self.get_parameter('fall_height_threshold').value
        var_win = int(self.get_parameter('variance_window').value)

        # ── TF listener ──────────────────────────────────────────────────────
        self._tf_buf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

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

        # ── Subscriber ───────────────────────────────────────────────────────
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)

        # ── State ────────────────────────────────────────────────────────────
        self._hip_win    = deque(maxlen=var_win)
        self._knee_win   = deque(maxlen=var_win)
        self._prev_sign  = None          # previous sign of hip_right (step counter)
        self._step_count = 0
        self._last_step_t = 0.0
        self._gait_period = 0.0
        self._fall_thresh = fall_thresh
        self._fallen      = False

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
            'base_height_m',
            'hip_symmetry_rad',
            'hip_variance_rad', 'knee_variance_rad',
            'step_count', 'gait_period_s',
            'fall_detected',
        ])

        self.get_logger().info(
            f'gait_analyser started — logging to {self._csv_path}\n'
            f'  fall threshold = {fall_thresh:.2f} m, variance window = {var_win} samples'
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _base_height(self) -> float:
        """Look up base_link world-z from TF (GZ bridge: slope_world → pady/base_link)."""
        try:
            tf = self._tf_buf.lookup_transform(
                'slope_world', 'pady/base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            return tf.transform.translation.z
        except Exception:
            return float('nan')

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

        if any(math.isnan(v) for v in (hip_r, hip_l, knee_r, knee_l)):
            return

        sim_t = self.get_clock().now().nanoseconds / 1e9

        # ── world height ─────────────────────────────────────────────────────
        height = self._base_height()

        # ── rolling variance ─────────────────────────────────────────────────
        self._hip_win.append(hip_r)
        self._hip_win.append(hip_l)
        self._knee_win.append(knee_r)
        self._knee_win.append(knee_l)
        hip_var  = self._pstdev(self._hip_win)
        knee_var = self._pstdev(self._knee_win)

        # ── step counting (hip_right zero-crossings) ──────────────────────────
        sign = 1 if hip_r >= 0.0 else -1
        if self._prev_sign is not None and sign != self._prev_sign:
            dt = sim_t - self._last_step_t
            if dt >= STEP_DEBOUNCE_S:
                self._step_count += 1
                if self._last_step_t > 0.0:
                    self._gait_period = dt
                self._last_step_t = sim_t
        self._prev_sign = sign

        # ── fall detection ────────────────────────────────────────────────────
        if not math.isnan(height) and height < self._fall_thresh and not self._fallen:
            self._fallen = True
            self.get_logger().warn(
                f'FALL DETECTED  t={sim_t:.2f}s  height={height:.3f}m  steps={self._step_count}'
            )
            self._csv_file.flush()   # ensure data is on disk

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
        if not math.isnan(height):
            _pf(self._pub_height, height)

        self._pub_fall.publish(Bool(data=self._fallen))
        self._pub_steps.publish(Int32(data=self._step_count))

        # ── CSV row ───────────────────────────────────────────────────────────
        self._csv.writerow([
            f'{sim_t:.4f}',
            f'{hip_r:.6f}',  f'{hip_l:.6f}',
            f'{knee_r:.6f}', f'{knee_l:.6f}',
            f'{height:.4f}' if not math.isnan(height) else '',
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
