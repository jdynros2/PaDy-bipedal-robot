#!/usr/bin/env python3
"""Yaw corrector — applies a restoring torque to the yaw_joint via the
ros_gz_bridge (same latched mechanism as hip torques and knee locks).

The torque is applied every physics step (deterministic, no subprocess jitter).
Subscribes to /pady/pose, computes PD correction, publishes Float64 to
/yaw_correction which bridges to the ApplyJointForce plugin on yaw_joint.

Parameters
----------
kp : float   Proportional gain (N·m / rad).  Default 10.0.
kd : float   Derivative gain (N·m·s / rad).  Default 3.0.
max_torque : float   Clamp magnitude (N·m).  Default 8.0.
rate : float   Control loop rate (Hz).  Default 50.0.
boost_multiplier : float   Gain multiplier during first step.  Default 3.0.
boost_duration : float   Boost window (s).  Default 3.0.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64


class YawCorrector(Node):
    def __init__(self):
        super().__init__('yaw_corrector')

        self.declare_parameter('kp', 15.0)
        self.declare_parameter('kd', 4.0)
        self.declare_parameter('max_torque', 8.0)
        self.declare_parameter('rate', 500.0)

        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value
        self.max_torque = self.get_parameter('max_torque').value
        rate = self.get_parameter('rate').value

        self._pub = self.create_publisher(Float64, '/yaw_correction', 10)

        self._last_yaw = 0.0
        self._last_time = 0.0
        self._first_time = 0.0
        self._min_dt = 1.0 / rate

        self.create_subscription(Pose, '/pady/pose', self._pose_cb, 10)
        self.get_logger().info(
            f'yaw_corrector (bridge): Kp={self.kp}, Kd={self.kd}, '
            f'max={self.max_torque} N·m, rate={rate} Hz')

    @staticmethod
    def _quat_to_yaw(q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def _pose_cb(self, msg: Pose):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_time < self._min_dt:
            return

        yaw = self._quat_to_yaw(msg.orientation)
        dt = now - self._last_time if self._last_time > 0 else self._min_dt
        yaw_rate = (yaw - self._last_yaw) / dt if dt > 0.001 else 0.0

        torque = -self.kp * yaw - self.kd * yaw_rate
        torque = max(-self.max_torque, min(self.max_torque, torque))

        self._last_yaw = yaw
        self._last_time = now

        self._pub.publish(Float64(data=torque))


def main(args=None):
    rclpy.init(args=args)
    node = YawCorrector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Zero out torque on shutdown
        node._pub.publish(Float64(data=0.0))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
