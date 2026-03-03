#!/usr/bin/env python3
"""Continuously publishes torque on hip joints during gait initiation.

Parameters
----------
torque : float
    Torque value (N·m) to apply to each hip.
body_force : float
    Forward force on base link (N).
start_time : float
    Simulation time (seconds) to start publishing.
stop_time : float
    Simulation time (seconds) to stop publishing.
rate : float
    Publish frequency (Hz).
use_sim_time : bool
    Use simulation time if true.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


def ns(n):
    return n.lstrip("/")


class ContinuousHipPush(Node):
    def __init__(self):
        super().__init__('continuous_hip_push')

        self.declare_parameter('torque', 1.0)
        self.declare_parameter('body_force', 0.0)
        self.declare_parameter('start_time', 0.0)
        self.declare_parameter('stop_time', 0.0)
        self.declare_parameter('rate', 50.0)
        self.declare_parameter('use_sim_time', True)

        self.torque = self.get_parameter('torque').get_parameter_value().double_value
        self.body_force = self.get_parameter('body_force').get_parameter_value().double_value
        self.start_time = self.get_parameter('start_time').get_parameter_value().double_value
        self.stop_time = self.get_parameter('stop_time').get_parameter_value().double_value
        rate = self.get_parameter('rate').get_parameter_value().double_value

        self.left_pub = self.create_publisher(Float64, ns('/hip_kick_left'), 10)
        self.right_pub = self.create_publisher(Float64, ns('/hip_kick_right'), 10)
        self.body_pub = self.create_publisher(Float64, ns('/base_push'), 10)

        self.timer = self.create_timer(1.0 / rate, self.publish_torque)
        self.get_logger().info(f"continuous_hip_push: apply {self.torque} Nm from {self.start_time}s to {self.stop_time}s")

    def publish_torque(self):
        now = self.get_clock().now().nanoseconds / 1e9
        msg = Float64()
        if self.start_time <= now < self.stop_time:
            msg.data = self.torque
            body_msg = Float64(data=self.body_force)
        else:
            msg.data = 0.0
            body_msg = Float64(data=0.0)

        self.left_pub.publish(msg)
        self.right_pub.publish(msg)
        self.body_pub.publish(body_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ContinuousHipPush()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
