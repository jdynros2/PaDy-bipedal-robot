#!/usr/bin/env python3
"""Continuously publishes torque on hip joints during gait initiation,
with contralateral arm counterswing coupling.

Parameters
----------
torque : float
    Torque value (N·m) to apply to each hip.
arm_torque : float
    Torque value (N·m) to apply to arms for counterswing initiation.
    Applied contralaterally: right arm opposes left hip, left arm opposes right hip.
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
from sensor_msgs.msg import JointState


def ns(n):
    return n.lstrip("/")


class ContinuousHipPush(Node):
    def __init__(self):
        super().__init__('continuous_hip_push')

        self.declare_parameter('torque', 1.0)
        self.declare_parameter('arm_torque', 0.5)
        self.declare_parameter('body_force', 0.0)
        self.declare_parameter('start_time', 0.0)
        self.declare_parameter('stop_time', 0.0)
        self.declare_parameter('rate', 50.0)
        self.declare_parameter('use_sim_time', True)

        self.torque = self.get_parameter('torque').get_parameter_value().double_value
        self.arm_torque = self.get_parameter('arm_torque').get_parameter_value().double_value
        self.body_force = self.get_parameter('body_force').get_parameter_value().double_value
        self.start_time = self.get_parameter('start_time').get_parameter_value().double_value
        self.stop_time = self.get_parameter('stop_time').get_parameter_value().double_value
        rate = self.get_parameter('rate').get_parameter_value().double_value

        # Hip publishers
        self.left_pub = self.create_publisher(Float64, ns('/hip_kick_left'), 10)
        self.right_pub = self.create_publisher(Float64, ns('/hip_kick_right'), 10)
        self.body_pub = self.create_publisher(Float64, ns('/base_push'), 10)

        # Arm publishers (contralateral coupling)
        self.arm_right_pub = self.create_publisher(Float64, ns('/arm_kick_right'), 10)
        self.arm_left_pub = self.create_publisher(Float64, ns('/arm_kick_left'), 10)

        # Track hip positions for contralateral arm drive
        self.hip_right_pos = 0.0
        self.hip_left_pos = 0.0
        self.create_subscription(JointState, ns('/joint_states'), self.joint_cb, 10)

        self.timer = self.create_timer(1.0 / rate, self.publish_torque)
        self.get_logger().info(
            f"continuous_hip_push: hip={self.torque}Nm arm={self.arm_torque}Nm "
            f"from {self.start_time}s to {self.stop_time}s"
        )

    def joint_cb(self, msg):
        for i, name in enumerate(msg.name):
            if name == 'hip_joint_right' and i < len(msg.position):
                self.hip_right_pos = msg.position[i]
            elif name == 'hip_joint_left' and i < len(msg.position):
                self.hip_left_pos = msg.position[i]

    def publish_torque(self):
        now = self.get_clock().now().nanoseconds / 1e9
        msg = Float64()

        if self.start_time <= now < self.stop_time:
            # Hip kick
            msg.data = self.torque
            body_msg = Float64(data=self.body_force)

            # Contralateral arm torque: arm follows opposite hip velocity direction
            # Right arm opposes right hip (follows left hip)
            # Left arm opposes left hip (follows right hip)
            # Use hip position as proxy: positive hip angle → apply negative arm torque
            arm_right_msg = Float64(data=-self.arm_torque * self._sign(self.hip_left_pos))
            arm_left_msg = Float64(data=-self.arm_torque * self._sign(self.hip_right_pos))
        else:
            msg.data = 0.0
            body_msg = Float64(data=0.0)
            arm_right_msg = Float64(data=0.0)
            arm_left_msg = Float64(data=0.0)

        self.left_pub.publish(msg)
        self.right_pub.publish(msg)
        self.body_pub.publish(body_msg)
        self.arm_right_pub.publish(arm_right_msg)
        self.arm_left_pub.publish(arm_left_msg)

    @staticmethod
    def _sign(x):
        """Return -1, 0, or 1."""
        if x > 0.01:
            return 1.0
        elif x < -0.01:
            return -1.0
        return 0.0


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
