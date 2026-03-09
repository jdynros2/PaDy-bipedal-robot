#!/usr/bin/env python3
"""Passive knee-lock controller for PaDy walker.

Mimics a mechanical knee latch: locks the knee at full extension during
stance phase, releases it when the leg enters swing phase.

Detection logic
---------------
* **Lock** (stance):  knee angle < LOCK_THRESHOLD  AND  knee velocity is
  near-zero or closing (extending).  A stiff PD torque drives the knee to 0.
* **Unlock** (swing): hip angle indicates the leg is trailing (hip < 0 for
  right, hip > 0 for left — the leg is behind the body), so the knee
  should be free to flex for ground clearance.

The hip-angle sign convention after the URDF origin offset:
  * hip_joint_right has +0.28 offset → joint=0 means leg is already forward.
    Negative joint values = leg behind body = trailing = swing phase.
  * hip_joint_left  has -0.28 offset → joint=0 means leg is already back.
    Positive joint values = leg in front = swing phase completing.

Parameters (all tunable via launch)
----------
lock_gain     : PD proportional gain (N·m/rad), default 80
lock_damping  : PD derivative gain  (N·m·s/rad), default 5
lock_threshold: Knee angle (rad) below which lock engages, default 0.08
unlock_hip    : Hip angle magnitude past which knee unlocks, default -0.05
rate          : Control loop frequency (Hz), default 100
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

# Joint name → index mapping (filled on first message)
_IDX = {}


class KneeLock(Node):
    def __init__(self):
        super().__init__('knee_lock')

        # Tuneable parameters
        self.declare_parameter('lock_gain', 150.0)
        self.declare_parameter('lock_damping', 8.0)
        self.declare_parameter('lock_preload', -20.0)
        self.declare_parameter('lock_threshold', 0.08)
        self.declare_parameter('unlock_hip', -0.05)
        self.declare_parameter('rate', 200.0)
        self.declare_parameter('startup_hold', 2.0)
        self.declare_parameter('swing_assist', 8.0)       # N·m flexion kick on unlock
        self.declare_parameter('swing_assist_dur', 0.25)   # seconds of assist after unlock

        self.Kp = self.get_parameter('lock_gain').value
        self.Kd = self.get_parameter('lock_damping').value
        self.preload = self.get_parameter('lock_preload').value
        self.lock_thresh = self.get_parameter('lock_threshold').value
        self.unlock_hip = self.get_parameter('unlock_hip').value
        rate = self.get_parameter('rate').value
        self.startup_hold = self.get_parameter('startup_hold').value
        self.swing_assist = self.get_parameter('swing_assist').value
        self.swing_assist_dur = self.get_parameter('swing_assist_dur').value

        # Publishers — write to Gazebo knee force topics via bridge
        self.pub_r = self.create_publisher(Float64, '/knee_lock_right', 10)
        self.pub_l = self.create_publisher(Float64, '/knee_lock_left', 10)

        # State
        self.locked_r = False  # right leg starts as swing — unlocked for clearance
        self.locked_l = True   # left leg starts as stance — locked to hold weight
        self._unlock_time_r = None  # sim time when right knee last unlocked
        self._unlock_time_l = None  # sim time when left knee last unlocked
        self._last_js = None
        self._first_js_time = None  # sim time of first joint state

        # Subscriber
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)

        # Control loop timer
        self.create_timer(1.0 / rate, self._control)

        self.get_logger().info(
            f'knee_lock: Kp={self.Kp} Kd={self.Kd} '
            f'lock_thresh={self.lock_thresh} unlock_hip={self.unlock_hip}'
        )

    def _js_cb(self, msg: JointState):
        """Cache latest joint state."""
        global _IDX
        if not _IDX:
            _IDX = {n: i for i, n in enumerate(msg.name)}
        self._last_js = msg

    def _control(self):
        js = self._last_js
        if js is None or not _IDX:
            # Pre-publish: hold lock torques before any joint state arrives
            if self.locked_l:
                msg = Float64()
                msg.data = self.preload
                self.pub_l.publish(msg)
            if self.locked_r:
                msg = Float64()
                msg.data = self.preload
                self.pub_r.publish(msg)
            return

        # Track first joint state time for startup hold
        now = self.get_clock().now().nanoseconds / 1e9
        if self._first_js_time is None:
            self._first_js_time = now
        elapsed = now - self._first_js_time

        pos = js.position
        vel = js.velocity if js.velocity else [0.0] * len(pos)

        try:
            hip_r = pos[_IDX['hip_joint_right']]
            hip_l = pos[_IDX['hip_joint_left']]
            knee_r = pos[_IDX['knee_joint_right']]
            knee_l = pos[_IDX['knee_joint_left']]
            knee_vel_r = vel[_IDX['knee_joint_right']]
            knee_vel_l = vel[_IDX['knee_joint_left']]
        except (KeyError, IndexError):
            return

        # During startup hold, keep initial lock states — don't run hip detection
        if elapsed >= self.startup_hold:
            # ── Right knee ──────────────────────────────────────
            # Right hip offset +0.28: hip_r < unlock_hip means leg is trailing → swing
            if hip_r < self.unlock_hip:
                if self.locked_r:  # transition: locked → unlocked
                    self._unlock_time_r = now
                self.locked_r = False
            elif knee_r < self.lock_thresh:
                self.locked_r = True
                self._unlock_time_r = None

            # ── Left knee ───────────────────────────────────────
            # Left hip offset -0.28: hip_l > -unlock_hip means leg is trailing → swing
            if hip_l > -self.unlock_hip:
                if self.locked_l:  # transition: locked → unlocked
                    self._unlock_time_l = now
                self.locked_l = False
            elif knee_l < self.lock_thresh:
                self.locked_l = True
                self._unlock_time_l = None

        # ── Apply torques based on lock state ───────────────
        # preload provides constant extension force even at angle=0
        if self.locked_r:
            torque_r = self.preload - self.Kp * knee_r - self.Kd * knee_vel_r
        else:
            # Swing assist: brief flexion torque right after unlock to
            # kick-start knee flexion (compensates for thigh stalling at
            # the hip joint limit, which otherwise starves the shin of
            # the inertial impulse needed for ground clearance).
            if (self._unlock_time_r is not None and
                    (now - self._unlock_time_r) < self.swing_assist_dur):
                torque_r = self.swing_assist   # positive = flexion
            else:
                torque_r = 0.0

        if self.locked_l:
            torque_l = self.preload - self.Kp * knee_l - self.Kd * knee_vel_l
        else:
            if (self._unlock_time_l is not None and
                    (now - self._unlock_time_l) < self.swing_assist_dur):
                torque_l = self.swing_assist
            else:
                torque_l = 0.0

        # Publish
        msg_r = Float64()
        msg_r.data = torque_r
        self.pub_r.publish(msg_r)

        msg_l = Float64()
        msg_l.data = torque_l
        self.pub_l.publish(msg_l)


def main(args=None):
    rclpy.init(args=args)
    node = KneeLock()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
