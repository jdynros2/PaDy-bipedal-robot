"""analysis.launch.py — Data-collection launch for repeatable experiments.

Wraps spawn_pady.launch.py and adds:
  * gait_analyser node  — publishes /gait/* metrics, auto-saves CSV
    * ros2 bag record     — records core run topics for post-analysis

Where to adjust parameters
--------------------------
- Gait/force/start-pose tuning: launch args in this file (passed to spawn).
- Bag content: `bag_record` topic list in this file.
- Metric definitions and CSV columns: `scripts/gait_analyser.py`.

Usage
-----
ros2 launch pady_robot analysis.launch.py
ros2 launch pady_robot analysis.launch.py kick_torque:=40.0 hip_push_torque:=8.0
"""

import datetime
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('pady_robot')

    # ── pass-through tuning args (edit defaults here or override on CLI) ────
    kick_torque_arg = DeclareLaunchArgument(
        'kick_torque', default_value='30.0',
        description='Hip kick magnitude (N·m)')
    kick_torque_right_arg = DeclareLaunchArgument(
        'kick_torque_right', default_value='-30.0',
        description='Right-hip initial kick torque (N·m), usually opposite sign to left')
    kick_follow_torque_arg = DeclareLaunchArgument(
        'kick_follow_torque', default_value='-18.0',
        description='Right-hip follow-through torque after initial kick (N·m)')
    hip_push_torque_arg = DeclareLaunchArgument(
        'hip_push_torque', default_value='5.0',
        description='Hip bias torque (N·m)')
    hip_push_start_arg = DeclareLaunchArgument(
        'hip_push_start_time', default_value='0.8',
        description='Bias start time (s)')
    hip_push_stop_arg = DeclareLaunchArgument(
        'hip_push_stop_time', default_value='12.0',
        description='Bias end time (s)')
    body_force_arg = DeclareLaunchArgument(
        'body_force', default_value='20.0',
        description='Forward force on base link (N)')
    spawn_x_arg = DeclareLaunchArgument(
        'spawn_x', default_value='0.45',
        description='Initial x-coordinate')
    spawn_pitch_arg = DeclareLaunchArgument(
        'spawn_pitch', default_value='0.275',
        description='Initial forward pitch (rad)')
    spawn_roll_arg = DeclareLaunchArgument(
        'spawn_roll', default_value='-0.08',
        description='Initial lateral roll (rad)')
    world_arg = DeclareLaunchArgument(
        'world', default_value='slope_3deg.sdf',
        description='World SDF filename inside pady_robot/worlds')
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation time')

    # ── Use analysis RViz layout (world-frame metrics view) ──────────────────
    analysis_rviz = os.path.join(pkg_dir, 'rviz', 'pady_analysis.rviz')

    # ── Include main simulation launch (single source of gait timing) ───────
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'spawn_pady.launch.py')
        ),
        launch_arguments={
            'rvizconfig':           analysis_rviz,
            'kick_torque':          LaunchConfiguration('kick_torque'),
            'kick_torque_right':    LaunchConfiguration('kick_torque_right'),
            'kick_follow_torque':   LaunchConfiguration('kick_follow_torque'),
            'hip_push_torque':      LaunchConfiguration('hip_push_torque'),
            'hip_push_start_time':  LaunchConfiguration('hip_push_start_time'),
            'hip_push_stop_time':   LaunchConfiguration('hip_push_stop_time'),
            'body_force':           LaunchConfiguration('body_force'),
            'spawn_x':              LaunchConfiguration('spawn_x'),
            'spawn_pitch':          LaunchConfiguration('spawn_pitch'),
            'spawn_roll':           LaunchConfiguration('spawn_roll'),
            'world':                LaunchConfiguration('world'),
            'use_sim_time':         LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    # ── Start analyser before unpause so step-1 is captured in CSV/bag ───────
    analyser_node = TimerAction(
        period=7.0,
        actions=[Node(
            package='pady_robot',
            executable='gait_analyser.py',
            name='gait_analyser',
            output='screen',
            parameters=[{
                'use_sim_time':           LaunchConfiguration('use_sim_time'),
            }],
        )],
    )

    # ── Rosbag capture set (edit this list when adding/removing metrics) ─────
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_dir = os.path.expanduser(f'~/ros2_ws/data/bag_{ts}')

    bag_record = TimerAction(
        period=7.0,
        actions=[ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '-o', bag_dir,
                '--topics',
                '/joint_states',
                '/pady/pose',
                '/gait/hip_right',
                '/gait/hip_left',
                '/gait/knee_right',
                '/gait/knee_left',
                '/gait/base_height',
                '/gait/hip_symmetry',
                '/gait/hip_variance',
                '/gait/knee_variance',
                '/gait/gait_period',
                '/gait/step_count',
                '/gait/fall_detected',
                '/tf',
                '/tf_static',
            ],
            output='screen',
        )],
    )

    return LaunchDescription([
        use_sim_time_arg,
        kick_torque_arg,
        kick_torque_right_arg,
        kick_follow_torque_arg,
        hip_push_torque_arg,
        hip_push_start_arg,
        hip_push_stop_arg,
        body_force_arg,
        spawn_x_arg,
        spawn_pitch_arg,
        spawn_roll_arg,
        world_arg,
        sim_launch,
        analyser_node,
        bag_record,
    ])
