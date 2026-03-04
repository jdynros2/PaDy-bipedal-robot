"""analysis.launch.py — PaDy full experimental data-collection launch.

Wraps spawn_pady.launch.py and adds:
  * gait_analyser node  — publishes /gait/* metrics, auto-saves CSV
  * ros2 bag record     — records all /gait/* + /joint_states + /tf

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

    # ── pass-through gait parameters ─────────────────────────────────────────
    kick_torque_arg = DeclareLaunchArgument(
        'kick_torque', default_value='30.0',
        description='Hip kick magnitude (N·m)')
    hip_push_torque_arg = DeclareLaunchArgument(
        'hip_push_torque', default_value='5.0',
        description='Hip bias torque (N·m)')
    hip_push_start_arg = DeclareLaunchArgument(
        'hip_push_start_time', default_value='7.0',
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
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation time')

    # ── analysis-specific parameters ─────────────────────────────────────────
    fall_threshold_arg = DeclareLaunchArgument(
        'fall_height_threshold', default_value='0.35',
        description='World-z below which fall is declared (m)')

    # ── RViz: override to world-frame analysis config ─────────────────────────
    analysis_rviz = os.path.join(pkg_dir, 'rviz', 'pady_analysis.rviz')

    # ── include the main simulation launch ───────────────────────────────────
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'spawn_pady.launch.py')
        ),
        launch_arguments={
            'rvizconfig':           analysis_rviz,
            'kick_torque':          LaunchConfiguration('kick_torque'),
            'hip_push_torque':      LaunchConfiguration('hip_push_torque'),
            'hip_push_start_time':  LaunchConfiguration('hip_push_start_time'),
            'hip_push_stop_time':   LaunchConfiguration('hip_push_stop_time'),
            'body_force':           LaunchConfiguration('body_force'),
            'spawn_x':              LaunchConfiguration('spawn_x'),
            'spawn_pitch':          LaunchConfiguration('spawn_pitch'),
            'use_sim_time':         LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    # ── gait analyser node (delayed until after unpause at T=8s) ──────────────
    analyser_node = TimerAction(
        period=9.0,
        actions=[Node(
            package='pady_robot',
            executable='gait_analyser.py',
            name='gait_analyser',
            output='screen',
            parameters=[{
                'use_sim_time':           LaunchConfiguration('use_sim_time'),
                'fall_height_threshold':  LaunchConfiguration('fall_height_threshold'),
            }],
        )],
    )

    # ── ros2 bag record ───────────────────────────────────────────────────────
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
        hip_push_torque_arg,
        hip_push_start_arg,
        hip_push_stop_arg,
        body_force_arg,
        spawn_x_arg,
        spawn_pitch_arg,
        fall_threshold_arg,
        sim_launch,
        analyser_node,
        bag_record,
    ])
