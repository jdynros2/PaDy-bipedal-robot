"""analysis.launch.py — PaDy full experimental data-collection launch.

Wraps spawn_pady.launch.py and adds:
  • gait_analyser node  — publishes /gait/* metrics, auto-saves CSV
  • rqt_plot (joint angles)        — live plot: hip + knee angles
  • rqt_plot (performance metrics) — live plot: base height + symmetry + variance
  • ros2 bag record                — records all /gait/* + /joint_states + /tf

Override the RViz config to use pady_analysis.rviz (world-frame view with TF display).

Usage
-----
ros2 launch pady_robot analysis.launch.py
ros2 launch pady_robot analysis.launch.py kick_torque:=40.0 hip_push_torque:=8.0
ros2 launch pady_robot analysis.launch.py record_bag:=false
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
        'spawn_pitch', default_value='0.28',
        description='Initial forward pitch (rad)')
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation time')

    # ── analysis-specific parameters ─────────────────────────────────────────
    record_bag_arg = DeclareLaunchArgument(
        'record_bag', default_value='true',
        description='Record a rosbag of all /gait/* and /joint_states topics')
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

    # ── gait analyser node ────────────────────────────────────────────────────
    analyser_node = Node(
        package='pady_robot',
        executable='gait_analyser.py',
        name='gait_analyser',
        output='screen',
        parameters=[{
            'use_sim_time':           LaunchConfiguration('use_sim_time'),
            'fall_height_threshold':  LaunchConfiguration('fall_height_threshold'),
        }],
    )

    # ── rqt_plot — joint angles ───────────────────────────────────────────────
    # Delay until analyser is publishing (topics appear ~8s in after physics starts)
    rqt_joints = TimerAction(
        period=12.0,
        actions=[ExecuteProcess(
            cmd=[
                'rqt_plot',
                '/gait/hip_right/data',
                '/gait/hip_left/data',
                '/gait/knee_right/data',
                '/gait/knee_left/data',
            ],
            output='screen',
        )],
    )

    # ── rqt_plot — performance metrics ───────────────────────────────────────
    rqt_perf = TimerAction(
        period=12.0,
        actions=[ExecuteProcess(
            cmd=[
                'rqt_plot',
                '/gait/base_height/data',
                '/gait/hip_symmetry/data',
                '/gait/hip_variance/data',
                '/gait/knee_variance/data',
            ],
            output='screen',
        )],
    )

    # ── ros2 bag record ───────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_dir = os.path.expanduser(f'~/ros2_ws/data/bag_{ts}')

    bag_record = TimerAction(
        period=7.0,   # start recording just before physics unpauses
        actions=[ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '-o', bag_dir,
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
        # ── declare args ──────────────────────────────────────────────────────
        use_sim_time_arg,
        kick_torque_arg,
        hip_push_torque_arg,
        hip_push_start_arg,
        hip_push_stop_arg,
        body_force_arg,
        spawn_x_arg,
        spawn_pitch_arg,
        record_bag_arg,
        fall_threshold_arg,
        # ── simulation ────────────────────────────────────────────────────────
        sim_launch,
        # ── analysis tools ───────────────────────────────────────────────────
        analyser_node,
        bag_record,
        rqt_joints,
        rqt_perf,
    ])
