"""headless.launch.py — PaDy simulation with no GUI, for automated parameter sweeps.

Identical timing to spawn_pady.launch.py but:
  - gz sim runs server-only (-s flag, no rendering window)
  - No RViz, no rqt_plot, no rosbag
  - gait_analyser included (saves CSV automatically)

Usage
-----
# Directly
ros2 launch pady_robot headless.launch.py kick_torque:=35.0 hip_push_torque:=7.0

# Via param_sweep.py (preferred)
python3 scripts/param_sweep.py
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    with open(os.path.join(get_package_share_directory('pady_robot'), 'urdf', 'pady.urdf'), 'r') as f:
        robot_desc = f.read()

    pkg_dir = get_package_share_directory('pady_robot')
    world_path = os.path.join(pkg_dir, 'worlds', 'slope_3deg.sdf')

    robot_desc_gz = robot_desc.replace(
        'package://pady_robot/',
        f'file://{pkg_dir}/'
    )

    # ── parameters ─────────────────────────────────────────────────────────────
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true')
    use_sim_time = LaunchConfiguration('use_sim_time')

    kick_torque_arg = DeclareLaunchArgument('kick_torque', default_value='30.0',
                                            description='Hip kick magnitude (N·m)')
    kick_torque = LaunchConfiguration('kick_torque')

    hip_push_torque_arg = DeclareLaunchArgument('hip_push_torque', default_value='5.0',
                                                description='Hip bias torque (N·m)')
    hip_push_torque = LaunchConfiguration('hip_push_torque')

    hip_push_start_arg = DeclareLaunchArgument('hip_push_start_time', default_value='7.0')
    hip_push_start_time = LaunchConfiguration('hip_push_start_time')

    hip_push_stop_arg = DeclareLaunchArgument('hip_push_stop_time', default_value='12.0')
    hip_push_stop_time = LaunchConfiguration('hip_push_stop_time')

    body_force_arg = DeclareLaunchArgument('body_force', default_value='20.0')
    body_force = LaunchConfiguration('body_force')

    spawn_x_arg = DeclareLaunchArgument('spawn_x', default_value='0.45')
    spawn_x = LaunchConfiguration('spawn_x')

    spawn_pitch_arg = DeclareLaunchArgument('spawn_pitch', default_value='0.28',
                                            description='Initial forward pitch (rad)')
    spawn_pitch = LaunchConfiguration('spawn_pitch')

    fall_threshold_arg = DeclareLaunchArgument('fall_height_threshold', default_value='0.35')

    # ── Gazebo server-only (no rendering window) ──────────────────────────────
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-s', world_path],
        output='screen'
    )

    spawn_robot = TimerAction(
        period=5.0,
        actions=[Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-string', robot_desc_gz,
                '-name',   'pady',
                '-x',      spawn_x,
                '-y',      '0',
                '-z',      '1.42',
                '-R',      '-0.2',
                '-P',      spawn_pitch,
                '-Y',      '0',
                '-J', 'hip_joint_right',  '0.55',
                '-J', 'hip_joint_left',   '-0.50',
                '-J', 'knee_joint_right',  '0.02',
                '-J', 'knee_joint_left',   '1.05',
            ],
            output='screen'
        )]
    )

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc, 'use_sim_time': use_sim_time}],
        output='screen'
    )

    joint_state_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[
            {'config_file': os.path.join(pkg_dir, 'config', 'bridge.yaml')},
            {'use_sim_time': use_sim_time},
        ],
        output='screen'
    )

    unpause = TimerAction(
        period=8.0,
        actions=[ExecuteProcess(
            cmd=[
                'gz', 'service',
                '-s', '/world/slope_world/control',
                '--reqtype', 'gz.msgs.WorldControl',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '2000',
                '--req', 'pause: false'
            ],
            output='screen'
        )]
    )

    kick_data = [TextSubstitution(text='data: '), kick_torque]

    kick_left = TimerAction(period=7.5, actions=[ExecuteProcess(
        cmd=['gz', 'topic', '-t', '/model/pady/joint/hip_joint_left/0/cmd_force',
             '-m', 'gz.msgs.Double', '-p', kick_data], output='screen')])

    kick_left_stop = TimerAction(period=8.15, actions=[ExecuteProcess(
        cmd=['gz', 'topic', '-t', '/model/pady/joint/hip_joint_left/0/cmd_force',
             '-m', 'gz.msgs.Double', '-p', 'data: 0.0'], output='screen')])

    kick_right = TimerAction(period=9.0, actions=[ExecuteProcess(
        cmd=['gz', 'topic', '-t', '/model/pady/joint/hip_joint_right/0/cmd_force',
             '-m', 'gz.msgs.Double', '-p', kick_data], output='screen')])

    kick_right_stop = TimerAction(period=9.15, actions=[ExecuteProcess(
        cmd=['gz', 'topic', '-t', '/model/pady/joint/hip_joint_right/0/cmd_force',
             '-m', 'gz.msgs.Double', '-p', 'data: 0.0'], output='screen')])

    release = TimerAction(period=10.5, actions=[
        ExecuteProcess(cmd=['gz', 'topic', '-t', '/model/pady/joint/hip_joint_right/0/cmd_force',
                            '-m', 'gz.msgs.Double', '-p', 'data: 0.0'], output='screen'),
        ExecuteProcess(cmd=['gz', 'topic', '-t', '/model/pady/joint/hip_joint_left/0/cmd_force',
                            '-m', 'gz.msgs.Double', '-p', 'data: 0.0'], output='screen'),
    ])

    hip_bias_node = Node(
        package='pady_robot',
        executable='continuous_hip_push.py',
        output='screen',
        parameters=[{
            'torque':      hip_push_torque,
            'body_force':  body_force,
            'start_time':  hip_push_start_time,
            'stop_time':   hip_push_stop_time,
            'rate':        50.0,
            'use_sim_time': use_sim_time,
        }]
    )

    analyser_node = Node(
        package='pady_robot',
        executable='gait_analyser.py',
        name='gait_analyser',
        output='screen',
        parameters=[{
            'use_sim_time':          use_sim_time,
            'fall_height_threshold': LaunchConfiguration('fall_height_threshold'),
        }]
    )

    return LaunchDescription([
        use_sim_time_arg,
        kick_torque_arg, hip_push_torque_arg,
        hip_push_start_arg, hip_push_stop_arg,
        body_force_arg, spawn_x_arg, spawn_pitch_arg,
        fall_threshold_arg,
        gazebo,
        robot_state_pub,
        joint_state_bridge,
        spawn_robot,
        kick_left,
        unpause,
        kick_left_stop,
        kick_right,
        kick_right_stop,
        release,
        hip_bias_node,
        analyser_node,
    ])
