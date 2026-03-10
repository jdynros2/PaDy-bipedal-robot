"""spawn_pady.launch.py — Main interactive simulation launch.

Use this launch file for manual tuning in Gazebo + RViz.

Gait initiation strategy
-------------------------
The robot spawns in a mid-walk stance (right leg forward / stance, left leg
back / swing with knee flexed, arms contralateral).  At unpause a single
bilateral kick pulse simultaneously pushes the body forward (via stance hip)
and accelerates the swing leg (via swing hip).  The continuous hip-push node
then sustains energy input to compensate heelstrike collision losses.

Where to adjust parameters
--------------------------
- Runtime tuning (CLI): kick/body-force/timing/spawn pose via launch args below.
- Initial joint offsets: baked into URDF joint origin rpy values (pady.urdf).
- Contact/friction/cant and joint limits: `urdf/pady.urdf`.
- Bridge topic mappings: `config/bridge.yaml`.
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

    world_arg = DeclareLaunchArgument(
        'world',
        default_value='slope_3deg.sdf',
        description='World SDF filename inside pady_robot/worlds')
    world_file = LaunchConfiguration('world')

    robot_desc_gz = robot_desc.replace(
        'package://pady_robot/',
        f'file://{pkg_dir}/'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation time')
    use_sim_time = LaunchConfiguration('use_sim_time')

    kick_torque_arg = DeclareLaunchArgument(
        'kick_torque', default_value='15.0',
        description='Left-hip kick magnitude (N·m, positive = swing leg forward)')
    kick_torque = LaunchConfiguration('kick_torque')

    kick_torque_right_arg = DeclareLaunchArgument(
        'kick_torque_right', default_value='-15.0',
        description='Right-hip kick magnitude (N·m, negative = body forward over stance foot)')
    kick_torque_right = LaunchConfiguration('kick_torque_right')

    kick_follow_torque_arg = DeclareLaunchArgument(
        'kick_follow_torque', default_value='0.0',
        description='(Legacy) Right-hip follow-through torque — unused in bilateral kick')
    kick_follow_torque = LaunchConfiguration('kick_follow_torque')

    hip_push_torque_arg = DeclareLaunchArgument(
        'hip_push_torque', default_value='3.0',
        description='Hip bias torque during swing (N·m)')
    hip_push_torque = LaunchConfiguration('hip_push_torque')

    hip_push_start_arg = DeclareLaunchArgument(
        'hip_push_start_time', default_value='0.0',
        description='Bias start time (s)')
    hip_push_start_time = LaunchConfiguration('hip_push_start_time')

    hip_push_stop_arg = DeclareLaunchArgument(
        'hip_push_stop_time', default_value='12.0',
        description='Bias end time (s)')
    hip_push_stop_time = LaunchConfiguration('hip_push_stop_time')

    body_force_arg = DeclareLaunchArgument(
        'body_force', default_value='5.0',
        description='Forward force on base link (N)')
    body_force = LaunchConfiguration('body_force')

    spawn_x_arg = DeclareLaunchArgument(
        'spawn_x', default_value='0.40',
        description='Initial x-coordinate')
    spawn_x = LaunchConfiguration('spawn_x')

    spawn_pitch_arg = DeclareLaunchArgument(
        'spawn_pitch', default_value='0.12',
        description='Initial forward pitch (rad, ~slope_angle + forward lean)')
    spawn_pitch = LaunchConfiguration('spawn_pitch')

    spawn_roll_arg = DeclareLaunchArgument(
        'spawn_roll', default_value='-0.06',
        description='Initial lateral roll (rad, negative = toward right stance foot)')
    spawn_roll = LaunchConfiguration('spawn_roll')

    rviz_config_arg = DeclareLaunchArgument(
        'rvizconfig',
        default_value=os.path.join(pkg_dir, 'rviz', 'pady.rviz'),
        description='Path to RViz configuration file')
    rviz_config = LaunchConfiguration('rvizconfig')

    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', [pkg_dir, '/worlds/', world_file]],
        output='screen'
    )

    spawn_robot = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-string', robot_desc_gz,
                    '-name', 'pady',
                    '-x', spawn_x,
                    '-y', '0',
                    '-z', '1.45',
                    '-R', spawn_roll,
                    '-P', spawn_pitch,
                    '-Y', '-0.01',
                ],
                output='screen'
            )
        ]
    )

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': use_sim_time,
        }],
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

    # ── Pre-lock knees via direct gz topic BEFORE unpause ─────────
    # ApplyJointForce latches the last value, so this holds through unpause.
    # Left knee locked (stance), right knee free (swing).
    knee_prelock = TimerAction(
        period=7.0,
        actions=[
            ExecuteProcess(
                cmd=['gz', 'topic', '-t', '/model/pady/joint/knee_joint_left/0/cmd_force',
                     '-m', 'gz.msgs.Double', '-p', 'data: -30.0'],
                output='screen'),
            ExecuteProcess(
                cmd=['gz', 'topic', '-t', '/model/pady/joint/knee_joint_right/0/cmd_force',
                     '-m', 'gz.msgs.Double', '-p', 'data: 0.0'],
                output='screen'),
        ],
    )

    unpause = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'gz', 'service',
                    '-s', '/world/slope_world/control',
                    '--reqtype', 'gz.msgs.WorldControl',
                    '--reptype', 'gz.msgs.Boolean',
                    '--timeout', '2000',
                    '--req', 'pause: false'
                ],
                output='screen'
            )
        ]
    )

    kick_data_left = [TextSubstitution(text='data: '), kick_torque]
    kick_data_right = [TextSubstitution(text='data: '), kick_torque_right]

    # ── Bilateral simultaneous kick: both hips at once ──────────────
    # Right (stance) hip: negative torque → body forward over stance foot.
    # Left (swing) hip:  positive torque → swing leg accelerates forward.
    # Duration ~0.35 s ≈ 60 % of corrected half-period (0.58 s).
    bilateral_kick = TimerAction(
        period=8.10,
        actions=[
            ExecuteProcess(
                cmd=['gz', 'topic', '-t', '/model/pady/joint/hip_joint_right/0/cmd_force',
                     '-m', 'gz.msgs.Double', '-p', kick_data_right],
                output='screen'),
            ExecuteProcess(
                cmd=['gz', 'topic', '-t', '/model/pady/joint/hip_joint_left/0/cmd_force',
                     '-m', 'gz.msgs.Double', '-p', kick_data_left],
                output='screen'),
        ],
    )

    kick_stop = TimerAction(
        period=8.45,
        actions=[
            ExecuteProcess(
                cmd=['gz', 'topic', '-t', '/model/pady/joint/hip_joint_right/0/cmd_force',
                     '-m', 'gz.msgs.Double', '-p', 'data: 0.0'],
                output='screen'),
            ExecuteProcess(
                cmd=['gz', 'topic', '-t', '/model/pady/joint/hip_joint_left/0/cmd_force',
                     '-m', 'gz.msgs.Double', '-p', 'data: 0.0'],
                output='screen'),
        ],
    )

    release = TimerAction(
        period=10.0,
        actions=[
            ExecuteProcess(
                cmd=['gz', 'topic', '-t', '/model/pady/joint/hip_joint_right/0/cmd_force',
                     '-m', 'gz.msgs.Double', '-p', 'data: 0.0'],
                output='screen'),
            ExecuteProcess(
                cmd=['gz', 'topic', '-t', '/model/pady/joint/hip_joint_left/0/cmd_force',
                     '-m', 'gz.msgs.Double', '-p', 'data: 0.0'],
                output='screen'),
        ],
    )

    hip_bias_node = TimerAction(
        period=7.7,
        actions=[Node(
            package='pady_robot',
            executable='continuous_hip_push.py',
            output='screen',
            parameters=[{
                'torque': hip_push_torque,
                'body_force': body_force,
                'start_time': hip_push_start_time,
                'stop_time': hip_push_stop_time,
                'rate': 50.0,
                'use_sim_time': use_sim_time,
            }],
        )],
    )

    knee_lock_node = TimerAction(
        period=6.0,
        actions=[Node(
            package='pady_robot',
            executable='knee_lock.py',
            name='knee_lock',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
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
        rviz_config_arg,
        gazebo,
        robot_state_pub,
        joint_state_bridge,
        spawn_robot,
        knee_prelock,
        unpause,
        bilateral_kick,
        kick_stop,
        release,
        hip_bias_node,
        knee_lock_node,
    ])
