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
    urdf_path = os.path.join(pkg_dir, 'urdf', 'pady.urdf')
    world_path = os.path.join(pkg_dir, 'worlds', 'slope_3deg.sdf')

    # Convert package:// URIs to file:// for Gazebo mesh loading
    robot_desc_gz = robot_desc.replace(
        'package://pady_robot/',
        f'file://{pkg_dir}/'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation time')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Gait initiation parameters (command-line tunable)
    kick_torque_arg = DeclareLaunchArgument(
        'kick_torque', default_value='30.0',
        description='Hip kick magnitude (N·m)')
    kick_torque = LaunchConfiguration('kick_torque')

    hip_push_torque_arg = DeclareLaunchArgument(
        'hip_push_torque', default_value='5.0',
        description='Hip bias torque during swing (N·m)')
    hip_push_torque = LaunchConfiguration('hip_push_torque')

    hip_push_start_arg = DeclareLaunchArgument(
        'hip_push_start_time', default_value='7.0',
        description='Bias start time (s)')
    hip_push_start_time = LaunchConfiguration('hip_push_start_time')

    hip_push_stop_arg = DeclareLaunchArgument(
        'hip_push_stop_time', default_value='12.0',
        description='Bias end time (s)')
    hip_push_stop_time = LaunchConfiguration('hip_push_stop_time')

    body_force_arg = DeclareLaunchArgument(
        'body_force', default_value='20.0',
        description='Forward force on base link (N)')
    body_force = LaunchConfiguration('body_force')

    spawn_x_arg = DeclareLaunchArgument(
        'spawn_x', default_value='0.45',
        description='Initial x-coordinate')
    spawn_x = LaunchConfiguration('spawn_x')
    spawn_pitch_arg = DeclareLaunchArgument(
        'spawn_pitch', default_value='0.275',
        description='Initial forward pitch (rad)')
    spawn_pitch = LaunchConfiguration('spawn_pitch')

    # RViz configuration (launch together with Gazebo)
    rviz_config_arg = DeclareLaunchArgument(
        'rvizconfig',
        default_value=os.path.join(pkg_dir, 'rviz', 'pady.rviz'),
        description='Path to RViz configuration file')
    rviz_config = LaunchConfiguration('rvizconfig')

    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', world_path],
        output='screen'
    )

    # Robot spawn with initial pose
    # Arms set contralateral to legs: right arm back (left leg forward), left arm forward (right leg forward)
    spawn_robot = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-string', robot_desc_gz,
                    '-name',   'pady',
                    '-x',      spawn_x,
                    '-y',      '0',
                    '-z',      '1.42',
                    '-R',      '-0.2',
                    '-P',      spawn_pitch,  # forward lean to initiate fall onto slope
                    '-Y',      '0',
                    '-J', 'hip_joint_right',  '0.55',   # stance: foot 0.56m ahead of hip
                    '-J', 'hip_joint_left',   '-0.50',  # swing: wide back for pendular energy
                    '-J', 'knee_joint_right',  '0.02',  # stance knee: at lower limit, locks on contact
                    '-J', 'knee_joint_left',   '1.05',  # swing knee: 46 mm ground clearance
                    '-J', 'arm_joint_right',  '-0.40',  # right arm back (contralateral to right leg forward)
                    '-J', 'arm_joint_left',    '0.40',  # left arm forward (contralateral to left leg back)
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

    # Collins-style gait kicks (hips only — arms swing passively via gravity)
    kick_data = [
        TextSubstitution(text='data: '),
        kick_torque,
    ]

    kick_left = TimerAction(
        period=8.3,
        actions=[
            ExecuteProcess(
                cmd=['gz', 'topic', '-t', '/model/pady/joint/hip_joint_left/0/cmd_force',
                     '-m', 'gz.msgs.Double', '-p', kick_data],
                output='screen'),
        ],
    )

    kick_left_stop = TimerAction(
        period=8.6,
        actions=[
            ExecuteProcess(
                cmd=['gz', 'topic', '-t', '/model/pady/joint/hip_joint_left/0/cmd_force',
                     '-m', 'gz.msgs.Double', '-p', 'data: 0.0'],
                output='screen'),
        ],
    )

    kick_right = TimerAction(
        period=9.8,
        actions=[
            ExecuteProcess(
                cmd=['gz', 'topic',
                     '-t', '/model/pady/joint/hip_joint_right/0/cmd_force',
                     '-m', 'gz.msgs.Double',
                     '-p', kick_data],
                output='screen'),
        ]
    )

    kick_right_stop = TimerAction(
        period=10.1,
        actions=[
            ExecuteProcess(
                cmd=['gz', 'topic', '-t', '/model/pady/joint/hip_joint_right/0/cmd_force',
                     '-m', 'gz.msgs.Double', '-p', 'data: 0.0'],
                output='screen'),
        ],
    )

    release = TimerAction(
        period=12.0,
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
        period=9.0,
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

    return LaunchDescription([
        use_sim_time_arg,
        kick_torque_arg,
        hip_push_torque_arg,
        hip_push_start_arg,
        hip_push_stop_arg,
        body_force_arg,
        spawn_x_arg,
        spawn_pitch_arg,
        rviz_config_arg,
        gazebo,
        robot_state_pub,
        joint_state_bridge,
        spawn_robot,       # T=5.0s:  spawn (paused)
        unpause,           # T=8.0s:  physics starts
        kick_left,         # T=8.3s:  left hip kick (300ms after unpause)
        kick_left_stop,    # T=8.6s:  end left kick (300ms burst)
        kick_right,        # T=9.8s:  right hip kick (1.5s swing period)
        kick_right_stop,   # T=10.1s: end right kick (300ms burst)
        release,           # T=12.0s: safety zero hips
        hip_bias_node,     # python node handles persistent torque during swing
    ])
