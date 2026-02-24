import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ── Paths ──────────────────────────────────────────────────────────
    pkg_dir    = get_package_share_directory('pady_robot')
    urdf_path  = os.path.join(pkg_dir, 'urdf',   'pady_simplified.urdf')
    world_path = os.path.join(pkg_dir, 'worlds', 'slope_3deg.sdf')

    # Read URDF as string (needed by robot_state_publisher)
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    # When URDF is passed to Gazebo as a string (-string flag), Gazebo converts
    # 'package://' URIs to 'model://' which it then searches in its own model
    # database — not in ROS package paths — so all meshes fail to load.
    # Fix: replace 'package://pady_robot/' with the real absolute 'file://' URI
    # so Gazebo can locate the STL files directly.
    robot_desc_gz = robot_desc.replace(
        'package://pady_robot/',
        f'file://{pkg_dir}/'
    )

    # ── Launch Arguments ───────────────────────────────────────────────
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    # ── 1. Start Gazebo with slope world (server + GUI) ────────────────
    # NOTE: do NOT pass -s here; -s starts the server headless (no window).
    # Without -s, gz sim launches both the physics server and the GUI client.
    # gz sim 8 (Jazzy) does not accept --run via the Ruby wrapper.
    # The simulation starts paused; press the Play button in the GUI,
    # or run:  gz service -s /world/slope_world/control --reqtype gz.msgs.WorldControl
    #          --reptype gz.msgs.Boolean --timeout 2000 --req 'pause: false'
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', world_path, '-v', '3'],
        output='screen'
    )

    # ── 2. Spawn robot on slope ────────────────────────────────────────
    # Delayed by 5 s to allow Gazebo to fully initialise before spawning.
    # Position: top of slope, 1.05m above surface, tilted -3deg to match slope.
    spawn_robot = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-string', robot_desc_gz,       # absolute file:// URIs for Gazebo
                    '-name',   'pady',              # robot name in Gazebo
                    '-x',      '0.25',                 # start of slope
                    '-y',      '0',                 # centered
                    '-z',      '1.50',              # 1.50m: slope surface (0.274) + leg+foot (1.084) + 0.10m drop clearance
                    '-R',      '0',                 # no roll
                    '-P',      '0.0524',            # +3deg pitch to match downhill slope (slope goes down in +X)
                    '-Y',      '0',                 # facing down slope
                ],
                output='screen'
            )
        ]
    )

    # ── 3. Robot State Publisher ───────────────────────────────────────
    # Publishes TF transforms from URDF joint positions
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    # ── 4. ROS–Gazebo bridge for joint states ──────────────────────────
    # Bridges Gazebo joint state topic → ROS /joint_states so that
    # robot_state_publisher receives live joint angles from the simulation.
    # (joint_state_publisher publishes synthetic/zero states and must NOT
    #  be used alongside a running Gazebo simulation.)
    joint_state_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/world/slope_world/model/pady/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model'],
        remappings=[
            ('/world/slope_world/model/pady/joint_state', '/joint_states')
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        gazebo,
        robot_state_pub,
        joint_state_bridge,
        spawn_robot,        # spawns after 5 s delay
    ])
