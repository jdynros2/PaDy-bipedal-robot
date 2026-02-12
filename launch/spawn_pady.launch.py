import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ── Paths ──────────────────────────────────────────────────────────
    pkg_dir    = get_package_share_directory('pady_robot')
    urdf_path  = os.path.join(pkg_dir, 'urdf',   'pady.urdf')
    world_path = os.path.join(pkg_dir, 'worlds', 'slope_3deg.sdf')

    # Read URDF as string (needed by robot_state_publisher)
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    # ── Launch Arguments ───────────────────────────────────────────────
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    # ── 1. Start Gazebo with slope world ───────────────────────────────
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_path, '-v', '3'],
        output='screen'
    )

    # ── 2. Spawn robot on slope ────────────────────────────────────────
    # Position: top of slope, 1.05m above surface, tilted -3deg to match slope
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-string', robot_desc,          # pass URDF as string
            '-name',   'pady',              # robot name in Gazebo
            '-x',      '0',                 # start of slope
            '-y',      '0',                 # centered
            '-z',      '1.05',              # 1.05m above slope surface
            '-R',      '0',                 # no roll
            '-P',      '-0.0524',           # -3deg pitch to match slope
            '-Y',      '0',                 # facing down slope
        ],
        output='screen'
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

    # ── 4. Joint State Publisher ───────────────────────────────────────
    # Broadcasts joint states for visualisation
    joint_state_pub = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        gazebo,
        spawn_robot,
        robot_state_pub,
        joint_state_pub,
    ])
