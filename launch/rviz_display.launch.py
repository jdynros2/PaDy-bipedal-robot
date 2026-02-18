import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ── Paths ──────────────────────────────────────────────────────────
    pkg_dir    = get_package_share_directory('pady_robot')
    urdf_path  = os.path.join(pkg_dir, 'urdf',   'pady_simplified.urdf')

    # Read URDF as string
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    # ── Launch Arguments ───────────────────────────────────────────────
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',  # false for RViz
        description='Use simulation time'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    # ── Robot State Publisher ───────────────────────────────────────
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    # ── Joint State Publisher GUI ───────────────────────────────────
    joint_state_pub = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen'
    )

    # ── RViz ────────────────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        robot_state_pub,
        joint_state_pub,
        rviz,
    ])