"""Lightweight URDF viewer — robot_state_publisher + joint_state_publisher_gui + RViz2.
    Use this for quick geometry/joint checks without Gazebo physics.

    What to tune here vs elsewhere
    -------------------------------
    - Tune link/joint geometry, limits, and visuals in `urdf/pady.urdf`.
    - Tune gait timing/forces in launch files (`spawn_pady` / `headless`).

   Usage (from a regular terminal, NOT VS Code's snap terminal):
     source /opt/ros/jazzy/setup.bash
     source ~/ros2_ws/install/setup.bash
     ros2 launch pady_robot view_urdf.launch.py
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory('pady_robot')
    urdf_file = os.path.join(pkg, 'urdf', 'pady.urdf')
    rviz_cfg = os.path.join(pkg, 'rviz', 'urdf_viewer.rviz')

    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}],
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_cfg],
        ),
    ])
