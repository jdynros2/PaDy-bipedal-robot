cmake_minimum_required(VERSION 3.8)
project(pady_robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Install all robot description directories
install(DIRECTORY
  urdf
  meshes
  worlds
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# install helper Python scripts so they can be run with `ros2 run`
install(PROGRAMS
  scripts/continuous_hip_push.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
