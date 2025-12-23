cmake_minimum_required(VERSION 3.0.2)
project(livox_filter_projector)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  pcl_ros
  pcl_conversions
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(livox_filter_projector src/livox_filter_projector.cpp)
target_link_libraries(livox_filter_projector ${catkin_LIBRARIES})
