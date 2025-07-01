cmake_minimum_required(VERSION 3.0.2)
project(turtle_vlm_gazebo)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  rospy
  std_msgs
  xacro
  turtlebot3_description
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
