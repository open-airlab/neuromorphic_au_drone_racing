cmake_minimum_required(VERSION 3.0.2)
project(event_preprocessor)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  dvs_msgs
  cv_bridge
  image_transport
)
find_package(Eigen3)
find_package(OpenCV)
find_package(cmake_modules REQUIRED)


## Generate messages in the 'msg' folder
add_message_files(
  DIRECTORY msg
  FILES
  HistogramStamped.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES image_tracking_controller
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime 
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${tf_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS} 
)

add_executable(event_preprocessor 
  src/event_preprocessor.cpp
  src/event_preprocessor_node.cpp
)

target_link_libraries(event_preprocessor ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})