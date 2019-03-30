cmake_minimum_required(VERSION 3.1)
project(pupil_utils)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11 -O3)

set(CATKIN_PACKAGE_DEPENDS
    std_msgs
    sensor_msgs
    gazebo_msgs
    geometry_msgs
    trajectory_msgs
    pupil_msgs
    ros_utils
    cv_bridge
    image_transport
    roscpp
    roslib
    rostopic
    gazebo_ros
    kinova_msgs
    kinova_driver
    kinova_description
    kdl_parser
)

## note: This depends on the location of the current developer OpenCV installation folder
set(Eigen3_INCLUDE_DIRS /usr/include/eigen3)
find_package(OpenCV 3.2 REQUIRED)

find_package(catkin REQUIRED COMPONENTS ${CATKIN_PACKAGE_DEPENDS})

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS ${CATKIN_PACKAGE_DEPENDS}
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

####============================================================================================####
####                           PUPIL_UTILS LIBRARY                                              ####
####============================================================================================####

set(LIBRARY_NAME "${PROJECT_NAME}")
set(CURRENT_NODE_DIR "src/nodelets")
file(GLOB CURRENT_NODE_SRC_FILES  "${CURRENT_NODE_DIR}/*.cc" )
add_library(${LIBRARY_NAME} ${CURRENT_NODE_SRC_FILES})
target_link_libraries( ${LIBRARY_NAME}
    ${OpenCL_LIBRARY}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    realsense
    sfml-system
    sfml-audio
)

####============================================================================================####
####                                     Detect Camera NODE                                     ####
####============================================================================================####

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages  don't collide
set(CURRENT_NODE_NAME "rs_detect")
set(CURRENT_NODE_DIR "src/${CURRENT_NODE_NAME}")
file(GLOB CURRENT_NODE_SRC_FILES "src/${CURRENT_NODE_NAME}/*.cc")
add_executable("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    ${CURRENT_NODE_SRC_FILES}
    ${LIBRARY_SRC_FILES}
)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
set_target_properties("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    PROPERTIES OUTPUT_NAME ${CURRENT_NODE_NAME} PREFIX ""
)

## Specify libraries to link a library or executable target against
target_link_libraries("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    realsense
)


####============================================================================================####
####                                     SAVE Camera NODE                                       ####
####============================================================================================####

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
set(CURRENT_NODE_NAME "rs_save")
set(CURRENT_NODE_DIR "src/${CURRENT_NODE_NAME}")
file(GLOB CURRENT_NODE_SRC_FILES "${CURRENT_NODE_DIR}/*.cc")
add_executable("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    ${CURRENT_NODE_SRC_FILES}
    ${LIBRARY_SRC_FILES}
)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
set_target_properties("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    PROPERTIES OUTPUT_NAME ${CURRENT_NODE_NAME} PREFIX ""
)

## Specify libraries to link a library or executable target against
target_link_libraries("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    ${catkin_LIBRARIES}
    realsense
)

####============================================================================================####
####                                     Detect Camera NODE                                     ####
####============================================================================================####

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages  don't collide
set(CURRENT_NODE_NAME "depth_save")
set(CURRENT_NODE_DIR "src/${CURRENT_NODE_NAME}")
file(GLOB CURRENT_NODE_SRC_FILES "src/${CURRENT_NODE_NAME}/*.cc")
add_executable("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    ${CURRENT_NODE_SRC_FILES}
    ${LIBRARY_SRC_FILES}
)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
set_target_properties("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    PROPERTIES OUTPUT_NAME ${CURRENT_NODE_NAME} PREFIX ""
)

## Specify libraries to link a library or executable target against
target_link_libraries("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    realsense
)

####============================================================================================####
####                                    SPACENAV_CONTROL NODE                                   ####
####============================================================================================####

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
set(CURRENT_NODE_NAME "spacenav_control")
set(CURRENT_NODE_DIR "src/${CURRENT_NODE_NAME}")
file(GLOB CURRENT_NODE_SRC_FILES "${CURRENT_NODE_DIR}/*.cc")
add_executable("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    ${CURRENT_NODE_SRC_FILES}
    ${LIBRARY_SRC_FILES}
)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
set_target_properties("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    PROPERTIES OUTPUT_NAME ${CURRENT_NODE_NAME} PREFIX ""
)

## Specify libraries to link a library or executable target against
target_link_libraries("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    ${catkin_LIBRARIES}
    realsense
)


####============================================================================================####
####                                    GAZE_FAKER NODE                                         ####
####============================================================================================####

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
set(CURRENT_NODE_NAME "gaze_faker")
set(CURRENT_NODE_DIR "src/${CURRENT_NODE_NAME}")
file(GLOB CURRENT_NODE_SRC_FILES "${CURRENT_NODE_DIR}/*.cc")
add_executable("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    ${CURRENT_NODE_SRC_FILES}
    ${LIBRARY_SRC_FILES}
)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
set_target_properties("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    PROPERTIES OUTPUT_NAME ${CURRENT_NODE_NAME} PREFIX ""
)

## Specify libraries to link a library or executable target against
target_link_libraries("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    realsense
)

####============================================================================================####
####                                    JACO_CONTROL NODE                                         ####
####============================================================================================####

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
set(CURRENT_NODE_NAME "jaco_control")
set(CURRENT_NODE_DIR "src/${CURRENT_NODE_NAME}")
file(GLOB CURRENT_NODE_SRC_FILES "${CURRENT_NODE_DIR}/*.cpp")
add_executable("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    ${CURRENT_NODE_SRC_FILES}
)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
set_target_properties("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    PROPERTIES OUTPUT_NAME ${CURRENT_NODE_NAME} PREFIX ""
)

## Specify libraries to link a library or executable target against
target_link_libraries("${PROJECT_NAME}_${CURRENT_NODE_NAME}"
    ${catkin_LIBRARIES}
)
