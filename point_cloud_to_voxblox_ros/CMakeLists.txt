cmake_minimum_required(VERSION 3.16)
project(point_cloud_to_voxblox_ros)

set(CMAKE_CXX_STANDARD 17)
find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

catkin_package()

find_package(Boost REQUIRED COMPONENTS system)  # Whichever components you need
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(Boost REQUIRED COMPONENTS thread)

#############
# Libraries #
#############

#cs_add_library(${PROJECT_NAME})

###############
# Executables #
###############
cs_add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

###########
# Export #
###########

cs_install()
cs_export()
