cmake_minimum_required(VERSION 3.10)
project(lidar_calibration)

set(CMAKE_CXX_STANDARD 17)

find_package(PCL 1.10 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(lidar_calibration lidar_calibration.cpp)
target_link_libraries(lidar_calibration ${PCL_LIBRARIES})

add_executable(merge_clouds merge_clouds.cpp)
target_link_libraries(merge_clouds ${PCL_LIBRARIES})