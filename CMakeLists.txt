cmake_minimum_required(VERSION 3.8)
project(my_package)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(shooter_node src/shooter_node.cpp)
add_executable(vision_node src/vision_node.cpp)

ament_target_dependencies(shooter_node
  rclcpp
  geometry_msgs
  sensor_msgs
  vision_msgs
  std_msgs
)
ament_target_dependencies(vision_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
  cv_bridge
  image_transport
  OpenCV
)

target_link_libraries(vision_node
  ${OpenCV_LIBS}
)

install(TARGETS
  shooter_node
  vision_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(
  rclcpp 
  sensor_msgs 
  geometry_msgs 
  cv_bridge 
  image_transport 
  OpenCV 
  std_msgs 
  vision_msgs
)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
