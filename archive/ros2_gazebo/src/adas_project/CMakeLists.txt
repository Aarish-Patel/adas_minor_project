cmake_minimum_required(VERSION 3.8)
project(adas_project)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch scripts urdf worlds rviz models
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
  scripts/behavior_generator.py
  scripts/dataset_exporter.py
  scripts/train_model.py
  scripts/inference_node.py
  scripts/fixed_adas.py
  scripts/ml_adas.py
  scripts/evaluator.py
  scripts/scenario_controller.py
  scripts/manual_driver.py
  scripts/traffic_behavior_node.py
  scripts/control_arbitration_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
