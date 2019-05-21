cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test_app)

find_package(Torch REQUIRED)

add_executable(test_app mnist.cpp)
target_link_libraries(test_app "${TORCH_LIBRARIES}")
set_property(TARGET test_app PROPERTY CXX_STANDARD 11)
