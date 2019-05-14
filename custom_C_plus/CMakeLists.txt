cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(Data2Tensor)

find_package(Torch REQUIRED)
#find_package(OpenCV REQUIRED)

# Define our library target
add_library(Data2Tensor SHARED ../test.cpp)
# Enable C++11
target_compile_features(Data2Tensor PRIVATE cxx_range_for)
# Link against LibTorch
target_link_libraries(Data2Tensor "${TORCH_LIBRARIES}")
# Link against OpenCV
target_link_libraries(Data2Tensor ) #(opencv_core opencv_imgproc)
