cmake_minimum_required(VERSION 3.22)
project(veloxpm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17) # pybind11 通常推荐 C++17 或更高版本以获得更好的兼容性
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89")
find_package(Python REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(pybind11)

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

pybind11_add_module(_veloxpm_core ${SRC})

target_link_libraries(_veloxpm_core PRIVATE CUDA::cublas CUDA::cusolver)
target_compile_options(_veloxpm_core PRIVATE -O3)

install(TARGETS _veloxpm_core DESTINATION veloxpm)
