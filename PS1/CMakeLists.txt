cmake_minimum_required(VERSION 3.5)
project(PS1)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp ClLoader.cpp ClLoader.h)
add_executable(PS1 ${SOURCE_FILES})

target_link_libraries(PS1 ${OPENCL_LIBRARY})