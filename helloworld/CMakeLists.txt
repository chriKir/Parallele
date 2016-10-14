cmake_minimum_required(VERSION 3.5)
project(helloworld)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCL REQUIRED)
message("OpenCL Version: ${OpenCL_VERSION_STRING}")

include_directories(${OpenCL_INCLUDE_DIR})

set(SOURCE_FILES main.cpp)

add_executable(helloworld ${SOURCE_FILES})

target_link_libraries(helloworld ${OpenCL_LIBRARIES})