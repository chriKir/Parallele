cmake_minimum_required(VERSION 3.5)
project(PS1)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

set( CMAKE_VERBOSE_MAKEFILE on )

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR})

set(SOURCE_FILES
        main.cpp
        ../include/ClWrapper.cpp
        ../include/ClWrapper.h)

add_executable(PS1 ${SOURCE_FILES})

set_property(TARGET PS1 PROPERTY CXX_STANDARD 11)
set_property(TARGET PS1 PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries(PS1 ${OpenCL_LIBRARY})