cmake_minimum_required(VERSION 3.5)
project(PS2)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#set( CMAKE_VERBOSE_MAKEFILE on )

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(SOURCE_FILES
        main.cpp
        ../include/ClLoader.cpp
        ../include/ClLoader.h)

add_executable(PS2 ${SOURCE_FILES})

set_property(TARGET PS2 PROPERTY CXX_STANDARD 11)
set_property(TARGET PS2 PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries(PS2 ${OpenCL_LIBRARY})