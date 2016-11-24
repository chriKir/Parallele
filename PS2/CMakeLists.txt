cmake_minimum_required(VERSION 3.5)
project(PS2)

configure_file(matrix.cl matrix.cl COPYONLY)

#set( CMAKE_VERBOSE_MAKEFILE on )

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR})

set(SOURCE_FILES
        main.cpp
        ../include/ClWrapper.cpp
        ../include/ClWrapper.h)

add_executable(PS2 ${SOURCE_FILES})

set_property(TARGET PS2 PROPERTY CXX_STANDARD 11)
set_property(TARGET PS2 PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries(PS2 ${OpenCL_LIBRARY})