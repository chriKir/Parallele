cmake_minimum_required(VERSION 3.5)
project(PS3-1)

configure_file(jacobi.c jacobi.c COPYONLY)

#set( CMAKE_VERBOSE_MAKEFILE on )

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR})

set(SOURCE_FILES
        main.cpp
        ../include/ClWrapper.cpp
        ../include/ClWrapper.h)

add_executable(PS3-1 ${SOURCE_FILES})

set_property(TARGET PS3-1 PROPERTY LANGUAGE CXX)
set_property(TARGET PS3-1 PROPERTY CXX_STANDARD 11)
set_property(TARGET PS3-1 PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries(PS3-1 ${OpenCL_LIBRARY})