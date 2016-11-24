cmake_minimum_required(VERSION 3.5)
project(PS5)

configure_file(reduction.cl reduction.cl COPYONLY)

set(CMAKE_VERBOSE_MAKEFILE on)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR})


set(SOURCE_FILES
        main.cpp
        ../include/ClWrapper.cpp
        )

add_executable(PS5 ${SOURCE_FILES})

set_property(TARGET PS5 PROPERTY LANGUAGE CXX)
set_property(TARGET PS5 PROPERTY CXX_STANDARD 11)
set_property(TARGET PS5 PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries(PS5 ${OpenCL_LIBRARY})