cmake_minimum_required(VERSION 3.5)
project(PS5)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#set( CMAKE_VERBOSE_MAKEFILE on )

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(SOURCE_FILES
        main.cpp
        ../include/time_ms.h
        ../include/ClLoader.cpp
        ../include/ClLoader.h)

add_executable(PS5 ${SOURCE_FILES})

set_property(TARGET PS5 PROPERTY LANGUAGE CXX)
set_property(TARGET PS5 PROPERTY CXX_STANDARD 11)
set_property(TARGET PS5 PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries(PS5 ${OpenCL_LIBRARY})