cmake_minimum_required(VERSION 3.6)
project(Parallele)

set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

# SYSTEM to suppress warnings from cl.cpp include.. split lib/include?
include_directories(SYSTEM "${PROJECT_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas")

add_subdirectory(PS1)
add_subdirectory(PS2)
add_subdirectory(PS3-1)
add_subdirectory(PS3-2)
add_subdirectory(PS5)
add_subdirectory(PS6)
add_subdirectory(PS7)
