cmake_minimum_required(VERSION 3.5)
project(PS8)

set(SOURCE_FILES_LIST_GEN list_gen.cpp)

set(SOURCE_FILES_LIST_SORT list_sort.cpp)


add_executable(list_gen ${SOURCE_FILES_LIST_GEN})
add_executable(list_sort ${SOURCE_FILES_LIST_SORT})