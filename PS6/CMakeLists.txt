cmake_minimum_required(VERSION 3.5)
project(auto_levels_ocl)

configure_file(test.png test.png COPYONLY)
configure_file(earth_huge.png earth_huge.png COPYONLY)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR})

set(serial
        serial/auto_levels.c
        ../include/stb_image.c
        ../include/stb_image_write.c
        )

set(SOURCE_FILES
        main.cpp
        ../include/ClWrapper.cpp
        )

add_executable(serial ${serial})
add_executable(auto_levels_ocl ${SOURCE_FILES})

set_property(TARGET auto_levels_ocl PROPERTY LANGUAGE CXX)
set_property(TARGET auto_levels_ocl PROPERTY CXX_STANDARD 11)
set_property(TARGET auto_levels_ocl PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries(serial ${OpenCL_LIBRARY} m)
target_link_libraries(auto_levels_ocl ${OpenCL_LIBRARY})
