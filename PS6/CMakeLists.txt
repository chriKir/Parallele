cmake_minimum_required(VERSION 3.5)
project(PS6)

configure_file(test.png test.png COPYONLY)
configure_file(test_min.png test_min.png COPYONLY)
configure_file(earth_huge.png earth_huge.png COPYONLY)

configure_file(auto_levels.cl auto_levels.cl COPYONLY)

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
        ../include/stb_image.c
        )

add_executable(auto_levels_serial ${serial})
add_executable(auto_levels_ocl ${SOURCE_FILES})

set_property(TARGET auto_levels_ocl PROPERTY LANGUAGE CXX)
set_property(TARGET auto_levels_ocl PROPERTY CXX_STANDARD 11)
set_property(TARGET auto_levels_ocl PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries(auto_levels_serial ${OpenCL_LIBRARY} m)
target_link_libraries(auto_levels_ocl ${OpenCL_LIBRARY})
