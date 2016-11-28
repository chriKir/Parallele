cmake_minimum_required(VERSION 3.5)
project(PS7)

# copy kernel
#configure_file(auto_levels.c auto_levels.c COPYONLY)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR})

set(serial
        serial/search_serial.c
        ../include/dSFMT.c
        )

#set(SOURCE_FILES
#        main.cpp
#        ../include/ClWrapper.cpp
#        ../include/stb_image.c
#        )

add_executable(search_serial ${serial})
#add_executable(auto_levels_ocl ${SOURCE_FILES})

#set_property(TARGET auto_levels_ocl PROPERTY LANGUAGE CXX)
#set_property(TARGET auto_levels_ocl PROPERTY CXX_STANDARD 11)
#set_property(TARGET auto_levels_ocl PROPERTY CXX_STANDARD_REQUIRED ON)

#if (APPLE)
#    target_link_libraries(auto_levels_ocl "-framework opencl")
#else()
#    target_link_libraries(auto_levels_ocl ${OpenCL_LIBRARY} m)
#endif (APPLE)

target_link_libraries(search_serial m)
