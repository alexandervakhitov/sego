# Compile the sego (stereo egomotion) library and the tests

# Copyright (c) 2018 Alexander Vakhitov <alexander.vakhitov@gmail.com>
# Redistribution and use is allowed according to the terms of the GPL v 3 license.

cmake_minimum_required(VERSION 3.1)
project(sego)

set(CMAKE_CXX_STANDARD 11)
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -w  -O3 -march=native ") #-Wall
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O3 -march=native")

#add_definitions(-D_GLIBCXX_DEBUG)

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ./cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${EIGEN3_INCLUDE_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
        pluecker.cpp
        pluecker_assign.cpp
        quadric_solver.cpp
        sego.cpp)

target_link_libraries(${PROJECT_NAME}
        opencv_core
        opencv_highgui
        opencv_calib3d
        opencv_imgcodecs
        lapack
        ${EIGEN3_LIBS}
        )

add_executable(segotest main.cpp)

link_directories(${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(segotest
        opencv_core
        opencv_highgui
        opencv_calib3d
        opencv_imgcodecs
        lapack
        ${EIGEN3_LIBS}
        ${PROJECT_NAME}
        )
