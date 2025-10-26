cmake_minimum_required(VERSION 3.18)
project(CNDA LANGUAGES CXX)

# C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only
add_library(cnda_headers INTERFACE)
target_include_directories(cnda_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# Testing
include(CTest)
if (BUILD_TESTING)
  add_subdirectory(tests)
endif()
