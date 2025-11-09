cmake_minimum_required(VERSION 3.18)
project(CNDA VERSION 0.1.0 LANGUAGES CXX)

# C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only
add_library(cnda_headers INTERFACE)
target_include_directories(cnda_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# Python bindings (required)
find_package(Python 3.9 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Create Python module
pybind11_add_module(cnda python/cnda_bindings.cpp)

target_link_libraries(cnda PRIVATE cnda_headers)

# Define version macro
target_compile_definitions(cnda PRIVATE CNDA_VERSION="${PROJECT_VERSION}")

# Set output directory
set_target_properties(cnda PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)

message(STATUS "Python ${Python_VERSION} found")
message(STATUS "pybind11 ${pybind11_VERSION} found")
message(STATUS "Building Python bindings for cnda")

# Testing
include(CTest)
if (BUILD_TESTING)
  add_subdirectory(tests)
endif()
