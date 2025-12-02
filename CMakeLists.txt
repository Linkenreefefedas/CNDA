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
find_package(Python 3.9...4.0 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Option for bounds checking
option(CNDA_BOUNDS_CHECK "Enable bounds checking" OFF)

# Create Python module
# Main bindings + AoS bindings separated for better organization
pybind11_add_module(cnda 
    python/module.cpp
    python/aos_types.cpp
)

target_link_libraries(cnda PRIVATE cnda_headers)

# Define version macro
target_compile_definitions(cnda PRIVATE CNDA_VERSION="${PROJECT_VERSION}")

# Add CNDA_BOUNDS_CHECK if enabled
if(CNDA_BOUNDS_CHECK)
  target_compile_definitions(cnda PRIVATE CNDA_BOUNDS_CHECK)
endif()

# Set output directory (configurable from outside, e.g. setup.py)
if (DEFINED CNDA_PYTHON_OUTPUT_DIR)
  # for setup.py 
  set(_cnda_output_dir "${CNDA_PYTHON_OUTPUT_DIR}")
else()
  # CMake build default：build/python
  set(_cnda_output_dir "${CMAKE_BINARY_DIR}/python")
endif()

set_target_properties(cnda PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${_cnda_output_dir}"
)

# On Windows with multi-config generators (VS), prevent adding /Release or /Debug subdirectory
# when CNDA_PYTHON_OUTPUT_DIR is set (i.e., when building via setup.py)
if (DEFINED CNDA_PYTHON_OUTPUT_DIR)
  foreach(config_type ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${config_type} config_type_upper)
    set_target_properties(cnda PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY_${config_type_upper} "${_cnda_output_dir}"
    )
  endforeach()
endif()

message(STATUS "Python ${Python_VERSION} found")
message(STATUS "pybind11 ${pybind11_VERSION} found")
message(STATUS "Building Python bindings for cnda")

# Benchmarks (optional)
option(BUILD_BENCHMARKS "Build benchmark executables" OFF)
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Testing
include(CTest)
if (BUILD_TESTING)
  # C++ tests
  add_subdirectory(tests)
  
  # Python tests (requires pytest)
  if (Python_FOUND)
    # Check if pytest is available
    execute_process(
      COMMAND ${Python_EXECUTABLE} -m pytest --version
      RESULT_VARIABLE PYTEST_RESULT
      OUTPUT_QUIET
      ERROR_QUIET
    )
    
    if (PYTEST_RESULT EQUAL 0)
      message(STATUS "pytest found - Python tests enabled")
      
      # Add Python binding tests
      # Python bindings always perform bounds checking regardless of CNDA_BOUNDS_CHECK
      # Therefore, all Python tests should always run
      add_test(
        NAME test_python_bindings
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=$<TARGET_FILE_DIR:cnda>"
                ${Python_EXECUTABLE} -m pytest 
                ${CMAKE_SOURCE_DIR}/tests/ -v
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      )
    else()
      message(WARNING "pytest not found - Python tests disabled. Install with: pip install pytest")
    endif()
  endif()
endif()

# Benchmarks (optional)
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
