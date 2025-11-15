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

message(STATUS "Python ${Python_VERSION} found")
message(STATUS "pybind11 ${pybind11_VERSION} found")
message(STATUS "Building Python bindings for cnda")

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
      
      # Set PYTHONPATH for tests to find the built module
      if (WIN32)
        set(PYTHON_TEST_PATH "${_cnda_output_dir}/$<CONFIG>")
      else()
        set(PYTHON_TEST_PATH "${_cnda_output_dir}")
      endif()
      
      # Add Python binding tests
      add_test(
        NAME test_python_bindings
        COMMAND ${Python_EXECUTABLE} -m pytest 
                ${CMAKE_SOURCE_DIR}/tests/test_python_bindings.py -v
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      )
      
      add_test(
        NAME test_numpy_interop
        COMMAND ${Python_EXECUTABLE} -m pytest 
                ${CMAKE_SOURCE_DIR}/tests/test_numpy_interop.py -v
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      )
      
      # Set PYTHONPATH environment for tests
      set_tests_properties(test_python_bindings test_numpy_interop
        PROPERTIES
          ENVIRONMENT "PYTHONPATH=${PYTHON_TEST_PATH}"
      )
    else()
      message(WARNING "pytest not found - Python tests disabled. Install with: pip install pytest")
    endif()
  endif()
endif()
