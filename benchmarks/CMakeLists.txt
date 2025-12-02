# CNDA Benchmarks
cmake_minimum_required(VERSION 3.18)

message(STATUS "Building CNDA benchmarks")

# Ensure Release build for benchmarks
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find Catch2 (should already be available from tests)
find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
    message(STATUS "Catch2 not found, fetching...")
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.0
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# ============================================================================
# Core Benchmarks
# ============================================================================
add_executable(bench_core bench_core.cpp)
target_link_libraries(bench_core PRIVATE cnda_headers Catch2::Catch2WithMain)
# Require C++17 for structured bindings
set_target_properties(bench_core PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
# IMPORTANT: Do NOT define CNDA_BOUNDS_CHECK for benchmarks (maximize performance)
target_compile_options(bench_core PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /DNDEBUG>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -DNDEBUG>
)

# Register as CTest
add_test(NAME benchmark_core COMMAND bench_core)

# ============================================================================
# AoS Benchmarks
# ============================================================================
add_executable(bench_aos bench_aos.cpp)
target_link_libraries(bench_aos PRIVATE cnda_headers Catch2::Catch2WithMain)
# Require C++17 for structured bindings
set_target_properties(bench_aos PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
# IMPORTANT: Do NOT define CNDA_BOUNDS_CHECK for benchmarks (maximize performance)
target_compile_options(bench_aos PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /DNDEBUG>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -DNDEBUG>
)

# Register as CTest
add_test(NAME benchmark_aos COMMAND bench_aos)

# ============================================================================
# Comparison Benchmarks
# ============================================================================
add_executable(bench_comparison bench_comparison.cpp)
target_link_libraries(bench_comparison PRIVATE cnda_headers Catch2::Catch2WithMain)
# Require C++17 for structured bindings
set_target_properties(bench_comparison PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
# IMPORTANT: Do NOT define CNDA_BOUNDS_CHECK for benchmarks (maximize performance)
target_compile_options(bench_comparison PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /DNDEBUG>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -DNDEBUG>
)

# Register as CTest
add_test(NAME benchmark_comparison COMMAND bench_comparison)

# ============================================================================
# Python Benchmarks (requires pytest-benchmark)
# ============================================================================
if(Python_FOUND)
    # Check if pytest-benchmark is available
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pytest_benchmark"
        RESULT_VARIABLE PYTEST_BENCHMARK_FOUND
        OUTPUT_QUIET
        ERROR_QUIET
    )
    
    if(PYTEST_BENCHMARK_FOUND EQUAL 0)
        message(STATUS "pytest-benchmark found - Python benchmarks enabled")
        
        add_test(
            NAME benchmark_numpy_interop
            COMMAND ${CMAKE_COMMAND} -E env
                    "PYTHONPATH=$<TARGET_FILE_DIR:cnda>"
                    ${Python_EXECUTABLE} -m pytest 
                    ${CMAKE_CURRENT_SOURCE_DIR}/bench_numpy_interop.py
                    -v --benchmark-only
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
        
    else()
        message(WARNING "pytest-benchmark not found. Install with: pip install pytest-benchmark")
    endif()
endif()

# ============================================================================
# Benchmark Summary Script
# ============================================================================
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/run_all_benchmarks.sh.in
    ${CMAKE_CURRENT_BINARY_DIR}/run_all_benchmarks.sh
    @ONLY
)
