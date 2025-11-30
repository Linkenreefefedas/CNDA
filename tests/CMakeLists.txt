include(FetchContent)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.4
)
FetchContent_MakeAvailable(catch2)

# Core C++ tests
add_executable(test_cpp_core 
  cpp/core/test_sanity.cpp 
  cpp/core/test_dtypes.cpp
  cpp/core/test_variadic_indexing.cpp
  cpp/core/test_alignment.cpp
  cpp/core/test_move_semantics.cpp
)
target_link_libraries(test_cpp_core PRIVATE Catch2::Catch2WithMain cnda_headers)

if(CNDA_BOUNDS_CHECK)
  target_compile_definitions(test_cpp_core PRIVATE CNDA_BOUNDS_CHECK)
endif()

# AoS C++ tests
add_executable(test_cpp_aos 
  cpp/aos/test_basic.cpp
  cpp/aos/test_indexing.cpp
  cpp/aos/test_field_layout.cpp
)
target_link_libraries(test_cpp_aos PRIVATE Catch2::Catch2WithMain cnda_headers)

if(CNDA_BOUNDS_CHECK)
  target_compile_definitions(test_cpp_aos PRIVATE CNDA_BOUNDS_CHECK)
endif()

# CTest integration
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(test_cpp_core)
catch_discover_tests(test_cpp_aos)
