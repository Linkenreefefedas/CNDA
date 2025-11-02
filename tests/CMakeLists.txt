include(FetchContent)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.4
)
FetchContent_MakeAvailable(catch2)

add_executable(test_sanity test_sanity.cpp)
target_link_libraries(test_sanity PRIVATE Catch2::Catch2WithMain cnda_headers)
target_compile_definitions(test_sanity PRIVATE CNDA_BOUNDS_CHECK)

# 讓 CTest 能發現並執行
include(Catch)
catch_discover_tests(test_sanity)
