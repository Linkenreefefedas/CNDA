#include <catch2/catch_test_macros.hpp>
#include <cnda/contiguous_nd.hpp>
#include <vector>

TEST_CASE("header compiles and a dummy instance constructs", "[sanity]") {
    cnda::ContiguousND<float> a({3, 4});
    REQUIRE(true); // 先放一個空斷言，確保流程打通
}
