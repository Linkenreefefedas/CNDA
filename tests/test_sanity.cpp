#include <catch2/catch_test_macros.hpp>
#include <cnda/contiguous_nd.hpp>
#include <vector>

TEST_CASE("header compiles and dummy instance constructs", "[sanity]") {
    cnda::ContiguousND<float> a({3, 4});
    REQUIRE(a.size() == 12);
}

TEST_CASE("row-major strides for 2D (3x4)", "[shape][stride]") {
    cnda::ContiguousND<double> a({3, 4});
    REQUIRE(a.ndim() == 2);
    REQUIRE(a.shape()[0] == 3);
    REQUIRE(a.shape()[1] == 4);

    // row-major in ELEMENTS: [4, 1]
    REQUIRE(a.strides().size() == 2);
    REQUIRE(a.strides()[0] == 4);
    REQUIRE(a.strides()[1] == 1);

    // Acceptance: size()==12
    REQUIRE(a.size() == 12);
}

TEST_CASE("row-major strides for 3D (2x3x4)", "[stride]") {
    cnda::ContiguousND<int> a({2, 3, 4});
    // Expected strides (elements): [3*4, 4, 1] = [12, 4, 1]
    REQUIRE(a.strides().size() == 3);
    REQUIRE(a.strides()[0] == 12);
    REQUIRE(a.strides()[1] == 4);
    REQUIRE(a.strides()[2] == 1);
    REQUIRE(a.size() == 2 * 3 * 4);
}

TEST_CASE("zero-sized dimension yields size=0 but valid metadata", "[edge]") {
    cnda::ContiguousND<float> a({0, 7});
    REQUIRE(a.size() == 0);
    REQUIRE(a.ndim() == 2);
    REQUIRE(a.strides()[0] == 7);
    REQUIRE(a.strides()[1] == 1);
    (void)a.data();
    SUCCEED("data() is callable on zero-sized array");
}
