#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <cnda/contiguous_nd.hpp>
#include <cstdint>

TEMPLATE_TEST_CASE("ContiguousND supports common dtypes", "[dtype]",
                   float, double, std::int32_t, std::int64_t)
{
    // 2D basic: shape, strides, size
    cnda::ContiguousND<TestType> a({3, 4});
    REQUIRE(a.ndim() == 2);
    REQUIRE(a.size() == static_cast<std::size_t>(12));
    REQUIRE(a.shape()[0] == 3);
    REQUIRE(a.shape()[1] == 4);
    REQUIRE(a.strides()[0] == 4);
    REQUIRE(a.strides()[1] == 1);

    // write/read using operator()(i,j)
    TestType v = static_cast<TestType>(42);
    a(1, 2) = v;
    REQUIRE(a(1, 2) == v);

    // cross-check index() mapping
    auto off = a.index({1, 2});
    REQUIRE(off == 6);

    // 1D case
    cnda::ContiguousND<TestType> b({5});
    for (std::size_t i = 0; i < 5; ++i) b(i) = static_cast<TestType>(i);
    REQUIRE(b(4) == static_cast<TestType>(4));

    // 3D write/read
    cnda::ContiguousND<TestType> c({2, 3, 4});
    TestType vv = static_cast<TestType>(7);
    c(1, 1, 2) = vv;
    REQUIRE(c(1, 1, 2) == vv);
}
