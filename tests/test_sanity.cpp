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

TEST_CASE("index() computes expected row-major offset (2D)", "[index]") {
    cnda::ContiguousND<int> a({3, 4});
    // For shape [3,4], strides = [4,1], so (1,2) => 1*4 + 2*1 = 6
    REQUIRE(a.index({1, 2}) == 6);
    REQUIRE(a.index({2, 3}) == 11);
}

TEST_CASE("operator() reads/writes 1D/2D/3D", "[op]") {
    SECTION("1D") {
        cnda::ContiguousND<int> a({5});
        for (std::size_t i = 0; i < 5; ++i) a(i) = static_cast<int>(i * 10);
        REQUIRE(a(0) == 0);
        REQUIRE(a(4) == 40);
    }
    SECTION("2D") {
        cnda::ContiguousND<int> a({3, 4});
        int v = 0;
        for (std::size_t i = 0; i < 3; ++i)
          for (std::size_t j = 0; j < 4; ++j)
            a(i, j) = v++;
        REQUIRE(a(1, 2) == 6);  // acceptance: a(1,2) retrieves expected element
        REQUIRE(a(2, 3) == 11);
    }
    SECTION("3D") {
        cnda::ContiguousND<int> a({2, 3, 4});
        int v = 0;
        for (std::size_t i = 0; i < 2; ++i)
          for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 4; ++k)
              a(i, j, k) = v++;
        // With strides [12,4,1], (1,1,2) => 1*12 + 1*4 + 2 = 18
        REQUIRE(a(1, 1, 2) == 18);
    }
}

TEST_CASE("operator() N-dimensional variadic access", "[op][ndim]") {
    SECTION("4D variadic") {
        cnda::ContiguousND<int> a({2, 3, 4, 5});
        // strides: [60, 20, 5, 1], total size: 2*3*4*5 = 120
        REQUIRE(a.size() == 120);
        REQUIRE(a.strides()[0] == 60);
        REQUIRE(a.strides()[1] == 20);
        REQUIRE(a.strides()[2] == 5);
        REQUIRE(a.strides()[3] == 1);
        
        // Test read/write access
        a(0, 0, 0, 0) = 1000;
        a(1, 2, 3, 4) = 9999;
        a(0, 1, 2, 3) = 5555;
        
        REQUIRE(a(0, 0, 0, 0) == 1000);
        REQUIRE(a(1, 2, 3, 4) == 9999);
        REQUIRE(a(0, 1, 2, 3) == 5555);
        
        // Verify index calculation: (1,2,3,4) = 1*60 + 2*20 + 3*5 + 4*1 = 119
        std::size_t expected_idx = 1 * 60 + 2 * 20 + 3 * 5 + 4 * 1;
        REQUIRE(expected_idx == 119);
    }
    
    SECTION("5D variadic") {
        cnda::ContiguousND<double> a({2, 2, 2, 2, 2});
        // strides: [16, 8, 4, 2, 1], total size: 32
        REQUIRE(a.size() == 32);
        
        // Test read/write with floating point values
        a(0, 0, 0, 0, 0) = 1.5;
        a(1, 1, 1, 1, 1) = 2.5;
        
        REQUIRE(a(0, 0, 0, 0, 0) == 1.5);
        REQUIRE(a(1, 1, 1, 1, 1) == 2.5);
        
        // Verify: (1,1,1,1,1) = 16+8+4+2+1 = 31
        std::size_t idx = 1*16 + 1*8 + 1*4 + 1*2 + 1*1;
        REQUIRE(idx == 31);
    }
}

TEST_CASE("operator() const correctness", "[op][const]") {
    // Test const access for multi-dimensional arrays
    SECTION("4D const access") {
        cnda::ContiguousND<int> a({2, 3, 4, 5});
        a(1, 2, 3, 4) = 777;
        
        const auto& ca = a;
        REQUIRE(ca(1, 2, 3, 4) == 777);
    }
}

TEST_CASE("operator() edge cases", "[op][edge]") {
    SECTION("Single element array") {
        cnda::ContiguousND<int> a({1});
        a(0) = 55;
        REQUIRE(a(0) == 55);
    }
    
    SECTION("All dimensions are 1") {
        // 4D array where each dimension is 1
        cnda::ContiguousND<int> a({1, 1, 1, 1});
        a(0, 0, 0, 0) = 88;
        REQUIRE(a(0, 0, 0, 0) == 88);
        REQUIRE(a.size() == 1);
    }
}

#ifdef CNDA_BOUNDS_CHECK
TEST_CASE("bounds checks throw on invalid indices", "[bounds]") {
    cnda::ContiguousND<int> a2({3, 4});
    REQUIRE_THROWS_AS(a2(3, 0), std::out_of_range); // i out of range
    REQUIRE_THROWS_AS(a2(0, 4), std::out_of_range); // j out of range
    REQUIRE_THROWS_AS(a2.index({3, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(a2.index({0, 0, 0}), std::out_of_range); // rank mismatch

    cnda::ContiguousND<int> a1({0});
    REQUIRE_THROWS_AS(a1(0), std::out_of_range); // zero-sized dimension
}

TEST_CASE("bounds checks for N-dimensional operator()", "[bounds][ndim]") {
    SECTION("Out of bounds") {
        // Test various dimensional out of bounds access
        cnda::ContiguousND<int> a1({5});
        REQUIRE_THROWS_AS(a1(5), std::out_of_range);
        
        cnda::ContiguousND<int> a4({2, 3, 4, 5});
        REQUIRE_THROWS_AS(a4(2, 0, 0, 0), std::out_of_range);
        REQUIRE_THROWS_AS(a4(0, 3, 0, 0), std::out_of_range);
        REQUIRE_THROWS_AS(a4(0, 0, 4, 0), std::out_of_range);
        REQUIRE_THROWS_AS(a4(0, 0, 0, 5), std::out_of_range);
    }
    
    SECTION("Rank mismatch") {
        // Test calling with wrong number of indices
        cnda::ContiguousND<int> a2({3, 4});
        REQUIRE_THROWS_AS(a2(0), std::out_of_range);       // 1D call on 2D array
        REQUIRE_THROWS_AS(a2(0, 0, 0), std::out_of_range); // 3D call on 2D array
        
        cnda::ContiguousND<int> a3({2, 3, 4});
        REQUIRE_THROWS_AS(a3(0, 0), std::out_of_range);    // 2D call on 3D array
        REQUIRE_THROWS_AS(a3(0, 0, 0, 0), std::out_of_range); // 4D call on 3D array
    }
}
#endif
