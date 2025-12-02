#include <catch2/catch_test_macros.hpp>
#include "cnda/contiguous_nd.hpp"
#include "cnda/aos_types.hpp"

using namespace cnda;

TEST_CASE("Variadic operator() with scalar types", "[indexing][variadic]") {
    SECTION("1D access") {
        ContiguousND<int> arr({10});
        
        // Set values using variadic operator()
        for (int i = 0; i < 10; ++i) {
            arr(i) = i * 10;
        }
        
        // Verify values
        for (int i = 0; i < 10; ++i) {
            REQUIRE(arr(i) == i * 10);
        }
    }
    
    SECTION("2D access") {
        ContiguousND<double> arr({3, 4});
        
        // Set values using variadic operator()
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                arr(i, j) = static_cast<double>(i * 10 + j);
            }
        }
        
        // Verify values
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                REQUIRE(arr(i, j) == static_cast<double>(i * 10 + j));
            }
        }
    }
    
    SECTION("3D access") {
        ContiguousND<float> arr({2, 3, 4});
        
        // Set values using variadic operator()
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t k = 0; k < 4; ++k) {
                    arr(i, j, k) = static_cast<float>(i * 100 + j * 10 + k);
                }
            }
        }
        
        // Verify values
        REQUIRE(arr(0, 0, 0) == 0.0f);
        REQUIRE(arr(0, 0, 1) == 1.0f);
        REQUIRE(arr(0, 1, 0) == 10.0f);
        REQUIRE(arr(1, 0, 0) == 100.0f);
        REQUIRE(arr(1, 2, 3) == 123.0f);
    }
    
    SECTION("4D access") {
        ContiguousND<int> arr({2, 3, 4, 5});
        
        // Set specific values
        arr(0, 0, 0, 0) = 1000;
        arr(0, 0, 0, 4) = 1004;
        arr(0, 0, 3, 0) = 1030;
        arr(0, 2, 0, 0) = 1200;
        arr(1, 0, 0, 0) = 2000;
        arr(1, 2, 3, 4) = 2234;
        
        // Verify values
        REQUIRE(arr(0, 0, 0, 0) == 1000);
        REQUIRE(arr(0, 0, 0, 4) == 1004);
        REQUIRE(arr(0, 0, 3, 0) == 1030);
        REQUIRE(arr(0, 2, 0, 0) == 1200);
        REQUIRE(arr(1, 0, 0, 0) == 2000);
        REQUIRE(arr(1, 2, 3, 4) == 2234);
    }
    
    SECTION("5D access") {
        ContiguousND<int> arr({2, 2, 2, 2, 2});
        
        // Set corner values
        arr(0, 0, 0, 0, 0) = 10000;
        arr(0, 0, 0, 0, 1) = 10001;
        arr(1, 1, 1, 1, 1) = 11111;
        
        // Verify
        REQUIRE(arr(0, 0, 0, 0, 0) == 10000);
        REQUIRE(arr(0, 0, 0, 0, 1) == 10001);
        REQUIRE(arr(1, 1, 1, 1, 1) == 11111);
    }
}

TEST_CASE("Variadic operator() with AoS types", "[indexing][variadic][aos]") {
    SECTION("Vec2f with 2D indexing") {
        ContiguousND<aos::Vec2f> arr({3, 4});
        
        // Set values using variadic operator()
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                arr(i, j).x = static_cast<float>(i);
                arr(i, j).y = static_cast<float>(j);
            }
        }
        
        // Verify values
        REQUIRE(arr(0, 0).x == 0.0f);
        REQUIRE(arr(0, 0).y == 0.0f);
        REQUIRE(arr(1, 2).x == 1.0f);
        REQUIRE(arr(1, 2).y == 2.0f);
        REQUIRE(arr(2, 3).x == 2.0f);
        REQUIRE(arr(2, 3).y == 3.0f);
    }
    
    SECTION("Vec3f with 3D indexing") {
        ContiguousND<aos::Vec3f> arr({2, 3, 4});
        
        // Set values
        arr(0, 0, 0).x = 1.0f;
        arr(0, 0, 0).y = 2.0f;
        arr(0, 0, 0).z = 3.0f;
        
        arr(1, 2, 3).x = 10.0f;
        arr(1, 2, 3).y = 20.0f;
        arr(1, 2, 3).z = 30.0f;
        
        // Verify
        REQUIRE(arr(0, 0, 0).x == 1.0f);
        REQUIRE(arr(0, 0, 0).y == 2.0f);
        REQUIRE(arr(0, 0, 0).z == 3.0f);
        REQUIRE(arr(1, 2, 3).x == 10.0f);
        REQUIRE(arr(1, 2, 3).y == 20.0f);
        REQUIRE(arr(1, 2, 3).z == 30.0f);
    }
    
    SECTION("Particle with 1D indexing") {
        ContiguousND<aos::Particle> particles({100});
        
        // Initialize particles using variadic operator()
        for (std::size_t i = 0; i < 100; ++i) {
            particles(i).x = static_cast<double>(i);
            particles(i).y = static_cast<double>(i * 2);
            particles(i).z = static_cast<double>(i * 3);
            particles(i).vx = static_cast<double>(i * 4);
            particles(i).vy = static_cast<double>(i * 5);
            particles(i).vz = static_cast<double>(i * 6);
            particles(i).mass = static_cast<double>(i + 1);
        }
        
        // Verify specific particles
        REQUIRE(particles(0).x == 0.0);
        REQUIRE(particles(0).mass == 1.0);
        REQUIRE(particles(50).x == 50.0);
        REQUIRE(particles(50).y == 100.0);
        REQUIRE(particles(50).vx == 200.0);
        REQUIRE(particles(99).mass == 100.0);
    }
    
    SECTION("Cell3D with 4D indexing") {
        ContiguousND<aos::Cell3D> grid({2, 3, 4, 5});
        
        // Set velocity field at specific locations
        grid(0, 0, 0, 0).u = 1.0f;
        grid(0, 0, 0, 0).v = 2.0f;
        grid(0, 0, 0, 0).w = 3.0f;
        grid(0, 0, 0, 0).flag = 1;
        
        grid(1, 2, 3, 4).u = 10.0f;
        grid(1, 2, 3, 4).v = 20.0f;
        grid(1, 2, 3, 4).w = 30.0f;
        grid(1, 2, 3, 4).flag = 5;
        
        // Verify
        REQUIRE(grid(0, 0, 0, 0).u == 1.0f);
        REQUIRE(grid(0, 0, 0, 0).v == 2.0f);
        REQUIRE(grid(0, 0, 0, 0).w == 3.0f);
        REQUIRE(grid(0, 0, 0, 0).flag == 1);
        REQUIRE(grid(1, 2, 3, 4).u == 10.0f);
        REQUIRE(grid(1, 2, 3, 4).v == 20.0f);
        REQUIRE(grid(1, 2, 3, 4).w == 30.0f);
        REQUIRE(grid(1, 2, 3, 4).flag == 5);
    }
}

TEST_CASE("initializer_list index() method", "[indexing][initializer_list]") {
    SECTION("Scalar type - 2D") {
        ContiguousND<int> arr({3, 4});
        
        // Test index() returns correct element offset
        REQUIRE(arr.index({0, 0}) == 0);
        REQUIRE(arr.index({0, 1}) == 1);
        REQUIRE(arr.index({0, 3}) == 3);
        REQUIRE(arr.index({1, 0}) == 4);
        REQUIRE(arr.index({1, 1}) == 5);
        REQUIRE(arr.index({2, 3}) == 11);
    }
    
    SECTION("Scalar type - 3D") {
        ContiguousND<double> arr({2, 3, 4});
        
        // Strides: [12, 4, 1]
        REQUIRE(arr.index({0, 0, 0}) == 0);
        REQUIRE(arr.index({0, 0, 1}) == 1);
        REQUIRE(arr.index({0, 1, 0}) == 4);
        REQUIRE(arr.index({1, 0, 0}) == 12);
        REQUIRE(arr.index({1, 2, 3}) == 1*12 + 2*4 + 3*1);
        REQUIRE(arr.index({1, 2, 3}) == 23);
    }
    
    SECTION("AoS type - Vec2f 2D") {
        ContiguousND<aos::Vec2f> arr({3, 4});
        
        // Same index calculations as scalar types (element-based)
        REQUIRE(arr.index({0, 0}) == 0);
        REQUIRE(arr.index({1, 2}) == 6);
        REQUIRE(arr.index({2, 3}) == 11);
    }
    
    SECTION("AoS type - Particle 1D") {
        ContiguousND<aos::Particle> particles({100});
        
        for (std::size_t i = 0; i < 100; ++i) {
            REQUIRE(particles.index({i}) == i);
        }
    }
    
    SECTION("Consistency between index() and operator()") {
        ContiguousND<aos::Vec3f> arr({2, 3, 4});
        
        // For every valid index, index() and operator() should be consistent
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t k = 0; k < 4; ++k) {
                    std::size_t offset = arr.index({i, j, k});
                    aos::Vec3f* ptr_from_index = arr.data() + offset;
                    aos::Vec3f* ptr_from_operator = &arr(i, j, k);
                    
                    REQUIRE(ptr_from_index == ptr_from_operator);
                }
            }
        }
    }
}

TEST_CASE("Bounds checking with CNDA_BOUNDS_CHECK", "[indexing][bounds]") {
    #ifdef CNDA_BOUNDS_CHECK
    
    SECTION("Variadic operator() - rank mismatch") {
        ContiguousND<int> arr({3, 4});
        
        // Wrong number of indices should throw
        REQUIRE_THROWS_AS(arr(0), std::out_of_range);
        REQUIRE_THROWS_AS(arr(0, 1, 2), std::out_of_range);
    }
    
    SECTION("Variadic operator() - out of bounds") {
        ContiguousND<float> arr({3, 4, 5});
        
        // Valid access
        REQUIRE_NOTHROW(arr(0, 0, 0));
        REQUIRE_NOTHROW(arr(2, 3, 4));
        
        // Out of bounds
        REQUIRE_THROWS_AS(arr(3, 0, 0), std::out_of_range);
        REQUIRE_THROWS_AS(arr(0, 4, 0), std::out_of_range);
        REQUIRE_THROWS_AS(arr(0, 0, 5), std::out_of_range);
    }
    
    SECTION("index() - rank mismatch") {
        ContiguousND<double> arr({2, 3});
        
        REQUIRE_THROWS_AS(arr.index({0}), std::out_of_range);
        REQUIRE_THROWS_AS(arr.index({0, 1, 2}), std::out_of_range);
    }
    
    SECTION("index() - out of bounds") {
        ContiguousND<int> arr({2, 3, 4});
        
        // Valid access
        REQUIRE_NOTHROW(arr.index({0, 0, 0}));
        REQUIRE_NOTHROW(arr.index({1, 2, 3}));
        
        // Out of bounds
        REQUIRE_THROWS_AS(arr.index({2, 0, 0}), std::out_of_range);
        REQUIRE_THROWS_AS(arr.index({0, 3, 0}), std::out_of_range);
        REQUIRE_THROWS_AS(arr.index({0, 0, 4}), std::out_of_range);
    }
    
    SECTION("AoS types - bounds checking") {
        ContiguousND<aos::Vec2f> arr({5, 5});
        
        // Valid
        REQUIRE_NOTHROW(arr(0, 0));
        REQUIRE_NOTHROW(arr(4, 4));
        
        // Invalid
        REQUIRE_THROWS_AS(arr(5, 0), std::out_of_range);
        REQUIRE_THROWS_AS(arr(0, 5), std::out_of_range);
        REQUIRE_THROWS_AS(arr(10, 10), std::out_of_range);
    }
    
    #else
    
    SECTION("Bounds checking disabled - no exceptions") {
        ContiguousND<int> arr({3, 4});
        
        // These would throw with CNDA_BOUNDS_CHECK, but compile without it
        // We can't test the actual behavior without bounds checking
        // Just verify compilation works
        REQUIRE(arr.ndim() == 2);
    }
    
    #endif
}

TEST_CASE("Const correctness for variadic operator()", "[indexing][const]") {
    SECTION("Const access with scalar type") {
        ContiguousND<int> arr({3, 4});
        arr(1, 2) = 42;
        
        const auto& const_arr = arr;
        REQUIRE(const_arr(1, 2) == 42);
    }
    
    SECTION("Const access with AoS type") {
        ContiguousND<aos::Vec3f> arr({2, 3});
        arr(1, 2).x = 1.0f;
        arr(1, 2).y = 2.0f;
        arr(1, 2).z = 3.0f;
        
        const auto& const_arr = arr;
        REQUIRE(const_arr(1, 2).x == 1.0f);
        REQUIRE(const_arr(1, 2).y == 2.0f);
        REQUIRE(const_arr(1, 2).z == 3.0f);
    }
}

TEST_CASE("Mixed integer types in variadic operator()", "[indexing][variadic]") {
    SECTION("Different integer types") {
        ContiguousND<float> arr({10, 10});
        
        // Should work with different integer types
        int i = 5;
        std::size_t j = 7;
        unsigned int k = 3;
        long m = 2;
        
        arr(i, j) = 1.0f;
        arr(k, k) = 2.0f;
        arr(m, m) = 3.0f;
        
        REQUIRE(arr(5, 7) == 1.0f);
        REQUIRE(arr(3, 3) == 2.0f);
        REQUIRE(arr(2, 2) == 3.0f);
    }
}
