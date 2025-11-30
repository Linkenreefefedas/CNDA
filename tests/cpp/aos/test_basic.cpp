/**
 * @file test_aos.cpp
 * @brief Test suite for Array-of-Structures (AoS) support in CNDA
 * 
 * This test suite validates:
 * 1. Construction of ContiguousND with struct types
 * 2. Element access and modification for struct members
 * 3. Memory layout and contiguity
 * 4. POD compliance of struct types
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cnda/contiguous_nd.hpp>
#include <cnda/aos_types.hpp>
#include <cstring>

using namespace cnda;
using namespace cnda::aos;

TEST_CASE("Vec2f basic operations", "[aos][vec2f]") {
    SECTION("Construction and initialization") {
        ContiguousND<Vec2f> arr({3, 4});
        REQUIRE(arr.ndim() == 2);
        REQUIRE(arr.size() == 12);
        REQUIRE(arr.shape() == std::vector<std::size_t>{3, 4});
    }
    
    SECTION("Element access and modification") {
        ContiguousND<Vec2f> arr({2, 3});
        arr(0, 0) = {1.0f, 2.0f};
        arr(0, 1) = {3.0f, 4.0f};
        arr(1, 2) = {5.0f, 6.0f};
        
        REQUIRE(arr(0, 0).x == 1.0f);
        REQUIRE(arr(0, 0).y == 2.0f);
        REQUIRE(arr(0, 1).x == 3.0f);
        REQUIRE(arr(0, 1).y == 4.0f);
        REQUIRE(arr(1, 2).x == 5.0f);
        REQUIRE(arr(1, 2).y == 6.0f);
    }
    
    SECTION("Field access via reference") {
        ContiguousND<Vec2f> arr({2, 2});
        arr(0, 0) = {10.0f, 20.0f};
        
        Vec2f& elem = arr(0, 0);
        REQUIRE(elem.x == 10.0f);
        REQUIRE(elem.y == 20.0f);
        
        elem.x = 30.0f;
        REQUIRE(arr(0, 0).x == 30.0f);
    }
    
    SECTION("Memory layout is contiguous") {
        ContiguousND<Vec2f> arr({2, 3});
        
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                float val = static_cast<float>(i * 10 + j);
                arr(i, j) = {val, val + 100.0f};
            }
        }
        
        Vec2f* ptr = arr.data();
        REQUIRE(ptr[0].x == 0.0f);
        REQUIRE(ptr[0].y == 100.0f);
        REQUIRE(ptr[1].x == 1.0f);
        REQUIRE(ptr[5].x == 12.0f);
    }
}

TEST_CASE("Vec3f basic operations", "[aos][vec3f]") {
    ContiguousND<Vec3f> arr({10, 10, 10});
    REQUIRE(arr.ndim() == 3);
    REQUIRE(arr.size() == 1000);
    
    arr(1, 1, 1) = {1.5f, 2.5f, 3.5f};
    REQUIRE(arr(1, 1, 1).x == 1.5f);
    REQUIRE(arr(1, 1, 1).y == 2.5f);
    REQUIRE(arr(1, 1, 1).z == 3.5f);
}

TEST_CASE("Cell2D fluid simulation example", "[aos][cell2d]") {
    ContiguousND<Cell2D> grid({5, 5});
    REQUIRE(grid.size() == 25);
    
    // Initialize boundaries and fluid cells
    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            bool is_boundary = (i == 0 || i == 4 || j == 0 || j == 4);
            grid(i, j) = {is_boundary ? 0.0f : 1.0f,
                         is_boundary ? 0.0f : 0.5f,
                         is_boundary ? -1 : 1};
        }
    }
    
    REQUIRE(grid(0, 0).flag == -1);
    REQUIRE(grid(2, 2).u == 1.0f);
    REQUIRE(grid(2, 2).v == 0.5f);
    REQUIRE(grid(2, 2).flag == 1);
}

TEST_CASE("Cell3D 3D simulation", "[aos][cell3d]") {
    ContiguousND<Cell3D> grid({10, 10, 10});
    REQUIRE(grid.size() == 1000);
    
    grid(1, 1, 1) = {1.0f, 2.0f, 3.0f, 1};
    REQUIRE(grid(1, 1, 1).u == 1.0f);
    REQUIRE(grid(1, 1, 1).v == 2.0f);
    REQUIRE(grid(1, 1, 1).w == 3.0f);
    REQUIRE(grid(1, 1, 1).flag == 1);
}

TEST_CASE("Particle system", "[aos][particle]") {
    ContiguousND<Particle> particles({5});
    REQUIRE(particles.size() == 5);
    
    double dt = 0.1;
    for (std::size_t i = 0; i < 5; ++i) {
        particles(i) = {0.0, 0.0, 0.0, static_cast<double>(i), 0.0, 0.0, 1.0};
        particles(i).x += particles(i).vx * dt;
    }
    
    REQUIRE(particles(0).x == 0.0);
    REQUIRE(particles(1).x == Catch::Approx(0.1));
    REQUIRE(particles(4).x == Catch::Approx(0.4));
    REQUIRE(particles(1).vx == 1.0);
    REQUIRE(particles(0).mass == 1.0);
}

TEST_CASE("MaterialPoint grid", "[aos][materialpoint]") {
    ContiguousND<MaterialPoint> grid({3, 3});
    REQUIRE(grid.size() == 9);
    
    // Set different materials: air, water, steel
    grid(0, 0) = {1.0f, 300.0f, 101.3f, 1};
    grid(1, 1) = {1000.0f, 293.0f, 101.3f, 2};
    grid(2, 2) = {7850.0f, 293.0f, 101.3f, 3};
    
    REQUIRE(grid(0, 0).density == Catch::Approx(1.0f));
    REQUIRE(grid(1, 1).density == Catch::Approx(1000.0f));
    REQUIRE(grid(2, 2).density == Catch::Approx(7850.0f));
    REQUIRE(grid(0, 0).id == 1);
    REQUIRE(grid(1, 1).id == 2);
    REQUIRE(grid(2, 2).id == 3);
}

TEST_CASE("Memory size validation", "[aos][memory]") {
    REQUIRE(sizeof(Vec2f) == 2 * sizeof(float));
    REQUIRE(sizeof(Vec3f) == 3 * sizeof(float));
    REQUIRE(sizeof(Cell2D) >= 2 * sizeof(float) + sizeof(std::int32_t));
    REQUIRE(sizeof(Cell3D) >= 3 * sizeof(float) + sizeof(std::int32_t));
    REQUIRE(sizeof(Particle) == 7 * sizeof(double));
    REQUIRE(sizeof(MaterialPoint) >= 3 * sizeof(float) + sizeof(std::int32_t));
    
    // Verify contiguous memory layout
    ContiguousND<Vec2f> arr({100});
    auto byte_dist = reinterpret_cast<char*>(&arr.data()[99]) - reinterpret_cast<char*>(&arr.data()[0]);
    REQUIRE(byte_dist == 99 * sizeof(Vec2f));
}

TEST_CASE("POD type guarantees", "[aos][pod]") {
    ContiguousND<Vec2f> src({2, 2}), dst({2, 2});
    src(0, 0) = {1.0f, 2.0f};
    src(1, 1) = {3.0f, 4.0f};
    
    std::memcpy(dst.data(), src.data(), src.size() * sizeof(Vec2f));
    
    REQUIRE(dst(0, 0).x == 1.0f);
    REQUIRE(dst(0, 0).y == 2.0f);
    REQUIRE(dst(1, 1).x == 3.0f);
    REQUIRE(dst(1, 1).y == 4.0f);
}

TEST_CASE("at() method with bounds checking", "[aos][bounds]") {
    ContiguousND<Vec2f> arr({2, 3});
    arr.at({0, 0}) = {1.0f, 2.0f};
    REQUIRE(arr.at({0, 0}).x == 1.0f);
    REQUIRE(arr.at({0, 0}).y == 2.0f);
    
    ContiguousND<Cell2D> grid({5, 5});
    grid.at({2, 3}) = {1.5f, 2.5f, 10};
    REQUIRE(grid.at({2, 3}).u == 1.5f);
    REQUIRE(grid.at({2, 3}).v == 2.5f);
    REQUIRE(grid.at({2, 3}).flag == 10);
    
    REQUIRE_THROWS_AS(arr.at({2, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(arr.at({0, 3}), std::out_of_range);
    REQUIRE_THROWS_AS(arr.at({5, 5}), std::out_of_range);
}
