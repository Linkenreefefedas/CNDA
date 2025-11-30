/**
 * @file test_aos_field_layout.cpp
 * @brief Comprehensive tests for AoS field layout, offsetof, and API consistency
 * 
 * This test suite validates:
 * 1. Field memory offsets (offsetof) match expected layout
 * 2. No padding between fields when not needed
 * 3. Field access APIs (get/set) correctly map to memory locations
 * 4. No field mix-up across different AoS types
 * 5. sizeof() matches expected field sum
 */

#include <catch2/catch_test_macros.hpp>
#include "cnda/contiguous_nd.hpp"
#include "cnda/aos_types.hpp"
#include <cstddef>
#include <cstdint>
#include <type_traits>

using namespace cnda;

// ==============================================================================
// Field Offset Tests (offsetof validation)
// ==============================================================================

TEST_CASE("Field memory offsets for Vec2f", "[aos][layout][offsetof]") {
    // Vec2f: struct { float x; float y; }
    // Expected layout: x at offset 0, y at offset 4
    
    REQUIRE(offsetof(aos::Vec2f, x) == 0);
    REQUIRE(offsetof(aos::Vec2f, y) == sizeof(float));
    
    // Total size should be 2 * sizeof(float) = 8 bytes
    REQUIRE(sizeof(aos::Vec2f) == 2 * sizeof(float));
    
    // No padding between fields
    REQUIRE(offsetof(aos::Vec2f, y) - offsetof(aos::Vec2f, x) == sizeof(float));
}

TEST_CASE("Field memory offsets for Vec3f", "[aos][layout][offsetof]") {
    // Vec3f: struct { float x; float y; float z; }
    // Expected layout: x at 0, y at 4, z at 8
    
    REQUIRE(offsetof(aos::Vec3f, x) == 0);
    REQUIRE(offsetof(aos::Vec3f, y) == sizeof(float));
    REQUIRE(offsetof(aos::Vec3f, z) == 2 * sizeof(float));
    
    // Total size should be 3 * sizeof(float) = 12 bytes
    REQUIRE(sizeof(aos::Vec3f) == 3 * sizeof(float));
    
    // No padding between fields
    REQUIRE(offsetof(aos::Vec3f, y) - offsetof(aos::Vec3f, x) == sizeof(float));
    REQUIRE(offsetof(aos::Vec3f, z) - offsetof(aos::Vec3f, y) == sizeof(float));
}

TEST_CASE("Field memory offsets for Cell2D", "[aos][layout][offsetof]") {
    // Cell2D: struct { float u; float v; int32_t flag; }
    // Expected layout: u at 0, v at 4, flag at 8
    
    REQUIRE(offsetof(aos::Cell2D, u) == 0);
    REQUIRE(offsetof(aos::Cell2D, v) == sizeof(float));
    REQUIRE(offsetof(aos::Cell2D, flag) == 2 * sizeof(float));
    
    // Total size should be 2 * sizeof(float) + sizeof(int32_t) = 12 bytes
    REQUIRE(sizeof(aos::Cell2D) == 2 * sizeof(float) + sizeof(std::int32_t));
    
    // No padding between fields
    REQUIRE(offsetof(aos::Cell2D, v) - offsetof(aos::Cell2D, u) == sizeof(float));
    REQUIRE(offsetof(aos::Cell2D, flag) - offsetof(aos::Cell2D, v) == sizeof(float));
}

TEST_CASE("Field memory offsets for Cell3D", "[aos][layout][offsetof]") {
    // Cell3D: struct { float u; float v; float w; int32_t flag; }
    // Expected layout: u at 0, v at 4, w at 8, flag at 12
    
    REQUIRE(offsetof(aos::Cell3D, u) == 0);
    REQUIRE(offsetof(aos::Cell3D, v) == sizeof(float));
    REQUIRE(offsetof(aos::Cell3D, w) == 2 * sizeof(float));
    REQUIRE(offsetof(aos::Cell3D, flag) == 3 * sizeof(float));
    
    // Total size should be 3 * sizeof(float) + sizeof(int32_t) = 16 bytes
    REQUIRE(sizeof(aos::Cell3D) == 3 * sizeof(float) + sizeof(std::int32_t));
    
    // No padding between fields
    REQUIRE(offsetof(aos::Cell3D, v) - offsetof(aos::Cell3D, u) == sizeof(float));
    REQUIRE(offsetof(aos::Cell3D, w) - offsetof(aos::Cell3D, v) == sizeof(float));
    REQUIRE(offsetof(aos::Cell3D, flag) - offsetof(aos::Cell3D, w) == sizeof(float));
}

TEST_CASE("Field memory offsets for Particle", "[aos][layout][offsetof]") {
    // Particle: struct { double x, y, z, vx, vy, vz, mass; }
    // All doubles, sequential layout
    
    REQUIRE(offsetof(aos::Particle, x) == 0);
    REQUIRE(offsetof(aos::Particle, y) == sizeof(double));
    REQUIRE(offsetof(aos::Particle, z) == 2 * sizeof(double));
    REQUIRE(offsetof(aos::Particle, vx) == 3 * sizeof(double));
    REQUIRE(offsetof(aos::Particle, vy) == 4 * sizeof(double));
    REQUIRE(offsetof(aos::Particle, vz) == 5 * sizeof(double));
    REQUIRE(offsetof(aos::Particle, mass) == 6 * sizeof(double));
    
    // Total size should be 7 * sizeof(double) = 56 bytes
    REQUIRE(sizeof(aos::Particle) == 7 * sizeof(double));
    
    // Verify all fields are contiguous
    REQUIRE(offsetof(aos::Particle, y) - offsetof(aos::Particle, x) == sizeof(double));
    REQUIRE(offsetof(aos::Particle, z) - offsetof(aos::Particle, y) == sizeof(double));
    REQUIRE(offsetof(aos::Particle, vx) - offsetof(aos::Particle, z) == sizeof(double));
    REQUIRE(offsetof(aos::Particle, vy) - offsetof(aos::Particle, vx) == sizeof(double));
    REQUIRE(offsetof(aos::Particle, vz) - offsetof(aos::Particle, vy) == sizeof(double));
    REQUIRE(offsetof(aos::Particle, mass) - offsetof(aos::Particle, vz) == sizeof(double));
}

TEST_CASE("Field memory offsets for MaterialPoint", "[aos][layout][offsetof]") {
    // MaterialPoint: struct { float density; float temperature; float pressure; int32_t id; }
    
    REQUIRE(offsetof(aos::MaterialPoint, density) == 0);
    REQUIRE(offsetof(aos::MaterialPoint, temperature) == sizeof(float));
    REQUIRE(offsetof(aos::MaterialPoint, pressure) == 2 * sizeof(float));
    REQUIRE(offsetof(aos::MaterialPoint, id) == 3 * sizeof(float));
    
    // Total size should be 3 * sizeof(float) + sizeof(int32_t) = 16 bytes
    REQUIRE(sizeof(aos::MaterialPoint) == 3 * sizeof(float) + sizeof(std::int32_t));
}

// ==============================================================================
// Field Access API Consistency Tests
// ==============================================================================

TEST_CASE("Vec2f field access matches memory layout", "[aos][field-api]") {
    ContiguousND<aos::Vec2f> arr({1});
    
    // Set via direct struct access
    arr(0).x = 1.5f;
    arr(0).y = 2.5f;
    
    // Verify via pointer arithmetic
    float* base = reinterpret_cast<float*>(&arr(0));
    REQUIRE(base[0] == 1.5f);  // x at offset 0
    REQUIRE(base[1] == 2.5f);  // y at offset 1 (sizeof(float))
    
    // Verify offsetof matches actual memory
    float* x_ptr = &arr(0).x;
    float* y_ptr = &arr(0).y;
    REQUIRE(reinterpret_cast<char*>(y_ptr) - reinterpret_cast<char*>(x_ptr) == sizeof(float));
}

TEST_CASE("Vec3f field access matches memory layout", "[aos][field-api]") {
    ContiguousND<aos::Vec3f> arr({1});
    
    arr(0).x = 10.0f;
    arr(0).y = 20.0f;
    arr(0).z = 30.0f;
    
    float* base = reinterpret_cast<float*>(&arr(0));
    REQUIRE(base[0] == 10.0f);  // x
    REQUIRE(base[1] == 20.0f);  // y
    REQUIRE(base[2] == 30.0f);  // z
    
    // Verify no field mix-up
    REQUIRE(arr(0).x == 10.0f);
    REQUIRE(arr(0).y == 20.0f);
    REQUIRE(arr(0).z == 30.0f);
}

TEST_CASE("Cell2D field access matches memory layout", "[aos][field-api]") {
    ContiguousND<aos::Cell2D> arr({1});
    
    arr(0).u = 5.5f;
    arr(0).v = 6.5f;
    arr(0).flag = 42;
    
    // Verify float fields
    float* float_base = reinterpret_cast<float*>(&arr(0));
    REQUIRE(float_base[0] == 5.5f);  // u
    REQUIRE(float_base[1] == 6.5f);  // v
    
    // Verify int field
    std::int32_t* int_ptr = &arr(0).flag;
    REQUIRE(*int_ptr == 42);
    
    // Verify offset
    REQUIRE(reinterpret_cast<char*>(int_ptr) - reinterpret_cast<char*>(&arr(0).u) 
            == 2 * sizeof(float));
}

TEST_CASE("Cell3D field access matches memory layout", "[aos][field-api]") {
    ContiguousND<aos::Cell3D> arr({1});
    
    arr(0).u = 1.0f;
    arr(0).v = 2.0f;
    arr(0).w = 3.0f;
    arr(0).flag = 99;
    
    float* float_base = reinterpret_cast<float*>(&arr(0));
    REQUIRE(float_base[0] == 1.0f);  // u
    REQUIRE(float_base[1] == 2.0f);  // v
    REQUIRE(float_base[2] == 3.0f);  // w
    
    REQUIRE(arr(0).flag == 99);
    
    // No field mix-up
    REQUIRE(arr(0).u == 1.0f);
    REQUIRE(arr(0).v == 2.0f);
    REQUIRE(arr(0).w == 3.0f);
}

TEST_CASE("Particle field access matches memory layout", "[aos][field-api]") {
    ContiguousND<aos::Particle> arr({1});
    
    arr(0).x = 1.0;
    arr(0).y = 2.0;
    arr(0).z = 3.0;
    arr(0).vx = 4.0;
    arr(0).vy = 5.0;
    arr(0).vz = 6.0;
    arr(0).mass = 7.0;
    
    double* base = reinterpret_cast<double*>(&arr(0));
    REQUIRE(base[0] == 1.0);   // x
    REQUIRE(base[1] == 2.0);   // y
    REQUIRE(base[2] == 3.0);   // z
    REQUIRE(base[3] == 4.0);   // vx
    REQUIRE(base[4] == 5.0);   // vy
    REQUIRE(base[5] == 6.0);   // vz
    REQUIRE(base[6] == 7.0);   // mass
    
    // Verify no mix-up
    REQUIRE(arr(0).x == 1.0);
    REQUIRE(arr(0).vx == 4.0);
    REQUIRE(arr(0).mass == 7.0);
}

// ==============================================================================
// Field Mix-up Prevention Tests
// ==============================================================================

TEST_CASE("No field mix-up between different AoS types", "[aos][field-api]") {
    SECTION("Vec2f x/y are distinct") {
        ContiguousND<aos::Vec2f> arr({2});
        arr(0).x = 1.0f;
        arr(0).y = 2.0f;
        arr(1).x = 10.0f;
        arr(1).y = 20.0f;
        
        // x should not be y
        REQUIRE(arr(0).x != arr(0).y);
        REQUIRE(arr(1).x != arr(1).y);
        
        // Values should be correct
        REQUIRE(arr(0).x == 1.0f);
        REQUIRE(arr(0).y == 2.0f);
    }
    
    SECTION("Vec3f x/y/z are distinct") {
        ContiguousND<aos::Vec3f> arr({2});
        arr(0) = {1.0f, 2.0f, 3.0f};
        arr(1) = {10.0f, 20.0f, 30.0f};
        
        REQUIRE(arr(0).x != arr(0).y);
        REQUIRE(arr(0).y != arr(0).z);
        REQUIRE(arr(0).x != arr(0).z);
        
        REQUIRE(arr(0).x == 1.0f);
        REQUIRE(arr(0).y == 2.0f);
        REQUIRE(arr(0).z == 3.0f);
    }
    
    SECTION("Cell2D u/v/flag are distinct") {
        ContiguousND<aos::Cell2D> arr({2});
        arr(0) = {5.5f, 6.5f, 42};
        
        // u, v are different floats
        REQUIRE(arr(0).u != arr(0).v);
        
        // flag is int, shouldn't accidentally read as float
        float u_val = arr(0).u;
        float v_val = arr(0).v;
        std::int32_t flag_val = arr(0).flag;
        
        REQUIRE(u_val == 5.5f);
        REQUIRE(v_val == 6.5f);
        REQUIRE(flag_val == 42);
    }
    
    SECTION("Particle position vs velocity fields distinct") {
        ContiguousND<aos::Particle> arr({1});
        arr(0).x = 1.0;
        arr(0).y = 2.0;
        arr(0).z = 3.0;
        arr(0).vx = 10.0;
        arr(0).vy = 20.0;
        arr(0).vz = 30.0;
        arr(0).mass = 100.0;
        
        // Position != velocity
        REQUIRE(arr(0).x != arr(0).vx);
        REQUIRE(arr(0).y != arr(0).vy);
        REQUIRE(arr(0).z != arr(0).vz);
        
        // All fields are distinct
        REQUIRE(arr(0).x == 1.0);
        REQUIRE(arr(0).vx == 10.0);
        REQUIRE(arr(0).mass == 100.0);
    }
}

// ==============================================================================
// Contiguous Layout Tests
// ==============================================================================

TEST_CASE("Array of AoS types maintains correct stride", "[aos][layout][stride]") {
    SECTION("Vec2f array stride") {
        ContiguousND<aos::Vec2f> arr({3});
        
        aos::Vec2f* p0 = &arr(0);
        aos::Vec2f* p1 = &arr(1);
        aos::Vec2f* p2 = &arr(2);
        
        // Stride should be sizeof(Vec2f)
        REQUIRE(reinterpret_cast<char*>(p1) - reinterpret_cast<char*>(p0) 
                == sizeof(aos::Vec2f));
        REQUIRE(reinterpret_cast<char*>(p2) - reinterpret_cast<char*>(p1) 
                == sizeof(aos::Vec2f));
    }
    
    SECTION("Cell3D array stride") {
        ContiguousND<aos::Cell3D> arr({5});
        
        for (std::size_t i = 0; i < 4; ++i) {
            aos::Cell3D* pi = &arr(i);
            aos::Cell3D* pi1 = &arr(i + 1);
            
            REQUIRE(reinterpret_cast<char*>(pi1) - reinterpret_cast<char*>(pi) 
                    == sizeof(aos::Cell3D));
        }
    }
}

TEST_CASE("sizeof() validation for all AoS types", "[aos][layout][sizeof]") {
    // Verify no unexpected padding
    REQUIRE(sizeof(aos::Vec2f) == 8);   // 2 * 4
    REQUIRE(sizeof(aos::Vec3f) == 12);  // 3 * 4
    REQUIRE(sizeof(aos::Cell2D) == 12); // 2 * 4 + 4
    REQUIRE(sizeof(aos::Cell3D) == 16); // 3 * 4 + 4
    REQUIRE(sizeof(aos::Particle) == 56); // 7 * 8
    REQUIRE(sizeof(aos::MaterialPoint) == 16); // 3 * 4 + 4
}
