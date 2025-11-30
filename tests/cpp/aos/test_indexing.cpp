#include <catch2/catch_test_macros.hpp>
#include "cnda/contiguous_nd.hpp"
#include "cnda/aos_types.hpp"
#include <cstddef>

using namespace cnda;

template<typename T>
std::size_t byte_offset(const T* ptr, const T* base) {
    return reinterpret_cast<const char*>(ptr) - reinterpret_cast<const char*>(base);
}

TEST_CASE("AoS indexing uses sizeof(T) correctly", "[aos][indexing]") {
    SECTION("Vec2f stride computation") {
        ContiguousND<aos::Vec2f> arr({3, 4});
        auto strides = arr.strides();
        
        REQUIRE(strides.size() == 2);
        REQUIRE(strides[0] == 4);
        REQUIRE(strides[1] == 1);
        REQUIRE(arr.index({1, 2}) == 6);
        
        aos::Vec2f* base = arr.data();
        REQUIRE(&arr(1, 2) == base + 6);
        REQUIRE(byte_offset(&arr(1, 2), base) == 6 * sizeof(aos::Vec2f));
    }
    
    SECTION("Cell3D pointer arithmetic") {
        ContiguousND<aos::Cell3D> arr({2, 3});
        auto strides = arr.strides();
        
        REQUIRE(strides[0] == 3);
        REQUIRE(strides[1] == 1);
        
        auto& cell = arr(1, 2);
        cell = {1.0f, 2.0f, 3.0f, 4};
        
        aos::Cell3D* base = arr.data();
        REQUIRE(&cell - base == 5);
        REQUIRE(byte_offset(&cell, base) == 5 * sizeof(aos::Cell3D));
        REQUIRE(cell.u == 1.0f);
        REQUIRE(cell.v == 2.0f);
        REQUIRE(cell.w == 3.0f);
        REQUIRE(cell.flag == 4);
    }
    
    SECTION("Particle array memory continuity") {
        ContiguousND<aos::Particle> particles({10});
        REQUIRE(particles.strides()[0] == 1);
        
        for (std::size_t i = 0; i < 10; ++i) {
            auto& p = particles(i);
            p = {static_cast<double>(i), static_cast<double>(i * 2), static_cast<double>(i * 3),
                 static_cast<double>(i * 4), static_cast<double>(i * 5), static_cast<double>(i * 6), 1.0};
        }
        
        aos::Particle* base = particles.data();
        for (std::size_t i = 0; i < 10; ++i) {
            REQUIRE(&particles(i) - base == static_cast<std::ptrdiff_t>(i));
            REQUIRE(byte_offset(&particles(i), base) == i * sizeof(aos::Particle));
            REQUIRE(particles(i).x == static_cast<double>(i));
            REQUIRE(particles(i).y == static_cast<double>(i * 2));
        }
    }
    
    SECTION("3D array element spacing") {
        ContiguousND<aos::Vec3f> arr({2, 3, 4});
        auto strides = arr.strides();
        
        REQUIRE(strides[0] == 12);
        REQUIRE(strides[1] == 4);
        REQUIRE(strides[2] == 1);
        
        aos::Vec3f* base = arr.data();
        REQUIRE(&arr(0, 0, 0) - base == 0);
        REQUIRE(&arr(0, 0, 1) - base == 1);
        REQUIRE(&arr(0, 1, 0) - base == 4);
        REQUIRE(&arr(1, 0, 0) - base == 12);
        REQUIRE(&arr(1, 2, 3) - base == 23);
        REQUIRE(byte_offset(&arr(1, 2, 3), base) == 23 * sizeof(aos::Vec3f));
    }
}

TEST_CASE("AoS vs scalar type sizeof comparison", "[aos][indexing]") {
    SECTION("float vs Vec2f indexing") {
        ContiguousND<float> scalar_arr({3, 4});
        ContiguousND<aos::Vec2f> aos_arr({3, 4});
        
        REQUIRE(scalar_arr.strides() == aos_arr.strides());
        REQUIRE(scalar_arr.size() == aos_arr.size());
        REQUIRE(scalar_arr.size() * sizeof(float) == 48);
        REQUIRE(aos_arr.size() * sizeof(aos::Vec2f) == 96);
    }
    
    SECTION("Pointer arithmetic scales by sizeof(T)") {
        ContiguousND<aos::Cell3D> arr({5, 5});
        
        aos::Cell3D* p0 = &arr(0, 0);
        aos::Cell3D* p1 = &arr(0, 1);
        aos::Cell3D* p2 = &arr(1, 0);
        
        REQUIRE(p1 - p0 == 1);
        REQUIRE(p2 - p0 == 5);
        REQUIRE(byte_offset(p1, p0) == sizeof(aos::Cell3D));
        REQUIRE(byte_offset(p2, p0) == 5 * sizeof(aos::Cell3D));
    }
}
