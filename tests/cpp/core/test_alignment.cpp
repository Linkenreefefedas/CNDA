#include <catch2/catch_test_macros.hpp>
#include "cnda/contiguous_nd.hpp"
#include "cnda/aos_types.hpp"
#include <type_traits>
#include <cstddef>
#include <cstdint>

using namespace cnda;

// Helper to check pointer alignment
template<typename T>
bool is_aligned(const T* ptr, std::size_t alignment = alignof(T)) {
    return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

// Helper to test basic alignment properties
template<typename T>
void test_basic_alignment(const ContiguousND<T>& arr) {
    REQUIRE(is_aligned(arr.data(), alignof(T)));
    REQUIRE(alignof(T) >= 1);
}

TEST_CASE("Memory alignment for scalar types", "[alignment][scalar]") {
    SECTION("float alignment") {
        ContiguousND<float> arr({100, 100});
        test_basic_alignment(arr);
        for (std::size_t i = 0; i < 10; ++i) {
            REQUIRE(is_aligned(&arr(i, 0)));
        }
    }
    
    SECTION("double alignment") {
        ContiguousND<double> arr({50, 50});
        test_basic_alignment(arr);
        for (std::size_t i = 0; i < 10; ++i) {
            REQUIRE(is_aligned(&arr(i, 0)));
        }
    }
    
    SECTION("int32_t alignment") {
        ContiguousND<std::int32_t> arr({1000});
        test_basic_alignment(arr);
    }
    
    SECTION("int64_t alignment") {
        ContiguousND<std::int64_t> arr({1000});
        test_basic_alignment(arr);
    }
}

TEST_CASE("Memory alignment for AoS types", "[alignment][aos]") {
    SECTION("Vec2f alignment") {
        ContiguousND<aos::Vec2f> arr({100, 100});
        test_basic_alignment(arr);
        REQUIRE(alignof(aos::Vec2f) >= alignof(float));
        
        for (std::size_t i = 0; i < 10; ++i) {
            for (std::size_t j = 0; j < 10; ++j) {
                REQUIRE(is_aligned(&arr(i, j)));
            }
        }
    }
    
    SECTION("Vec3f alignment") {
        ContiguousND<aos::Vec3f> arr({50, 50, 50});
        test_basic_alignment(arr);
        REQUIRE(alignof(aos::Vec3f) >= alignof(float));
        
        REQUIRE(is_aligned(&arr(0, 0, 0)));
        REQUIRE(is_aligned(&arr(10, 20, 30)));
        REQUIRE(is_aligned(&arr(49, 49, 49)));
    }
    
    SECTION("Cell2D alignment") {
        ContiguousND<aos::Cell2D> arr({64, 64});
        test_basic_alignment(arr);
        REQUIRE(alignof(aos::Cell2D) >= alignof(float));
        REQUIRE(alignof(aos::Cell2D) >= alignof(std::int32_t));
        
        for (std::size_t i = 0; i < 64; i += 8) {
            for (std::size_t j = 0; j < 64; j += 8) {
                REQUIRE(is_aligned(&arr(i, j)));
            }
        }
    }
    
    SECTION("Cell3D alignment") {
        ContiguousND<aos::Cell3D> arr({32, 32, 32});
        test_basic_alignment(arr);
        REQUIRE(alignof(aos::Cell3D) >= alignof(float));
        
        REQUIRE(is_aligned(&arr(0, 0, 0)));
        REQUIRE(is_aligned(&arr(15, 15, 15)));
        REQUIRE(is_aligned(&arr(31, 31, 31)));
    }
    
    SECTION("Particle alignment") {
        ContiguousND<aos::Particle> particles({10000});
        test_basic_alignment(particles);
        REQUIRE(alignof(aos::Particle) >= alignof(double));
        
        for (std::size_t i = 0; i < 100; ++i) {
            REQUIRE(is_aligned(&particles(i)));
        }
    }
    
    SECTION("MaterialPoint alignment") {
        ContiguousND<aos::MaterialPoint> grid({50, 50});
        test_basic_alignment(grid);
        REQUIRE(alignof(aos::MaterialPoint) >= alignof(float));
        REQUIRE(alignof(aos::MaterialPoint) >= alignof(std::int32_t));
        
        for (std::size_t i = 0; i < 50; i += 5) {
            for (std::size_t j = 0; j < 50; j += 5) {
                REQUIRE(is_aligned(&grid(i, j)));
            }
        }
    }
}

TEST_CASE("Alignment properties of AoS types", "[alignment][properties]") {
    SECTION("Vec2f properties") {
        INFO("Vec2f size: " << sizeof(aos::Vec2f));
        INFO("Vec2f alignment: " << alignof(aos::Vec2f));
        
        // 2 floats = 8 bytes
        REQUIRE(sizeof(aos::Vec2f) == 8);
        REQUIRE(alignof(aos::Vec2f) == alignof(float));
    }
    
    SECTION("Vec3f properties") {
        INFO("Vec3f size: " << sizeof(aos::Vec3f));
        INFO("Vec3f alignment: " << alignof(aos::Vec3f));
        
        // 3 floats = 12 bytes
        REQUIRE(sizeof(aos::Vec3f) == 12);
        REQUIRE(alignof(aos::Vec3f) == alignof(float));
    }
    
    SECTION("Cell2D properties") {
        INFO("Cell2D size: " << sizeof(aos::Cell2D));
        INFO("Cell2D alignment: " << alignof(aos::Cell2D));
        
        // 2 floats + 1 int32 = 12 bytes (no padding needed)
        REQUIRE(sizeof(aos::Cell2D) == 12);
    }
    
    SECTION("Cell3D properties") {
        INFO("Cell3D size: " << sizeof(aos::Cell3D));
        INFO("Cell3D alignment: " << alignof(aos::Cell3D));
        
        // 3 floats + 1 int32 = 16 bytes
        REQUIRE(sizeof(aos::Cell3D) == 16);
    }
    
    SECTION("Particle properties") {
        INFO("Particle size: " << sizeof(aos::Particle));
        INFO("Particle alignment: " << alignof(aos::Particle));
        
        // 7 doubles = 56 bytes
        REQUIRE(sizeof(aos::Particle) == 56);
        REQUIRE(alignof(aos::Particle) == alignof(double));
    }
    
    SECTION("MaterialPoint properties") {
        INFO("MaterialPoint size: " << sizeof(aos::MaterialPoint));
        INFO("MaterialPoint alignment: " << alignof(aos::MaterialPoint));
        
        // 3 floats + 1 int32 = 16 bytes
        REQUIRE(sizeof(aos::MaterialPoint) == 16);
        REQUIRE(alignof(aos::MaterialPoint) == alignof(float));
    }
}

TEST_CASE("std::vector provides correct alignment", "[alignment][vector]") {
    SECTION("Verify std::vector alignment guarantees") {
        REQUIRE(is_aligned(std::vector<aos::Vec2f>(100).data()));
        REQUIRE(is_aligned(std::vector<aos::Vec3f>(100).data()));
        REQUIRE(is_aligned(std::vector<aos::Particle>(100).data()));
        REQUIRE(is_aligned(std::vector<aos::MaterialPoint>(100).data()));
    }
}

TEST_CASE("Alignment across different array sizes", "[alignment][boundary]") {
    SECTION("Small arrays") {
        test_basic_alignment(ContiguousND<aos::Vec2f>({1}));
        test_basic_alignment(ContiguousND<aos::Vec3f>({2}));
        test_basic_alignment(ContiguousND<aos::Particle>({3}));
    }
    
    SECTION("Large arrays") {
        test_basic_alignment(ContiguousND<aos::Vec2f>({10000}));
        test_basic_alignment(ContiguousND<aos::Cell3D>({100, 100}));
        test_basic_alignment(ContiguousND<aos::Particle>({5000}));
    }
    
    SECTION("Odd-sized arrays") {
        ContiguousND<aos::Vec3f> arr1({13, 17, 19});
        ContiguousND<aos::Cell2D> arr2({31, 37});
        
        test_basic_alignment(arr1);
        test_basic_alignment(arr2);
        
        REQUIRE(is_aligned(&arr1(12, 16, 18)));
        REQUIRE(is_aligned(&arr2(30, 36)));
    }
}

TEST_CASE("POD type guarantees for alignment safety", "[alignment][pod]") {
    // Test all AoS types at once using a helper
    auto test_type_traits = [](auto type_tag) {
        using T = decltype(type_tag);
        REQUIRE(std::is_standard_layout<T>::value);
        REQUIRE(std::is_trivially_copyable<T>::value);
        REQUIRE(std::is_pod<T>::value);
    };
    
    SECTION("All AoS types satisfy POD requirements") {
        test_type_traits(aos::Vec2f{});
        test_type_traits(aos::Vec3f{});
        test_type_traits(aos::Cell2D{});
        test_type_traits(aos::Cell3D{});
        test_type_traits(aos::Particle{});
        test_type_traits(aos::MaterialPoint{});
    }
}

TEST_CASE("No UB with aligned struct operations", "[alignment][safety]") {
    SECTION("Read and write operations are safe") {
        ContiguousND<aos::Vec3f> arr({100, 100});
        
        // This should not cause UB due to misalignment
        for (std::size_t i = 0; i < 100; ++i) {
            for (std::size_t j = 0; j < 100; ++j) {
                arr(i, j).x = static_cast<float>(i);
                arr(i, j).y = static_cast<float>(j);
                arr(i, j).z = static_cast<float>(i + j);
            }
        }
        
        // Verify values
        for (std::size_t i = 0; i < 100; ++i) {
            for (std::size_t j = 0; j < 100; ++j) {
                REQUIRE(arr(i, j).x == static_cast<float>(i));
                REQUIRE(arr(i, j).y == static_cast<float>(j));
                REQUIRE(arr(i, j).z == static_cast<float>(i + j));
            }
        }
    }
    
    SECTION("Memcpy operations are safe for POD types") {
        ContiguousND<aos::Particle> src({100});
        ContiguousND<aos::Particle> dst({100});
        
        // Initialize source
        for (std::size_t i = 0; i < 100; ++i) {
            src(i).x = static_cast<double>(i);
            src(i).y = static_cast<double>(i * 2);
            src(i).z = static_cast<double>(i * 3);
            src(i).mass = static_cast<double>(i + 1);
        }
        
        // Safe to memcpy POD types
        std::memcpy(dst.data(), src.data(), 100 * sizeof(aos::Particle));
        
        // Verify
        for (std::size_t i = 0; i < 100; ++i) {
            REQUIRE(dst(i).x == src(i).x);
            REQUIRE(dst(i).y == src(i).y);
            REQUIRE(dst(i).z == src(i).z);
            REQUIRE(dst(i).mass == src(i).mass);
        }
    }
}

TEST_CASE("Alignment with non-owning views", "[alignment][view]") {
    SECTION("External data preserves alignment") {
        auto owner = std::make_shared<std::vector<aos::Vec2f>>(100);
        aos::Vec2f* data_ptr = owner->data();
        
        REQUIRE(is_aligned(data_ptr));
        
        ContiguousND<aos::Vec2f> view({10, 10}, data_ptr, owner);
        
        REQUIRE(is_aligned(view.data()));
        REQUIRE(view.data() == data_ptr);
    }
}
