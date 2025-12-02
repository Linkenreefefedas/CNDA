/**
 * @file bench_aos.cpp
 * @brief Array-of-Structures (AoS) performance benchmarks
 * 
 * Tests:
 * - AoS field access patterns
 * - Cache line utilization
 * - Struct size impact on performance
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <cnda/contiguous_nd.hpp>
#include <cnda/aos_types.hpp>

using namespace cnda;
using namespace cnda::aos;

// ============================================================================
// Small Struct: Vec2f (8 bytes)
// ============================================================================

TEST_CASE("Vec2f access patterns", "[benchmark][aos][vec2f]") {
    constexpr size_t N = 1000;
    ContiguousND<Vec2f> arr({N, N});
    
    BENCHMARK("Vec2f write all fields 1000x1000") {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                arr(i, j).x = static_cast<float>(i);
                arr(i, j).y = static_cast<float>(j);
            }
        }
        return arr(0, 0).x;
    };
    
    BENCHMARK("Vec2f read magnitude calculation") {
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                const auto& v = arr(i, j);
                sum += v.x * v.x + v.y * v.y;
            }
        }
        return sum;
    };
    
    // Single field access (partial cache line usage)
    BENCHMARK("Vec2f read x field only") {
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                sum += arr(i, j).x;
            }
        }
        return sum;
    };
}

// ============================================================================
// Medium Struct: Cell2D (12 bytes)
// ============================================================================

TEST_CASE("Cell2D fluid simulation", "[benchmark][aos][cell2d]") {
    constexpr size_t N = 500;
    ContiguousND<Cell2D> grid({N, N});
    
    // Initialize velocity field
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            grid(i, j) = {1.0f, 0.5f, 1};  // u, v, flag
        }
    }
    
    BENCHMARK("Cell2D velocity field update (advection)") {
        for (size_t i = 1; i < N-1; ++i) {
            for (size_t j = 1; j < N-1; ++j) {
                auto& cell = grid(i, j);
                const auto& left = grid(i-1, j);
                const auto& right = grid(i+1, j);
                const auto& bottom = grid(i, j-1);
                const auto& top = grid(i, j+1);
                
                cell.u = 0.25f * (left.u + right.u + bottom.u + top.u);
                cell.v = 0.25f * (left.v + right.v + bottom.v + top.v);
            }
        }
        return grid(N/2, N/2).u;
    };
}

// ============================================================================
// Large Struct: Particle (56 bytes)
// ============================================================================

TEST_CASE("Particle system", "[benchmark][aos][particle]") {
    constexpr size_t N = 100'000; // 100k particles
    ContiguousND<Particle> particles({N});
    
    // Initialize
    for (size_t i = 0; i < N; ++i) {
        particles(i) = {
            static_cast<double>(i % 100),     // x
            static_cast<double>(i / 100),     // y
            0.0,                              // z
            1.0, 1.0, 0.0,                   // vx, vy, vz
            1.0                               // mass
        };
    }
    
    BENCHMARK("Particle position update (Euler)") {
        const double dt = 0.01;
        for (size_t i = 0; i < N; ++i) {
            auto& p = particles(i);
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.z += p.vz * dt;
        }
        return particles(0).x;
    };
    
    BENCHMARK("Particle gravity force") {
        const double g = -9.81;
        const double dt = 0.01;
        for (size_t i = 0; i < N; ++i) {
            auto& p = particles(i);
            p.vz += g * dt;
            p.z += p.vz * dt;
        }
        return particles(0).z;
    };
}

// ============================================================================
// Cache Efficiency Test
// ============================================================================

TEST_CASE("Cache efficiency: struct size impact", "[benchmark][cache]") {
    constexpr size_t N = 1'000'000;
    
    // Small struct (8 bytes) - 8 per cache line
    ContiguousND<Vec2f> small({N});
    BENCHMARK("Vec2f sequential access 1M") {
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            sum += small(i).x + small(i).y;
        }
        return sum;
    };
    
    // Medium struct (12 bytes) - 5 per cache line
    ContiguousND<Vec3f> medium({N});
    BENCHMARK("Vec3f sequential access 1M") {
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            sum += medium(i).x + medium(i).y + medium(i).z;
        }
        return sum;
    };
    
    // Large struct (56 bytes) - 1 per cache line
    ContiguousND<Particle> large({N});
    BENCHMARK("Particle sequential access 1M") {
        double sum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            sum += large(i).x + large(i).y + large(i).z;
        }
        return sum;
    };
}

// ============================================================================
// AoS vs SoA Comparison (Conceptual)
// ============================================================================

// Simulated SoA layout for comparison
struct Vec2fSoA {
    std::vector<float> x;
    std::vector<float> y;
    Vec2fSoA(size_t n) : x(n), y(n) {}
};

TEST_CASE("AoS vs SoA layout", "[benchmark][comparison]") {
    constexpr size_t N = 1'000'000;
    
    // AoS: ContiguousND<Vec2f>
    ContiguousND<Vec2f> aos({N});
    for (size_t i = 0; i < N; ++i) {
        aos(i) = {static_cast<float>(i), static_cast<float>(i * 2)};
    }
    
    BENCHMARK("AoS: access both fields") {
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            const auto& v = aos(i);
            sum += v.x + v.y;
        }
        return sum;
    };
    
    BENCHMARK("AoS: access single field (x)") {
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            sum += aos(i).x;
        }
        return sum;
    };
    
    // SoA: separate arrays
    Vec2fSoA soa(N);
    for (size_t i = 0; i < N; ++i) {
        soa.x[i] = static_cast<float>(i);
        soa.y[i] = static_cast<float>(i * 2);
    }
    
    BENCHMARK("SoA: access single field (x)") {
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            sum += soa.x[i];
        }
        return sum;
    };
    
    // Note: SoA wins for single-field access due to better cache utilization
    // AoS wins when accessing multiple fields together
}
