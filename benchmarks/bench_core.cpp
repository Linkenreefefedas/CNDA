/**
 * @file bench_core.cpp
 * @brief Core performance benchmarks for CNDA
 * 
 * Tests fundamental operations:
 * - Array construction overhead
 * - 1D/2D/3D sequential access (row-major)
 * - Random access patterns
 * - operator() vs at() bounds checking overhead
 * - Comparison with flat std::vector<T> and raw pointers
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <cnda/contiguous_nd.hpp>
#include <vector>
#include <numeric>

using namespace cnda;

// ============================================================================
// Construction Benchmarks
// ============================================================================

TEST_CASE("Construction overhead", "[benchmark][construction]") {
    BENCHMARK("ContiguousND<float> 100x100") {
        return ContiguousND<float>({100, 100});
    };
    
    BENCHMARK("ContiguousND<double> 100x100") {
        return ContiguousND<double>({100, 100});
    };
    
    BENCHMARK("ContiguousND<float> 1000x1000") {
        return ContiguousND<float>({1000, 1000});
    };
}

// ============================================================================
// Sequential Access Benchmarks (Cache-Friendly)
// ============================================================================

TEST_CASE("Sequential row-major access", "[benchmark][access][sequential]") {
    constexpr size_t N = 1000;
    ContiguousND<float> arr({N, N});
    
    // Baseline: raw pointer sequential write
    BENCHMARK("Raw pointer write 1000x1000") {
        float* data = arr.data();
        for (size_t i = 0; i < N * N; ++i) {
            data[i] = static_cast<float>(i);
        }
        return data[0];
    };
    
    // CNDA operator() row-major
    BENCHMARK("ContiguousND operator() write 1000x1000") {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                arr(i, j) = static_cast<float>(i * N + j);
            }
        }
        return arr(0, 0);
    };
    
    // Sequential read
    BENCHMARK("ContiguousND operator() read 1000x1000") {
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                sum += arr(i, j);
            }
        }
        return sum;
    };
}

// ============================================================================
// Random Access Benchmarks (Cache-Unfriendly)
// ============================================================================

TEST_CASE("Random access patterns", "[benchmark][access][random]") {
    constexpr size_t N = 1000;
    ContiguousND<float> arr({N, N});
    
    // Generate random indices
    std::vector<std::pair<size_t, size_t>> indices;
    indices.reserve(N * 10);
    for (size_t k = 0; k < N * 10; ++k) {
        indices.push_back({k % N, (k * 7) % N});
    }
    
    BENCHMARK("Random access read 10000 ops") {
        float sum = 0.0f;
        for (const auto& [i, j] : indices) {
            sum += arr(i, j);
        }
        return sum;
    };
}

// ============================================================================
// Bounds Checking Overhead
// ============================================================================

TEST_CASE("Bounds checking overhead", "[benchmark][bounds]") {
    ContiguousND<float> arr({100, 100});
    
    BENCHMARK("operator() no bounds check") {
        float sum = 0.0f;
        for (size_t i = 0; i < 100; ++i) {
            for (size_t j = 0; j < 100; ++j) {
                sum += arr(i, j);
            }
        }
        return sum;
    };
    
    BENCHMARK("at() with bounds check") {
        float sum = 0.0f;
        for (size_t i = 0; i < 100; ++i) {
            for (size_t j = 0; j < 100; ++j) {
                sum += arr.at({i, j});
            }
        }
        return sum;
    };
}

// ============================================================================
// Multi-dimensional Access
// ============================================================================

TEST_CASE("3D array access", "[benchmark][3d]") {
    constexpr size_t N = 100;
    ContiguousND<float> arr3d({N, N, N});
    
    BENCHMARK("3D operator() write 100x100x100") {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                for (size_t k = 0; k < N; ++k) {
                    arr3d(i, j, k) = static_cast<float>(i + j + k);
                }
            }
        }
        return arr3d(0, 0, 0);
    };
    
    BENCHMARK("3D sequential read 100x100x100") {
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                for (size_t k = 0; k < N; ++k) {
                    sum += arr3d(i, j, k);
                }
            }
        }
        return sum;
    };
}

// ============================================================================
// Memory Bandwidth Test
// ============================================================================

TEST_CASE("Memory bandwidth estimation", "[benchmark][bandwidth]") {
    constexpr size_t SIZE = 10'000'000; // 10M floats = 40MB
    ContiguousND<float> arr({SIZE});
    
    BENCHMARK("Sequential write 10M floats (40MB)") {
        for (size_t i = 0; i < SIZE; ++i) {
            arr(i) = static_cast<float>(i);
        }
        return arr(0);
    };
    
    BENCHMARK("Sequential read 10M floats (40MB)") {
        float sum = 0.0f;
        for (size_t i = 0; i < SIZE; ++i) {
            sum += arr(i);
        }
        return sum;
    };
    
    // Copy operation
    ContiguousND<float> arr2({SIZE});
    BENCHMARK("Copy 10M floats (40MB)") {
        float* src = arr.data();
        float* dst = arr2.data();
        for (size_t i = 0; i < SIZE; ++i) {
            dst[i] = src[i];
        }
        return dst[0];
    };
}

// ============================================================================
// Comparison with flat std::vector and raw pointer
// ============================================================================

TEST_CASE("Comparison: CNDA vs flat std::vector", "[benchmark][comparison]") {
    constexpr size_t N = 1000;
    
    // CNDA 2D array
    ContiguousND<float> cnda_arr({N, N});
    
    BENCHMARK("CNDA operator(i, j)") {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                cnda_arr(i, j) = static_cast<float>(i * N + j);
            }
        }
        return cnda_arr(0, 0);
    };
    
    // Flat std::vector with manual indexing
    std::vector<float> flat_vec(N * N);
    
    BENCHMARK("std::vector<T> manual indexing") {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                flat_vec[i * N + j] = static_cast<float>(i * N + j);
            }
        }
        return flat_vec[0];
    };
    
    // Raw pointer (theoretical best)
    float* raw_ptr = cnda_arr.data();
    
    BENCHMARK("Raw pointer (baseline)") {
        for (size_t i = 0; i < N * N; ++i) {
            raw_ptr[i] = static_cast<float>(i);
        }
        return raw_ptr[0];
    };
}