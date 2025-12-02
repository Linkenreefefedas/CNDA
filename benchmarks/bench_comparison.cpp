/**
 * @file bench_comparison.cpp
 * @brief High-level comparison: CNDA vs nested std::vector
 * 
 * Focuses on:
 * 1. 3D arrays: ContiguousND vs std::vector<std::vector<std::vector<T>>>
 * 2. Memory overhead calculation (nested vector inefficiency)
 * 
 * For core operations comparison, see bench_core.cpp
 * For Python comparison, see bench_numpy_interop.py
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <cnda/contiguous_nd.hpp>
#include <vector>
#include <numeric>

using namespace cnda;

// ============================================================================
// 1. Memory Overhead Comparison
// ============================================================================

TEST_CASE("Memory overhead: CNDA vs nested vector", "[benchmark][comparison][memory]") {
    
    const size_t N = 1000;
    
    INFO("=== Memory Overhead Analysis ===");
    INFO("Array size: " << N << "x" << N << " floats");
    INFO("Pure data size: " << (N * N * sizeof(float) / 1024.0) << " KB");
    INFO("");
    
    // CNDA: minimal overhead (shape + strides vectors)
    ContiguousND<float> cnda_arr({N, N});
    size_t cnda_metadata = sizeof(ContiguousND<float>) - sizeof(std::vector<float>);
    INFO("CNDA metadata: ~" << cnda_metadata << " bytes");
    
    // std::vector<vector<T>>: N inner vector objects
    std::vector<std::vector<float>> nested(N, std::vector<float>(N));
    size_t nested_overhead = N * sizeof(std::vector<float>);
    INFO("Nested vector overhead: ~" << nested_overhead / 1024.0 << " KB");
    INFO("  (Each inner vector: ~" << sizeof(std::vector<float>) << " bytes x " << N << " rows)");
    INFO("");
    
    // Percentage calculation
    double cnda_overhead_pct = 100.0 * cnda_metadata / (N * N * sizeof(float));
    double nested_overhead_pct = 100.0 * nested_overhead / (N * N * sizeof(float));
    
    INFO("CNDA overhead: " << cnda_overhead_pct << "% (negligible)");
    INFO("Nested vector overhead: " << nested_overhead_pct << "% (significant!)");
    INFO("");
    INFO("Conclusion: CNDA uses ~" << (nested_overhead / cnda_metadata) 
         << "x less metadata than nested vector");
    
    BENCHMARK("Memory analysis complete") {
        return cnda_arr(0, 0);
    };
}

// ============================================================================
// 2. 3D Array Comparison: Clean API vs Nested Vector Complexity
// ============================================================================

TEST_CASE("3D arrays: CNDA vs triple nested vector", "[benchmark][comparison][3d]") {
    
    constexpr size_t D1 = 100, D2 = 100, D3 = 100;
    
    INFO("=== 3D Array Comparison ===");
    INFO("Dimensions: " << D1 << "x" << D2 << "x" << D3);
    INFO("Total elements: " << (D1 * D2 * D3));
    INFO("Memory: " << (D1 * D2 * D3 * sizeof(float) / 1024.0 / 1024.0) << " MB");
    INFO("");
    
    // CNDA: Clean API with contiguous memory
    ContiguousND<float> cnda_3d({D1, D2, D3});
    BENCHMARK("CNDA: arr(i,j,k) - Clean & Contiguous") {
        float sum = 0.0f;
        for (size_t i = 0; i < D1; ++i) {
            for (size_t j = 0; j < D2; ++j) {
                for (size_t k = 0; k < D3; ++k) {
                    cnda_3d(i, j, k) = static_cast<float>(i + j + k);
                    sum += cnda_3d(i, j, k);
                }
            }
        }
        return sum;
    };
    
    // Triple nested vector: Complex construction, fragmented memory
    std::vector<std::vector<std::vector<float>>> nested_3d(
        D1, std::vector<std::vector<float>>(
            D2, std::vector<float>(D3)
        )
    );
    BENCHMARK("Nested vector: vec[i][j][k] - Fragmented") {
        float sum = 0.0f;
        for (size_t i = 0; i < D1; ++i) {
            for (size_t j = 0; j < D2; ++j) {
                for (size_t k = 0; k < D3; ++k) {
                    nested_3d[i][j][k] = static_cast<float>(i + j + k);
                    sum += nested_3d[i][j][k];
                }
            }
        }
        return sum;
    };
    
    INFO("");
    INFO("Key difference:");
    INFO("- CNDA: Single contiguous allocation, cache-friendly");
    INFO("- Nested vector: " << (D1 * D2 + D1 + 1) << " separate allocations!");
}
