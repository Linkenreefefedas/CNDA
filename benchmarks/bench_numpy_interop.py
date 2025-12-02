"""
NumPy interoperability performance benchmarks

Tests:
1. Zero-copy overhead (from_numpy/to_numpy)
2. Deep copy overhead
3. Round-trip conversion latency
4. Memory bandwidth utilization
"""

import pytest
import numpy as np
import sys
import os

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'python', 'Release'))

import cnda


# ============================================================================
# Zero-Copy Overhead
# ============================================================================

@pytest.mark.benchmark(group="zero-copy-from-numpy")
def test_from_numpy_zero_copy_small(benchmark):
    """Small array (1KB): NumPy → CNDA zero-copy"""
    arr = np.arange(256, dtype=np.float32)  # 1KB
    
    def convert():
        return cnda.from_numpy_f32(arr, copy=False)
    
    result = benchmark(convert)
    assert result.size == 256


@pytest.mark.benchmark(group="zero-copy-from-numpy")
def test_from_numpy_zero_copy_large(benchmark):
    """Large array (100MB): NumPy → CNDA zero-copy"""
    arr = np.arange(25_000_000, dtype=np.float32).reshape(5000, 5000)  # ~100MB
    
    def convert():
        return cnda.from_numpy_f32(arr, copy=False)
    
    result = benchmark(convert)
    assert result.size == 25_000_000


@pytest.mark.benchmark(group="zero-copy-to-numpy")
def test_to_numpy_zero_copy_small(benchmark):
    """Small array (1KB): CNDA → NumPy zero-copy"""
    arr = cnda.ContiguousND_f32([256])
    
    def convert():
        return arr.to_numpy(copy=False)
    
    result = benchmark(convert)
    assert result.size == 256


@pytest.mark.benchmark(group="zero-copy-to-numpy")
def test_to_numpy_zero_copy_large(benchmark):
    """Large array (100MB): CNDA → NumPy zero-copy"""
    arr = cnda.ContiguousND_f32([5000, 5000])  # ~100MB
    
    def convert():
        return arr.to_numpy(copy=False)
    
    result = benchmark(convert)
    assert result.size == 25_000_000


# ============================================================================
# Deep Copy Overhead
# ============================================================================

@pytest.mark.benchmark(group="deep-copy")
def test_from_numpy_deep_copy_1mb(benchmark):
    """1MB array: NumPy → CNDA deep copy"""
    arr = np.arange(250_000, dtype=np.float32)  # 1MB
    
    def convert():
        return cnda.from_numpy_f32(arr, copy=True)
    
    result = benchmark(convert)
    assert result.size == 250_000


@pytest.mark.benchmark(group="deep-copy")
def test_from_numpy_deep_copy_100mb(benchmark):
    """100MB array: NumPy → CNDA deep copy"""
    arr = np.arange(25_000_000, dtype=np.float32)  # 100MB
    
    def convert():
        return cnda.from_numpy_f32(arr, copy=True)
    
    result = benchmark(convert)
    assert result.size == 25_000_000


@pytest.mark.benchmark(group="deep-copy")
def test_to_numpy_deep_copy_100mb(benchmark):
    """100MB array: CNDA → NumPy deep copy"""
    arr = cnda.ContiguousND_f32([25_000_000])
    
    def convert():
        return arr.to_numpy(copy=True)
    
    result = benchmark(convert)
    assert result.size == 25_000_000


# ============================================================================
# Round-Trip Latency
# ============================================================================

@pytest.mark.benchmark(group="round-trip")
def test_roundtrip_zero_copy_2d(benchmark):
    """Round-trip: NumPy → CNDA → NumPy (zero-copy)"""
    original = np.arange(10000, dtype=np.float32).reshape(100, 100)
    
    def roundtrip():
        cnda_arr = cnda.from_numpy_f32(original, copy=False)
        return cnda_arr.to_numpy(copy=False)
    
    result = benchmark(roundtrip)
    assert result.shape == (100, 100)


@pytest.mark.benchmark(group="round-trip")
def test_roundtrip_deep_copy_2d(benchmark):
    """Round-trip: NumPy → CNDA → NumPy (deep-copy)"""
    original = np.arange(10000, dtype=np.float32).reshape(100, 100)
    
    def roundtrip():
        cnda_arr = cnda.from_numpy_f32(original, copy=True)
        return cnda_arr.to_numpy(copy=True)
    
    result = benchmark(roundtrip)
    assert result.shape == (100, 100)


# ============================================================================
# Memory Access Patterns
# ============================================================================

@pytest.mark.benchmark(group="access-pattern")
def test_sequential_access_numpy(benchmark):
    """Sequential access: pure NumPy"""
    arr = np.arange(1_000_000, dtype=np.float32)
    
    def access():
        return np.sum(arr)
    
    result = benchmark(access)
    assert result > 0


@pytest.mark.benchmark(group="access-pattern")
def test_sequential_access_cnda_from_numpy(benchmark):
    """Sequential access: CNDA from NumPy (zero-copy)"""
    np_arr = np.arange(1_000_000, dtype=np.float32)
    cnda_arr = cnda.from_numpy_f32(np_arr, copy=False)
    
    def access():
        # Access through NumPy view
        return cnda_arr.to_numpy(copy=False).sum()
    
    result = benchmark(access)
    assert result > 0


# ============================================================================
# AoS Interop
# ============================================================================

@pytest.mark.benchmark(group="aos-interop")
def test_aos_vec2f_from_numpy(benchmark):
    """Vec2f structured array: NumPy → CNDA"""
    dtype = np.dtype([('x', np.float32), ('y', np.float32)])
    arr = np.zeros((1000, 1000), dtype=dtype)
    
    def convert():
        return cnda.from_numpy_Vec2f(arr, copy=False)
    
    result = benchmark(convert)
    assert result.size == 1_000_000


@pytest.mark.benchmark(group="aos-interop")
def test_aos_vec2f_to_numpy(benchmark):
    """Vec2f structured array: CNDA → NumPy"""
    arr = cnda.ContiguousND_Vec2f([1000, 1000])
    
    def convert():
        return arr.to_numpy(copy=False)
    
    result = benchmark(convert)
    assert result.size == 1_000_000


# ============================================================================
# Comparison with Pure NumPy Operations
# ============================================================================

@pytest.mark.benchmark(group="comparison")
def test_numpy_array_creation(benchmark):
    """Baseline: NumPy array creation"""
    
    def create():
        return np.zeros((1000, 1000), dtype=np.float32)
    
    result = benchmark(create)
    assert result.shape == (1000, 1000)


@pytest.mark.benchmark(group="comparison")
def test_cnda_array_creation(benchmark):
    """CNDA array creation"""
    
    def create():
        return cnda.ContiguousND_f32([1000, 1000])
    
    result = benchmark(create)
    assert result.shape == (1000, 1000)


@pytest.mark.benchmark(group="comparison")
def test_numpy_copy(benchmark):
    """Baseline: NumPy array copy"""
    arr = np.arange(1_000_000, dtype=np.float32)
    
    def copy_arr():
        return arr.copy()
    
    result = benchmark(copy_arr)
    assert result.size == 1_000_000


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--benchmark-only', '--benchmark-group-by=group'])
