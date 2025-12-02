"""
Test suite for NumPy interoperability with CNDA.

This test suite covers:
1. C++ → NumPy: copy=False (zero-copy view) and copy=True (deep copy)
2. NumPy → C++: copy=False (zero-copy view) and copy=True (deep copy)
3. Non-C-contiguous arrays (Fortran-order, transposed, sliced)
4. All supported dtypes (f32, f64, i32, i64)
5. Capsule ownership and lifetime safety
"""

import sys
import os
import pytest


@pytest.fixture(scope="module")
def np():
    """Import numpy or skip tests if not available."""
    pytest.importorskip('numpy')
    import numpy
    return numpy


@pytest.fixture(scope="module")
def cnda():
    """Import cnda module."""
    import cnda as cnda_module
    return cnda_module


# ==============================================================================
# NumPy → C++ Tests (from_numpy)
# ==============================================================================

class TestFromNumpyZeroCopy:
    """Test from_numpy with copy=False (zero-copy view from NumPy to C++)."""
    
    @pytest.mark.parametrize(
        "dtype_name, func_name",
        [
            ("float32", "from_numpy_f32"),
            ("float64", "from_numpy_f64"),
            ("int32", "from_numpy_i32"),
            ("int64", "from_numpy_i64"),
        ],
    )
    def test_c_contiguous_zero_copy(self, np, cnda, dtype_name, func_name):
        np_dtype = getattr(np, dtype_name)
        cnda_from_numpy_func = getattr(cnda, func_name)
        """Test zero-copy from C-contiguous NumPy arrays for all supported dtypes."""
        x = np.array([[1, 2], [3, 4]], dtype=np_dtype, order='C')
        assert x.flags['C_CONTIGUOUS']
        
        arr = cnda_from_numpy_func(x, copy=False)
        
        # Verify values
        assert arr[0, 0] == 1
        assert arr[1, 1] == 4
        
        # Verify shape and metadata
        assert arr.shape == (2, 2)
        assert arr.ndim == 2
        assert arr.size == 4
        
        # Verify zero-copy (same memory address)
        assert arr.data_ptr() == x.ctypes.data
    
    def test_generic_from_numpy(self, np, cnda):
        """Test generic from_numpy with automatic dtype detection."""
        # Test float32
        x_f32 = np.array([[1.0, 2.0]], dtype=np.float32)
        arr_f32 = cnda.from_numpy(x_f32, copy=False)
        assert type(arr_f32).__name__ == 'ContiguousND_f32'
        assert arr_f32.data_ptr() == x_f32.ctypes.data
        
        # Test float64
        x_f64 = np.array([[1.0, 2.0]], dtype=np.float64)
        arr_f64 = cnda.from_numpy(x_f64, copy=False)
        assert type(arr_f64).__name__ == 'ContiguousND_f64'
        
        # Test int32
        x_i32 = np.array([[1, 2]], dtype=np.int32)
        arr_i32 = cnda.from_numpy(x_i32, copy=False)
        assert type(arr_i32).__name__ == 'ContiguousND_i32'
        
        # Test int64
        x_i64 = np.array([[1, 2]], dtype=np.int64)
        arr_i64 = cnda.from_numpy(x_i64, copy=False)
        assert type(arr_i64).__name__ == 'ContiguousND_i64'
    
    def test_memory_sharing_from_numpy(self, np, cnda):
        """Test that modifications in NumPy are reflected in C++ (zero-copy)."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='C')
        arr = cnda.from_numpy_f32(x, copy=False)
        
        # Modify NumPy array
        x[0, 0] = 99.0
        
        # Should be reflected in C++ array (zero-copy)
        assert arr[0, 0] == 99.0


class TestFromNumpyDeepCopy:
    """Test from_numpy with copy=True (deep copy from NumPy to C++)."""
    
    def test_deep_copy_from_numpy(self, np, cnda):
        """Test deep copy from float32 NumPy array."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        arr = cnda.from_numpy_f32(x, copy=True)
        
        # Verify values
        assert arr[0, 0] == 1.0
        assert arr[1, 1] == 4.0
        
        # Verify it's a copy (different memory addresses)
        assert arr.data_ptr() != x.ctypes.data
    
    def test_memory_independence_on_deep_copy(self, np, cnda):
        """Test that modifications in NumPy don't affect C++ (deep copy)."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        arr = cnda.from_numpy_f32(x, copy=True)
        
        original_value = arr[0, 0]
        
        # Modify NumPy array
        x[0, 0] = 99.0
        
        # C++ array should be unchanged (deep copy)
        assert arr[0, 0] == original_value
    
    def test_fortran_order_with_copy(self, np, cnda):
        """Test that copy=True works with Fortran-order arrays."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='F')
        assert x.flags['F_CONTIGUOUS']
        assert not x.flags['C_CONTIGUOUS']
        
        # Should work with copy=True
        arr = cnda.from_numpy_f32(x, copy=True)
        assert arr[0, 0] == 1.0
        assert arr[1, 1] == 4.0
    
    def test_transposed_array_with_copy(self, np, cnda):
        """Test that copy=True works with transposed (non-contiguous) arrays."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).T
        assert not x.flags['C_CONTIGUOUS']
        
        # Should work with copy=True
        arr = cnda.from_numpy_f32(x, copy=True)
        # After transpose: x.T[0,0]=1.0, x.T[0,1]=3.0, x.T[1,0]=2.0, x.T[1,1]=4.0
        assert arr[0, 0] == 1.0
        assert arr[0, 1] == 3.0
        assert arr[1, 0] == 2.0
        assert arr[1, 1] == 4.0
    
    def test_sliced_array_with_copy(self, np, cnda):
        """Test that copy=True works with sliced (non-contiguous) arrays."""
        x = np.arange(20, dtype=np.float32).reshape(5, 4)
        x_sliced = x[::2, ::2]  # Non-contiguous slice
        assert not x_sliced.flags['C_CONTIGUOUS']
        
        # Should work with copy=True
        arr = cnda.from_numpy_f32(x_sliced, copy=True)
        assert arr.shape == (3, 2)

# ==============================================================================
# C++ → NumPy Tests (to_numpy)
# ==============================================================================

class TestToNumpyZeroCopy:
    """Test to_numpy with copy=False (zero-copy view from C++ to NumPy)."""
    
    @pytest.mark.parametrize(
        "class_name, dtype_name",
        [
            ("ContiguousND_f32", "float32"),
            ("ContiguousND_f64", "float64"),
            ("ContiguousND_i32", "int32"),
            ("ContiguousND_i64", "int64"),
        ],
    )
    def test_zero_copy_to_numpy(self, np, cnda, class_name, dtype_name):
        cnda_class = getattr(cnda, class_name)
        np_dtype = getattr(np, dtype_name)
        """Test zero-copy export to NumPy for all supported dtypes."""
        arr = cnda_class([2, 3])
        arr[0, 0] = 1
        arr[1, 2] = 99
        
        np_arr = arr.to_numpy(copy=False)
        
        # Verify values
        assert np_arr[0, 0] == 1
        assert np_arr[1, 2] == 99
        
        # Verify metadata
        assert np_arr.shape == (2, 3)
        assert np_arr.dtype == np_dtype
        assert np_arr.flags['C_CONTIGUOUS']
        
        # Verify zero-copy (same memory address)
        assert arr.data_ptr() == np_arr.ctypes.data
    
    def test_memory_sharing_to_numpy(self, np, cnda):
        """Test that modifications in C++ are reflected in NumPy (zero-copy)."""
        arr = cnda.ContiguousND_f32([2, 2])
        arr[0, 0] = 1.0
        
        np_arr = arr.to_numpy(copy=False)
        
        # Modify C++ array
        arr[0, 0] = 99.0
        
        # Should be reflected in NumPy array (zero-copy)
        assert np_arr[0, 0] == 99.0


class TestToNumpyDeepCopy:
    """Test to_numpy with copy=True (deep copy from C++ to NumPy)."""
    
    def test_memory_independence_on_deep_copy(self, np, cnda):
        """Test that modifications in C++ don't affect NumPy (deep copy)."""
        arr = cnda.ContiguousND_f32([2, 2])
        arr[0, 0] = 1.0
        
        np_arr = arr.to_numpy(copy=True)
        original_value = np_arr[0, 0]
        
        # Modify C++ array
        arr[0, 0] = 99.0
        
        # NumPy array should be unchanged (deep copy)
        assert np_arr[0, 0] == original_value
    
    def test_all_dtypes_deep_copy(self, np, cnda):
        """Test deep copy for all supported dtypes."""
        test_cases = [
            (cnda.ContiguousND_f32, np.float32),
            (cnda.ContiguousND_f64, np.float64),
            (cnda.ContiguousND_i32, np.int32),
            (cnda.ContiguousND_i64, np.int64),
        ]
        
        for cnda_class, np_dtype in test_cases:
            arr = cnda_class([2, 2])
            np_arr = arr.to_numpy(copy=True)
            
            assert np_arr.dtype == np_dtype
            assert arr.data_ptr() != np_arr.ctypes.data


# ==============================================================================
# Round-trip Tests
# ==============================================================================

class TestRoundTrip:
    """Test round-trip conversions between NumPy and C++."""
    
    def test_numpy_to_cpp_to_numpy_zero_copy(self, np, cnda):
        """Test NumPy → C++ → NumPy with zero-copy."""
        x_orig = np.arange(12, dtype=np.float32).reshape(3, 4)
        
        # NumPy → C++ (zero-copy)
        arr = cnda.from_numpy_f32(x_orig, copy=False)
        
        # C++ → NumPy (zero-copy)
        x_back = arr.to_numpy(copy=False)
        
        # All three should share memory
        assert x_orig.ctypes.data == arr.data_ptr()
        assert arr.data_ptr() == x_back.ctypes.data
        
        # Values should match
        assert np.array_equal(x_orig, x_back)
    
    def test_numpy_to_cpp_to_numpy_deep_copy(self, np, cnda):
        """Test NumPy → C++ → NumPy with deep copies."""
        x_orig = np.arange(6, dtype=np.float64).reshape(2, 3)
        
        # NumPy → C++ (deep copy)
        arr = cnda.from_numpy_f64(x_orig, copy=True)
        
        # C++ → NumPy (deep copy)
        x_back = arr.to_numpy(copy=True)
        
        # All should have different memory
        assert x_orig.ctypes.data != arr.data_ptr()
        assert arr.data_ptr() != x_back.ctypes.data
        
        # Values should match
        assert np.allclose(x_orig, x_back)
    
    def test_modification_persistence_zero_copy(self, np, cnda):
        """Test that modifications persist through zero-copy round-trip."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='C')
        arr = cnda.from_numpy_f32(x, copy=False)
        
        # Modify via C++
        arr[0, 0] = 99.0
        
        # Convert back to NumPy (zero-copy)
        y = arr.to_numpy(copy=False)
        
        # Original NumPy array should reflect change
        assert x[0, 0] == 99.0
        assert y[0, 0] == 99.0


# ==============================================================================
# Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_1d_array(self, np, cnda):
        """Test 1D array conversion."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        arr = cnda.from_numpy_f32(x, copy=False)
        
        assert arr.ndim == 1
        assert arr.shape == (4,)
        assert arr[0] == 1.0
        assert arr[3] == 4.0
        
        y = arr.to_numpy(copy=False)
        assert y.shape == (4,)
    
    def test_3d_array(self, np, cnda):
        """Test 3D array conversion."""
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        arr = cnda.from_numpy_f32(x, copy=False)
        
        assert arr.ndim == 3
        assert arr.shape == (2, 3, 4)
        
        y = arr.to_numpy(copy=False)
        assert y.shape == (2, 3, 4)
        assert np.array_equal(x, y)
    
    def test_zero_sized_dimension(self, np, cnda):
        """Test array with zero-sized dimension."""
        x = np.zeros((0, 5), dtype=np.float32)
        arr = cnda.from_numpy_f32(x, copy=True)
        
        assert arr.shape == (0, 5)
        assert arr.size == 0
        
        y = arr.to_numpy(copy=False)
        assert y.shape == (0, 5)
    
    def test_large_array(self, np, cnda):
        """Test larger array conversion."""
        x = np.arange(10000, dtype=np.float32).reshape(100, 100)
        arr = cnda.from_numpy_f32(x, copy=False)
        
        assert arr.shape == (100, 100)
        assert arr.size == 10000
        
        # Spot check some values
        assert arr[0, 0] == 0.0
        assert arr[50, 50] == 5050.0
        assert arr[99, 99] == 9999.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
