"""
Test suite for NumPy interoperability with CNDA.

This test suite covers:
1. C++ → NumPy: copy=False (zero-copy view) and copy=True (deep copy)
2. NumPy → C++: copy=False (zero-copy view) and copy=True (deep copy)
3. Non-C-contiguous arrays (Fortran-order, transposed, sliced)
4. Unsupported dtypes error handling
5. All supported dtypes (f32, f64, i32, i64)
"""

import sys
import os
import pytest

# Add build directory to path
build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build', 'python', 'Release')
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)


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
    
    def test_f32_c_contiguous(self, np, cnda):
        """Test zero-copy from C-contiguous float32 NumPy array."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='C')
        assert x.flags['C_CONTIGUOUS']
        
        arr = cnda.from_numpy_f32(x, copy=False)
        
        # Verify values
        assert arr[0, 0] == 1.0
        assert arr[0, 1] == 2.0
        assert arr[1, 0] == 3.0
        assert arr[1, 1] == 4.0
        
        # Verify shape and metadata
        assert arr.shape == (2, 2)
        assert arr.ndim == 2
        assert arr.size == 4
        
        # Verify zero-copy (same memory address)
        assert arr.data_ptr() == x.ctypes.data
    
    def test_f64_c_contiguous(self, np, cnda):
        """Test zero-copy from C-contiguous float64 NumPy array."""
        x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64, order='C')
        arr = cnda.from_numpy_f64(x, copy=False)
        
        assert arr[0, 0] == 1.5
        assert arr[1, 1] == 4.5
        assert arr.data_ptr() == x.ctypes.data
    
    def test_i32_c_contiguous(self, np, cnda):
        """Test zero-copy from C-contiguous int32 NumPy array."""
        x = np.array([[1, 2], [3, 4]], dtype=np.int32, order='C')
        arr = cnda.from_numpy_i32(x, copy=False)
        
        assert arr[0, 0] == 1
        assert arr[1, 1] == 4
        assert arr.data_ptr() == x.ctypes.data
    
    def test_i64_c_contiguous(self, np, cnda):
        """Test zero-copy from C-contiguous int64 NumPy array."""
        x = np.array([[1, 2], [3, 4]], dtype=np.int64, order='C')
        arr = cnda.from_numpy_i64(x, copy=False)
        
        assert arr[0, 0] == 1
        assert arr[1, 1] == 4
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
    
    def test_memory_sharing(self, np, cnda):
        """Test that modifications in NumPy are reflected in C++ (zero-copy)."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='C')
        arr = cnda.from_numpy_f32(x, copy=False)
        
        # Modify NumPy array
        x[0, 0] = 99.0
        
        # Should be reflected in C++ array (zero-copy)
        assert arr[0, 0] == 99.0


class TestFromNumpyDeepCopy:
    """Test from_numpy with copy=True (deep copy from NumPy to C++)."""
    
    def test_f32_deep_copy(self, np, cnda):
        """Test deep copy from float32 NumPy array."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        arr = cnda.from_numpy_f32(x, copy=True)
        
        # Verify values
        assert arr[0, 0] == 1.0
        assert arr[1, 1] == 4.0
        
        # Verify it's a copy (different memory addresses)
        assert arr.data_ptr() != x.ctypes.data
    
    def test_memory_independence(self, np, cnda):
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


class TestFromNumpyNonContiguous:
    """Test from_numpy with non-C-contiguous arrays (should fail with copy=False)."""
    
    def test_fortran_order_error(self, np, cnda):
        """Test that Fortran-order array raises error with copy=False."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='F')
        assert not x.flags['C_CONTIGUOUS']
        
        with pytest.raises((ValueError, RuntimeError), match="C-contiguous"):
            cnda.from_numpy_f32(x, copy=False)
    
    def test_transposed_array_error(self, np, cnda):
        """Test that transposed array raises error with copy=False."""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).T
        assert not x.flags['C_CONTIGUOUS']
        
        with pytest.raises((ValueError, RuntimeError), match="C-contiguous"):
            cnda.from_numpy_f32(x, copy=False)
    
    def test_sliced_array_error(self, np, cnda):
        """Test that sliced (non-contiguous) array raises error with copy=False."""
        x = np.arange(20, dtype=np.float32).reshape(5, 4)
        x_sliced = x[::2, ::2]  # Non-contiguous
        
        if not x_sliced.flags['C_CONTIGUOUS']:
            with pytest.raises((ValueError, RuntimeError), match="C-contiguous"):
                cnda.from_numpy_f32(x_sliced, copy=False)


class TestFromNumpyUnsupportedDtype:
    """Test from_numpy with unsupported dtypes."""
    
    def test_uint8_error(self, np, cnda):
        """Test that uint8 dtype raises TypeError."""
        x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        
        with pytest.raises(TypeError, match="Unsupported dtype"):
            cnda.from_numpy(x, copy=False)
    
    def test_uint16_error(self, np, cnda):
        """Test that uint16 dtype raises TypeError."""
        x = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        
        with pytest.raises(TypeError, match="Unsupported dtype"):
            cnda.from_numpy(x, copy=False)
    
    def test_float16_error(self, np, cnda):
        """Test that float16 dtype raises TypeError."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
        
        with pytest.raises(TypeError, match="Unsupported dtype"):
            cnda.from_numpy(x, copy=False)
    
    def test_complex_error(self, np, cnda):
        """Test that complex dtype raises TypeError."""
        x = np.array([[1+2j, 3+4j]], dtype=np.complex64)
        
        with pytest.raises(TypeError, match="Unsupported dtype"):
            cnda.from_numpy(x, copy=False)


# ==============================================================================
# C++ → NumPy Tests (to_numpy)
# ==============================================================================

class TestToNumpyZeroCopy:
    """Test to_numpy with copy=False (zero-copy view from C++ to NumPy)."""
    
    def test_f32_zero_copy(self, np, cnda):
        """Test zero-copy export of float32 array to NumPy."""
        arr = cnda.ContiguousND_f32([2, 3])
        arr[0, 0] = 1.0
        arr[1, 2] = 99.5
        
        np_arr = arr.to_numpy(copy=False)
        
        # Verify values
        assert np_arr[0, 0] == 1.0
        assert np_arr[1, 2] == 99.5
        
        # Verify metadata
        assert np_arr.shape == (2, 3)
        assert np_arr.dtype == np.float32
        assert np_arr.flags['C_CONTIGUOUS']
        
        # Verify zero-copy (same memory address)
        assert arr.data_ptr() == np_arr.ctypes.data
    
    def test_f64_zero_copy(self, np, cnda):
        """Test zero-copy export of float64 array to NumPy."""
        arr = cnda.ContiguousND_f64([2, 2])
        arr[0, 0] = 1.5
        
        np_arr = arr.to_numpy(copy=False)
        assert np_arr[0, 0] == 1.5
        assert np_arr.dtype == np.float64
        assert arr.data_ptr() == np_arr.ctypes.data
    
    def test_i32_zero_copy(self, np, cnda):
        """Test zero-copy export of int32 array to NumPy."""
        arr = cnda.ContiguousND_i32([2, 2])
        arr[0, 0] = 42
        
        np_arr = arr.to_numpy(copy=False)
        assert np_arr[0, 0] == 42
        assert np_arr.dtype == np.int32
        assert arr.data_ptr() == np_arr.ctypes.data
    
    def test_i64_zero_copy(self, np, cnda):
        """Test zero-copy export of int64 array to NumPy."""
        arr = cnda.ContiguousND_i64([2, 2])
        arr[0, 0] = 9223372036854775807
        
        np_arr = arr.to_numpy(copy=False)
        assert np_arr[0, 0] == 9223372036854775807
        assert np_arr.dtype == np.int64
        assert arr.data_ptr() == np_arr.ctypes.data
    
    def test_memory_sharing(self, np, cnda):
        """Test that modifications in C++ are reflected in NumPy (zero-copy)."""
        arr = cnda.ContiguousND_f32([2, 2])
        arr[0, 0] = 1.0
        
        np_arr = arr.to_numpy(copy=False)
        
        # Modify C++ array
        arr[0, 0] = 99.0
        
        # Should be reflected in NumPy array (zero-copy)
        assert np_arr[0, 0] == 99.0
    
    def test_lifetime_management(self, np, cnda):
        """Test that NumPy array keeps C++ data alive via capsule."""
        arr = cnda.ContiguousND_f32([2, 2])
        arr[0, 0] = 42.0
        
        np_arr = arr.to_numpy(copy=False)
        data_ptr = np_arr.ctypes.data
        
        # Delete original C++ reference
        del arr
        
        # NumPy array should still be valid (capsule keeps data alive)
        assert np_arr[0, 0] == 42.0
        assert np_arr.ctypes.data == data_ptr


class TestToNumpyDeepCopy:
    """Test to_numpy with copy=True (deep copy from C++ to NumPy)."""
    
    def test_f32_deep_copy(self, np, cnda):
        """Test deep copy export of float32 array to NumPy."""
        arr = cnda.ContiguousND_f32([2, 3])
        arr[0, 0] = 1.0
        arr[1, 2] = 5.0
        
        np_arr = arr.to_numpy(copy=True)
        
        # Verify values
        assert np_arr[0, 0] == 1.0
        assert np_arr[1, 2] == 5.0
        
        # Verify it's a copy (different memory addresses)
        assert arr.data_ptr() != np_arr.ctypes.data
    
    def test_memory_independence(self, np, cnda):
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
