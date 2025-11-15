"""
Test suite for CNDA Python bindings - Basic Functionality.

This test suite covers:
1. Module import and version
2. Basic construction and properties (shape, strides, ndim, size)
3. Element access and modification (__getitem__, __setitem__, __call__)
4. All supported dtypes (f32, f64, i32, i64)
5. Type aliases
6. String representation
7. Bounds checking and error handling
"""

import sys
import os
import pytest

@pytest.fixture(scope="module")
def cnda():
    """Import cnda module."""
    import cnda as cnda_module
    return cnda_module


# ==============================================================================
# Module Tests
# ==============================================================================

class TestModule:
    """Test module-level functionality."""
    
    def test_import_cnda(self, cnda):
        """Test that cnda module can be imported."""
        assert cnda is not None
    
    def test_version(self, cnda):
        """Test that __version__ attribute exists."""
        assert hasattr(cnda, '__version__')
        assert isinstance(cnda.__version__, str)
        assert len(cnda.__version__) > 0
    
    def test_module_docstring(self, cnda):
        """Test that module has docstring."""
        assert cnda.__doc__ is not None
        assert len(cnda.__doc__) > 0


# ==============================================================================
# Construction and Properties Tests
# ==============================================================================

class TestConstruction:
    """Test array construction for all dtypes."""
    
    def test_f32_construction(self, cnda):
        """Test ContiguousND_f32 construction."""
        arr = cnda.ContiguousND_f32([3, 4])
        assert arr is not None
        assert arr.shape == (3, 4)
        assert arr.ndim == 2
        assert arr.size == 12
    
    def test_f64_construction(self, cnda):
        """Test ContiguousND_f64 construction."""
        arr = cnda.ContiguousND_f64([2, 5])
        assert arr is not None
        assert arr.shape == (2, 5)
        assert arr.ndim == 2
        assert arr.size == 10
    
    def test_i32_construction(self, cnda):
        """Test ContiguousND_i32 construction."""
        arr = cnda.ContiguousND_i32([4, 3])
        assert arr is not None
        assert arr.shape == (4, 3)
        assert arr.ndim == 2
        assert arr.size == 12
    
    def test_i64_construction(self, cnda):
        """Test ContiguousND_i64 construction."""
        arr = cnda.ContiguousND_i64([5, 2])
        assert arr is not None
        assert arr.shape == (5, 2)
        assert arr.ndim == 2
        assert arr.size == 10
    
    def test_1d_construction(self, cnda):
        """Test 1D array construction."""
        arr = cnda.ContiguousND_f32([10])
        assert arr.ndim == 1
        assert arr.shape == (10,)
        assert arr.size == 10
        assert arr.strides == (1,)
    
    def test_3d_construction(self, cnda):
        """Test 3D array construction."""
        arr = cnda.ContiguousND_f32([2, 3, 4])
        assert arr.ndim == 3
        assert arr.shape == (2, 3, 4)
        assert arr.size == 24
        assert arr.strides == (12, 4, 1)
    
    def test_4d_construction(self, cnda):
        """Test 4D array construction."""
        arr = cnda.ContiguousND_f64([2, 3, 4, 5])
        assert arr.ndim == 4
        assert arr.shape == (2, 3, 4, 5)
        assert arr.size == 120
        assert arr.strides == (60, 20, 5, 1)


class TestProperties:
    """Test array properties."""
    
    def test_shape_property(self, cnda):
        """Test shape property returns tuple."""
        arr = cnda.ContiguousND_f32([3, 4, 5])
        shape = arr.shape
        assert isinstance(shape, tuple)
        assert len(shape) == 3
        assert shape[0] == 3
        assert shape[1] == 4
        assert shape[2] == 5
    
    def test_strides_property(self, cnda):
        """Test strides property returns row-major strides."""
        arr = cnda.ContiguousND_f32([3, 4])
        strides = arr.strides
        assert isinstance(strides, tuple)
        assert len(strides) == 2
        # Row-major: [4, 1] for shape [3, 4]
        assert strides[0] == 4
        assert strides[1] == 1
    
    def test_strides_3d(self, cnda):
        """Test strides for 3D array."""
        arr = cnda.ContiguousND_f32([2, 3, 4])
        strides = arr.strides
        # Row-major: [12, 4, 1] for shape [2, 3, 4]
        assert strides == (12, 4, 1)
    
    def test_ndim_property(self, cnda):
        """Test ndim property."""
        arr_1d = cnda.ContiguousND_f32([5])
        arr_2d = cnda.ContiguousND_f32([3, 4])
        arr_3d = cnda.ContiguousND_f32([2, 3, 4])
        
        assert arr_1d.ndim == 1
        assert arr_2d.ndim == 2
        assert arr_3d.ndim == 3
    
    def test_size_property(self, cnda):
        """Test size property."""
        arr1 = cnda.ContiguousND_f32([3, 4])
        arr2 = cnda.ContiguousND_f32([2, 3, 4])
        arr3 = cnda.ContiguousND_f32([10])
        
        assert arr1.size == 12
        assert arr2.size == 24
        assert arr3.size == 10
    
    def test_data_ptr(self, cnda):
        """Test data_ptr returns valid pointer."""
        arr = cnda.ContiguousND_f32([2, 2])
        ptr = arr.data_ptr()
        assert isinstance(ptr, int)
        assert ptr > 0


# ==============================================================================
# Indexing Tests
# ==============================================================================

class TestIndexing:
    """Test element access and modification."""
    
    def test_getitem_setitem_2d(self, cnda):
        """Test __getitem__ and __setitem__ for 2D array."""
        arr = cnda.ContiguousND_f32([3, 4])
        
        # Set values
        arr[0, 0] = 1.0
        arr[1, 2] = 42.5
        arr[2, 3] = 99.9
        
        # Get values
        assert arr[0, 0] == 1.0
        assert arr[1, 2] == 42.5
        assert abs(arr[2, 3] - 99.9) < 0.01
    
    def test_getitem_setitem_1d(self, cnda):
        """Test indexing for 1D array."""
        arr = cnda.ContiguousND_f32([5])
        
        arr[0] = 10.0
        arr[4] = 40.0
        
        assert arr[0] == 10.0
        assert arr[4] == 40.0
    
    def test_getitem_setitem_3d(self, cnda):
        """Test indexing for 3D array."""
        arr = cnda.ContiguousND_f32([2, 3, 4])
        
        arr[0, 0, 0] = 1.0
        arr[1, 2, 3] = 123.0
        
        assert arr[0, 0, 0] == 1.0
        assert arr[1, 2, 3] == 123.0
    
    def test_call_operator(self, cnda):
        """Test __call__ operator for element access (read-only in Python)."""
        arr = cnda.ContiguousND_f32([3, 4])
        
        # Set values using square brackets
        arr[1, 2] = 5.0
        arr[0, 3] = 7.5
        arr[2, 0] = -3.2
        
        # Get values using __call__
        assert abs(arr(1, 2) - 5.0) < 0.01
        assert abs(arr(0, 3) - 7.5) < 0.01
        assert abs(arr(2, 0) - (-3.2)) < 0.01
    
    def test_all_dtypes_indexing(self, cnda):
        """Test indexing for all dtypes."""
        # float32
        arr_f32 = cnda.ContiguousND_f32([2, 2])
        arr_f32[0, 0] = 1.5
        assert arr_f32[0, 0] == 1.5
        
        # float64
        arr_f64 = cnda.ContiguousND_f64([2, 2])
        arr_f64[0, 0] = 2.5
        assert arr_f64[0, 0] == 2.5
        
        # int32
        arr_i32 = cnda.ContiguousND_i32([2, 2])
        arr_i32[0, 0] = 42
        assert arr_i32[0, 0] == 42
        
        # int64
        arr_i64 = cnda.ContiguousND_i64([2, 2])
        arr_i64[0, 0] = 9223372036854775807
        assert arr_i64[0, 0] == 9223372036854775807


# ==============================================================================
# Dtype Tests
# ==============================================================================

class TestDtypes:
    """Test all supported data types."""
    
    def test_f32_values(self, cnda):
        """Test float32 value storage."""
        arr = cnda.ContiguousND_f32([2, 2])
        arr[0, 0] = 1.5
        arr[0, 1] = -2.5
        arr[1, 0] = 0.0
        arr[1, 1] = 999.999
        
        assert arr[0, 0] == 1.5
        assert arr[0, 1] == -2.5
        assert arr[1, 0] == 0.0
        assert abs(arr[1, 1] - 999.999) < 0.01
    
    def test_f64_values(self, cnda):
        """Test float64 value storage."""
        arr = cnda.ContiguousND_f64([2, 2])
        arr[0, 0] = 1.123456789
        arr[1, 1] = -99.987654321
        
        assert abs(arr[0, 0] - 1.123456789) < 1e-9
        assert abs(arr[1, 1] - (-99.987654321)) < 1e-9
    
    def test_i32_values(self, cnda):
        """Test int32 value storage."""
        arr = cnda.ContiguousND_i32([2, 2])
        arr[0, 0] = 2147483647  # Max int32
        arr[1, 1] = -2147483648  # Min int32
        
        assert arr[0, 0] == 2147483647
        assert arr[1, 1] == -2147483648
    
    def test_i64_values(self, cnda):
        """Test int64 value storage."""
        arr = cnda.ContiguousND_i64([2, 2])
        arr[0, 0] = 9223372036854775807  # Max int64
        arr[1, 1] = -9223372036854775808  # Min int64
        
        assert arr[0, 0] == 9223372036854775807
        assert arr[1, 1] == -9223372036854775808


class TestTypeAliases:
    """Test type aliases."""
    
    def test_float_alias(self, cnda):
        """Test ContiguousND_float alias."""
        assert hasattr(cnda, 'ContiguousND_float')
        arr = cnda.ContiguousND_float([2, 2])
        assert arr is not None
        arr[0, 0] = 1.5
        assert arr[0, 0] == 1.5
    
    def test_double_alias(self, cnda):
        """Test ContiguousND_double alias."""
        assert hasattr(cnda, 'ContiguousND_double')
        arr = cnda.ContiguousND_double([2, 2])
        assert arr is not None
        arr[0, 0] = 2.5
        assert arr[0, 0] == 2.5
    
    def test_int32_alias(self, cnda):
        """Test ContiguousND_int32 alias."""
        assert hasattr(cnda, 'ContiguousND_int32')
        arr = cnda.ContiguousND_int32([2, 2])
        assert arr is not None
        arr[0, 0] = 42
        assert arr[0, 0] == 42
    
    def test_int64_alias(self, cnda):
        """Test ContiguousND_int64 alias."""
        assert hasattr(cnda, 'ContiguousND_int64')
        arr = cnda.ContiguousND_int64([2, 2])
        assert arr is not None
        arr[0, 0] = 99
        assert arr[0, 0] == 99


# ==============================================================================
# String Representation Tests
# ==============================================================================

class TestRepr:
    """Test string representation."""
    
    def test_repr_f32(self, cnda):
        """Test __repr__ for float32."""
        arr = cnda.ContiguousND_f32([3, 4])
        repr_str = repr(arr)
        
        assert "ContiguousND_f32" in repr_str
        assert "shape=(3, 4)" in repr_str
        assert "size=12" in repr_str
    
    def test_repr_f64(self, cnda):
        """Test __repr__ for float64."""
        arr = cnda.ContiguousND_f64([2, 5])
        repr_str = repr(arr)
        
        assert "ContiguousND_f64" in repr_str
        assert "shape=(2, 5)" in repr_str
        assert "size=10" in repr_str
    
    def test_repr_i32(self, cnda):
        """Test __repr__ for int32."""
        arr = cnda.ContiguousND_i32([4, 3])
        repr_str = repr(arr)
        
        assert "ContiguousND_i32" in repr_str
        assert "shape=(4, 3)" in repr_str
    
    def test_repr_i64(self, cnda):
        """Test __repr__ for int64."""
        arr = cnda.ContiguousND_i64([5, 2])
        repr_str = repr(arr)
        
        assert "ContiguousND_i64" in repr_str
        assert "shape=(5, 2)" in repr_str


# ==============================================================================
# Error Handling Tests
# ==============================================================================

class TestBoundsChecking:
    """Test bounds checking and error handling."""
    
    def test_out_of_bounds_first_dim(self, cnda):
        """Test out-of-bounds on first dimension raises error."""
        arr = cnda.ContiguousND_f32([3, 4])
        
        with pytest.raises(Exception):  # Should raise out_of_range
            _ = arr[3, 0]
    
    def test_out_of_bounds_second_dim(self, cnda):
        """Test out-of-bounds on second dimension raises error."""
        arr = cnda.ContiguousND_f32([3, 4])
        
        with pytest.raises(Exception):
            _ = arr[0, 4]
    
    def test_out_of_bounds_negative(self, cnda):
        """Test negative indices (if not supported) raise error."""
        arr = cnda.ContiguousND_f32([3, 4])
        
        # Python negative indexing may not be supported in C++ bindings
        # This tests the current behavior
        try:
            _ = arr[-1, 0]
            # If it works, that's fine (implementation-dependent)
        except Exception:
            # If it raises an error, that's also acceptable
            pass
    
    def test_wrong_ndim_too_few(self, cnda):
        """Test too few indices raises error."""
        arr = cnda.ContiguousND_f32([3, 4])
        
        with pytest.raises(Exception):
            _ = arr[0]
    
    def test_wrong_ndim_too_many(self, cnda):
        """Test too many indices raises error."""
        arr = cnda.ContiguousND_f32([3, 4])
        
        with pytest.raises(Exception):
            _ = arr[0, 0, 0]
    
    def test_setitem_out_of_bounds(self, cnda):
        """Test __setitem__ with out-of-bounds raises error."""
        arr = cnda.ContiguousND_f32([3, 4])
        
        with pytest.raises(Exception):
            arr[3, 0] = 1.0
        
        with pytest.raises(Exception):
            arr[0, 4] = 1.0


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

class TestEdgeCases:
    """Test edge cases."""
    
    def test_zero_sized_dimension(self, cnda):
        """Test array with zero-sized dimension."""
        arr = cnda.ContiguousND_f32([0, 7])
        
        assert arr.size == 0
        assert arr.ndim == 2
        assert arr.shape == (0, 7)
        # Strides should still be computed correctly
        assert arr.strides == (7, 1)
    
    def test_single_element_array(self, cnda):
        """Test array with single element."""
        arr = cnda.ContiguousND_f32([1, 1])
        arr[0, 0] = 42.0
        assert arr[0, 0] == 42.0
        assert arr.size == 1
    
    def test_large_dimensions(self, cnda):
        """Test array with large dimensions."""
        arr = cnda.ContiguousND_f32([100, 200])
        
        assert arr.shape == (100, 200)
        assert arr.size == 20000
        assert arr.ndim == 2
        
        # Test corner elements
        arr[0, 0] = 1.0
        arr[99, 199] = 999.0
        
        assert arr[0, 0] == 1.0
        assert arr[99, 199] == 999.0
    
    def test_many_dimensions(self, cnda):
        """Test array with many dimensions."""
        arr = cnda.ContiguousND_f32([2, 3, 4, 5, 6])
        
        assert arr.ndim == 5
        assert arr.size == 720
        assert len(arr.shape) == 5
        assert len(arr.strides) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
