"""
Pytest test suite for CNDA Python bindings.
"""

import sys
import os
import pytest

# Add build directory to path
build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build', 'python')
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)


def test_import_cnda():
    """Test that cnda module can be imported."""
    import cnda
    assert cnda is not None


def test_version():
    """Test that __version__ attribute exists and is accessible."""
    import cnda
    assert hasattr(cnda, '__version__')
    assert isinstance(cnda.__version__, str)
    assert len(cnda.__version__) > 0
    print(f"CNDA version: {cnda.__version__}")


class TestContiguousND_f32:
    """Test suite for float (f32) dtype variant."""
    
    def test_construction(self):
        """Test that ContiguousND_f32 can be constructed."""
        import cnda
        arr = cnda.ContiguousND_f32([3, 4])
        assert arr is not None
    
    def test_shape(self):
        """Test that shape returns a tuple."""
        import cnda
        arr = cnda.ContiguousND_f32([3, 4])
        shape = arr.shape
        assert isinstance(shape, tuple)
        assert len(shape) == 2
        assert shape[0] == 3
        assert shape[1] == 4
    
    def test_strides(self):
        """Test that strides returns a tuple with correct row-major values."""
        import cnda
        arr = cnda.ContiguousND_f32([3, 4])
        strides = arr.strides
        assert isinstance(strides, tuple)
        assert len(strides) == 2
        # Row-major: [4, 1]
        assert strides[0] == 4
        assert strides[1] == 1
    
    def test_ndim(self):
        """Test ndim property."""
        import cnda
        arr = cnda.ContiguousND_f32([3, 4, 5])
        assert arr.ndim == 3
    
    def test_size(self):
        """Test size property."""
        import cnda
        arr = cnda.ContiguousND_f32([3, 4])
        assert arr.size == 12
    
    def test_indexing_2d(self):
        """Test indexing with __call__ for 2D array."""
        import cnda
        arr = cnda.ContiguousND_f32([3, 4])
        
        # Set some values using __call__
        arr(0, 0)  # Access element
        arr(1, 2)
        arr(2, 3)
    
    def test_getitem_setitem(self):
        """Test __getitem__ and __setitem__ with tuple indexing."""
        import cnda
        arr = cnda.ContiguousND_f32([3, 4])
        
        # Set values
        arr[0, 0] = 1.0
        arr[1, 2] = 42.5
        arr[2, 3] = 99.9
        
        # Get values
        assert arr[0, 0] == 1.0
        assert arr[1, 2] == 42.5
        assert arr[2, 3] == 99.9
    
    def test_1d_array(self):
        """Test 1D array operations."""
        import cnda
        arr = cnda.ContiguousND_f32([5])
        
        assert arr.ndim == 1
        assert arr.size == 5
        assert arr.shape == (5,)
        assert arr.strides == (1,)
        
        # Test indexing
        arr[0] = 10.0
        arr[4] = 40.0
        assert arr[0] == 10.0
        assert arr[4] == 40.0
    
    def test_3d_array(self):
        """Test 3D array operations."""
        import cnda
        arr = cnda.ContiguousND_f32([2, 3, 4])
        
        assert arr.ndim == 3
        assert arr.size == 24
        assert arr.shape == (2, 3, 4)
        # Row-major strides: [12, 4, 1]
        assert arr.strides == (12, 4, 1)
        
        # Test indexing
        arr[0, 0, 0] = 1.0
        arr[1, 2, 3] = 123.0
        assert arr[0, 0, 0] == 1.0
        assert arr[1, 2, 3] == 123.0


class TestContiguousND_f64:
    """Test suite for double (f64) dtype variant."""
    
    def test_construction(self):
        """Test that ContiguousND_f64 can be constructed."""
        import cnda
        arr = cnda.ContiguousND_f64([3, 4])
        assert arr is not None
    
    def test_shape_strides(self):
        """Test shape and strides for double."""
        import cnda
        arr = cnda.ContiguousND_f64([3, 4])
        assert arr.shape == (3, 4)
        assert arr.strides == (4, 1)
    
    def test_indexing(self):
        """Test indexing for double."""
        import cnda
        arr = cnda.ContiguousND_f64([2, 3])
        
        arr[0, 0] = 1.5
        arr[1, 2] = 99.99
        
        assert arr[0, 0] == 1.5
        assert arr[1, 2] == 99.99


class TestContiguousND_i32:
    """Test suite for int32_t (i32) dtype variant."""
    
    def test_construction(self):
        """Test that ContiguousND_i32 can be constructed."""
        import cnda
        arr = cnda.ContiguousND_i32([3, 4])
        assert arr is not None
    
    def test_shape_strides(self):
        """Test shape and strides for int32."""
        import cnda
        arr = cnda.ContiguousND_i32([3, 4])
        assert arr.shape == (3, 4)
        assert arr.strides == (4, 1)
    
    def test_indexing(self):
        """Test indexing for int32."""
        import cnda
        arr = cnda.ContiguousND_i32([2, 3])
        
        arr[0, 0] = 42
        arr[1, 2] = -100
        
        assert arr[0, 0] == 42
        assert arr[1, 2] == -100


class TestContiguousND_i64:
    """Test suite for int64_t (i64) dtype variant."""
    
    def test_construction(self):
        """Test that ContiguousND_i64 can be constructed."""
        import cnda
        arr = cnda.ContiguousND_i64([3, 4])
        assert arr is not None
    
    def test_shape_strides(self):
        """Test shape and strides for int64."""
        import cnda
        arr = cnda.ContiguousND_i64([3, 4])
        assert arr.shape == (3, 4)
        assert arr.strides == (4, 1)
    
    def test_indexing(self):
        """Test indexing for int64."""
        import cnda
        arr = cnda.ContiguousND_i64([2, 3])
        
        arr[0, 0] = 9223372036854775807  # Max int64
        arr[1, 2] = -9223372036854775808  # Min int64
        
        assert arr[0, 0] == 9223372036854775807
        assert arr[1, 2] == -9223372036854775808


class TestTypeAliases:
    """Test that type aliases work correctly."""
    
    def test_float_alias(self):
        """Test ContiguousND_float alias."""
        import cnda
        assert hasattr(cnda, 'ContiguousND_float')
        arr = cnda.ContiguousND_float([2, 2])
        assert arr is not None
    
    def test_double_alias(self):
        """Test ContiguousND_double alias."""
        import cnda
        assert hasattr(cnda, 'ContiguousND_double')
        arr = cnda.ContiguousND_double([2, 2])
        assert arr is not None
    
    def test_int32_alias(self):
        """Test ContiguousND_int32 alias."""
        import cnda
        assert hasattr(cnda, 'ContiguousND_int32')
        arr = cnda.ContiguousND_int32([2, 2])
        assert arr is not None
    
    def test_int64_alias(self):
        """Test ContiguousND_int64 alias."""
        import cnda
        assert hasattr(cnda, 'ContiguousND_int64')
        arr = cnda.ContiguousND_int64([2, 2])
        assert arr is not None


class TestBoundsChecking:
    """Test that out-of-bounds access raises appropriate errors."""
    
    def test_out_of_bounds(self):
        """Test that out-of-bounds indexing raises an exception."""
        import cnda
        arr = cnda.ContiguousND_f32([3, 4])
        
        with pytest.raises(Exception):  # Should raise out_of_range
            _ = arr[3, 0]  # Out of bounds on first dimension
        
        with pytest.raises(Exception):
            _ = arr[0, 4]  # Out of bounds on second dimension
    
    def test_wrong_ndim(self):
        """Test that wrong number of indices raises an exception."""
        import cnda
        arr = cnda.ContiguousND_f32([3, 4])
        
        with pytest.raises(Exception):  # Should raise out_of_range
            _ = arr[0]  # Too few indices
        
        with pytest.raises(Exception):
            _ = arr[0, 0, 0]  # Too many indices


class TestRepr:
    """Test string representation."""
    
    def test_repr(self):
        """Test __repr__ method."""
        import cnda
        arr = cnda.ContiguousND_f32([3, 4])
        repr_str = repr(arr)
        
        assert "ContiguousND_f32" in repr_str
        assert "shape=(3, 4)" in repr_str
        assert "size=12" in repr_str


class TestZeroSized:
    """Test zero-sized arrays."""
    
    def test_zero_dimension(self):
        """Test array with zero-sized dimension."""
        import cnda
        arr = cnda.ContiguousND_f32([0, 7])
        
        assert arr.size == 0
        assert arr.ndim == 2
        assert arr.shape == (0, 7)
        # Strides should still be computed correctly
        assert arr.strides == (7, 1)


if __name__ == '__main__':
    # Can run directly: python test_python_bindings.py
    pytest.main([__file__, '-v'])
