"""
Test suite for CNDA AoS error handling.

This test suite covers error handling for invalid inputs, dtype mismatches, and layout issues.
"""

import pytest
import numpy as np


@pytest.fixture(scope="module")
def cnda():
    """Import cnda module."""
    import cnda as cnda_module
    return cnda_module


class TestErrorHandling:
    """Test error handling for AoS operations."""
    
    def test_invalid_shape(self, cnda):
        """Test that invalid shapes raise ValueError."""
        with pytest.raises(ValueError):
            arr = cnda.ContiguousND_Vec2f([])
    
    def test_from_numpy_dtype_mismatch_field_count(self, cnda):
        """Test from_numpy: TypeError when field count mismatches."""
        # Vec2f expects 2 fields, but provide 3
        wrong_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        np_arr = np.zeros((3, 3), dtype=wrong_dtype)
        
        with pytest.raises(TypeError) as excinfo:
            cnda.from_numpy_Vec2f(np_arr, copy=False)
        assert "exactly 2 fields" in str(excinfo.value)
        
        # Cell2D expects 3 fields, but provide 2
        wrong_dtype = np.dtype([('u', np.float32), ('v', np.float32)])
        np_arr = np.zeros((3, 3), dtype=wrong_dtype)
        
        with pytest.raises(TypeError) as excinfo:
            cnda.from_numpy_Cell2D(np_arr, copy=False)
        assert "exactly 3 fields" in str(excinfo.value)
        
        # Cell3D expects 4 fields, but provide 3
        wrong_dtype = np.dtype([('u', np.float32), ('v', np.float32), ('w', np.float32)])
        np_arr = np.zeros((3, 3), dtype=wrong_dtype)
        
        with pytest.raises(TypeError) as excinfo:
            cnda.from_numpy_Cell3D(np_arr, copy=False)
        assert "exactly 4 fields" in str(excinfo.value)
    
    def test_from_numpy_dtype_mismatch_field_type(self, cnda):
        """Test from_numpy: TypeError when field type is wrong."""
        # Vec2f expects float32, but provide float64
        wrong_dtype = np.dtype([('x', np.float64), ('y', np.float64)])
        np_arr = np.zeros((3, 3), dtype=wrong_dtype)
        
        with pytest.raises(TypeError) as excinfo:
            cnda.from_numpy_Vec2f(np_arr, copy=False)
        assert "float32" in str(excinfo.value)
        
        # Cell2D expects (float32, float32, int32), but provide all float32
        wrong_dtype = np.dtype([('u', np.float32), ('v', np.float32), ('flag', np.float32)])
        np_arr = np.zeros((3, 3), dtype=wrong_dtype)
        
        with pytest.raises(TypeError) as excinfo:
            cnda.from_numpy_Cell2D(np_arr, copy=False)
        assert "int32" in str(excinfo.value) or "flag" in str(excinfo.value)
        
        # Cell3D expects int32 for flag, but provide int64
        wrong_dtype = np.dtype([('u', np.float32), ('v', np.float32), 
                                 ('w', np.float32), ('flag', np.int64)])
        np_arr = np.zeros((3, 3), dtype=wrong_dtype)
        
        with pytest.raises(TypeError) as excinfo:
            cnda.from_numpy_Cell3D(np_arr, copy=False)
        assert "int32" in str(excinfo.value) or "flag" in str(excinfo.value)
    
    def test_from_numpy_dtype_mismatch_not_structured(self, cnda):
        """Test from_numpy: TypeError when dtype is not structured."""
        # Plain float32 array (not structured)
        np_arr = np.zeros((3, 3), dtype=np.float32)
        
        with pytest.raises(TypeError) as excinfo:
            cnda.from_numpy_Vec2f(np_arr, copy=False)
        assert "structured dtype" in str(excinfo.value) or "plain array" in str(excinfo.value)
        
        # Plain int32 array
        np_arr = np.zeros((3, 3), dtype=np.int32)
        
        with pytest.raises(TypeError) as excinfo:
            cnda.from_numpy_Cell2D(np_arr, copy=False)
        assert "structured dtype" in str(excinfo.value) or "plain array" in str(excinfo.value)
    
    def test_from_numpy_dtype_mismatch_wrong_field_names(self, cnda):
        """Test from_numpy: TypeError when field names are wrong."""
        # Vec2f expects ['x', 'y'], but provide ['a', 'b']
        wrong_dtype = np.dtype([('a', np.float32), ('b', np.float32)])
        np_arr = np.zeros((3, 3), dtype=wrong_dtype)
        
        with pytest.raises(TypeError) as excinfo:
            cnda.from_numpy_Vec2f(np_arr, copy=False)
        assert "field" in str(excinfo.value).lower() and ("'x'" in str(excinfo.value) or "'a'" in str(excinfo.value))
        
        # Cell2D expects ['u', 'v', 'flag'], but provide ['x', 'y', 'z']
        wrong_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.int32)])
        np_arr = np.zeros((3, 3), dtype=wrong_dtype)
        
        with pytest.raises(TypeError) as excinfo:
            cnda.from_numpy_Cell2D(np_arr, copy=False)
        assert "field" in str(excinfo.value).lower()
    
    def test_from_numpy_non_contiguous_layout(self, cnda):
        """Test from_numpy: ValueError for non-contiguous layout."""
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        
        # Create a contiguous array first
        np_arr = np.zeros((10, 10), dtype=vec2f_dtype)
        
        # Create a non-contiguous view using slicing with step
        np_view = np_arr[::2, ::2]  # Every other element
        
        # Non-contiguous arrays should raise ValueError for zero-copy
        assert not np_view.flags['C_CONTIGUOUS']
        
        with pytest.raises(ValueError) as excinfo:
            cnda.from_numpy_Vec2f(np_view, copy=False)
        assert "contiguous" in str(excinfo.value).lower()
    
    def test_from_numpy_fortran_order_layout(self, cnda):
        """Test from_numpy: ValueError for Fortran-order (column-major) layout."""
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        
        # Create Fortran-order array (column-major)
        np_arr = np.asfortranarray(np.zeros((5, 5), dtype=vec2f_dtype))
        
        assert np_arr.flags['F_CONTIGUOUS']
        assert not np_arr.flags['C_CONTIGUOUS']
        
        # Should raise ValueError because CNDA expects C-contiguous (row-major)
        with pytest.raises(ValueError) as excinfo:
            cnda.from_numpy_Vec2f(np_arr, copy=False)
        assert "contiguous" in str(excinfo.value).lower() or "row-major" in str(excinfo.value).lower()
    
    def test_from_numpy_transposed_layout(self, cnda):
        """Test from_numpy: ValueError for transposed (non-C-contiguous) layout."""
        vec3f_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        
        # Create C-contiguous array
        np_arr = np.zeros((4, 6), dtype=vec3f_dtype)
        assert np_arr.flags['C_CONTIGUOUS']
        
        # Transpose changes the layout
        np_transposed = np_arr.T
        assert not np_transposed.flags['C_CONTIGUOUS']
        
        # Should raise ValueError for non-C-contiguous layout
        with pytest.raises(ValueError) as excinfo:
            cnda.from_numpy_Vec3f(np_transposed, copy=False)
        assert "contiguous" in str(excinfo.value).lower()
    
    def test_from_numpy_wrong_strides(self, cnda):
        """Test from_numpy: ValueError when strides don't match expected layout."""
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        
        # Create array with custom strides (using as_strided - advanced technique)
        np_arr = np.zeros((4, 5), dtype=vec2f_dtype)
        
        # Create a view with incorrect strides
        # Normal strides for (4, 5) with itemsize=8: (40, 8)
        # Create wrong strides: (48, 8) - skip one element per row
        try:
            from numpy.lib.stride_tricks import as_strided
            np_wrong_strides = as_strided(np_arr, shape=(3, 5), strides=(48, 8))
            
            # This should fail with ValueError because strides don't match row-major layout
            with pytest.raises(ValueError) as excinfo:
                cnda.from_numpy_Vec2f(np_wrong_strides, copy=False)
            # Error message may mention "stride" or "contiguous" depending on implementation
            assert "stride" in str(excinfo.value).lower() or "contiguous" in str(excinfo.value).lower()
        except ImportError:
            pytest.skip("numpy.lib.stride_tricks not available")
    
    def test_from_numpy_empty_array(self, cnda):
        """Test from_numpy handles empty arrays correctly."""
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        
        # Empty array (0 elements) has non-standard strides, requires copy=True
        np_arr = np.zeros((0,), dtype=vec2f_dtype)
        result = cnda.from_numpy_Vec2f(np_arr, copy=True)
        assert result.size == 0
    
    def test_from_numpy_misaligned_data(self, cnda):
        """Test from_numpy handles misaligned data (implementation may be permissive)."""
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        byte_data = np.zeros(100, dtype=np.uint8)
        
        try:
            # Create a view starting at byte 1 (misaligned for float32)
            np_misaligned = np.ndarray(
                shape=(2, 3),
                dtype=vec2f_dtype,
                buffer=byte_data[1:]
            )
        except (ValueError, TypeError):
            # NumPy itself might reject creating the misaligned view
            pytest.skip("Cannot create misaligned NumPy array on this platform")
        
        # Implementation is currently permissive with alignment
        # This is acceptable behavior
        result = cnda.from_numpy_Vec2f(np_misaligned, copy=False)
        assert result.size == 6
    
    def test_from_numpy_readonly_array(self, cnda):
        """Test from_numpy(copy=False) accepts read-only NumPy arrays."""
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        np_arr = np.zeros((3, 3), dtype=vec2f_dtype)
        
        # Make array read-only
        np_arr.flags.writeable = False
        
        # Implementation currently accepts read-only arrays for zero-copy
        cnda_arr = cnda.from_numpy_Vec2f(np_arr, copy=False)
        
        # Modification through C++ interface currently works
        # (NumPy's readonly flag is not enforced by pybind11)
        cnda_arr.set_x((0, 0), 1.0)
        assert cnda_arr.get_x((0, 0)) == pytest.approx(1.0)
    
    def test_invalid_indices(self, cnda):
        """Test that invalid field access raises IndexError for out-of-bounds."""
        arr = cnda.ContiguousND_Vec2f([2, 2])
        
        # These should work
        arr.set_x((0, 0), 1.0)
        
        # Out of bounds should raise IndexError
        with pytest.raises(IndexError):
            arr.get_x((5, 5))
        
        # Wrong number of indices - IndexError (rank mismatch is bounds check failure)
        with pytest.raises(IndexError):
            arr.get_x((0,))  # 1D index for 2D array
        
        with pytest.raises(IndexError):
            arr.get_x((0, 0, 0))  # 3D index for 2D array
    
    def test_invalid_field_values(self, cnda):
        """Test that invalid field values raise TypeError."""
        arr = cnda.ContiguousND_Vec2f([2, 2])
        
        # These should work (normal float values)
        arr.set_x((0, 0), 1.5)
        arr.set_x((0, 0), float('inf'))
        arr.set_x((0, 0), float('nan'))
        
        # Type errors (non-numeric types) should raise TypeError
        with pytest.raises(TypeError):
            arr.set_x((0, 0), "invalid")
        
        with pytest.raises(TypeError):
            arr.set_x((0, 0), None)
        
        # For integer fields (Cell2D flag)
        grid = cnda.ContiguousND_Cell2D([2, 2])
        
        # Valid integer
        grid.set_flag((0, 0), 42)
        
        # Float to int field should raise TypeError (pybind11 doesn't auto-convert)
        with pytest.raises(TypeError):
            grid.set_flag((0, 0), 3.7)


