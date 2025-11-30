"""
Test suite for CNDA AoS copy semantics (copy=True).

This test suite covers deep copy semantics for to_numpy(copy=True) and from_numpy(copy=True).
"""

import pytest
import numpy as np


@pytest.fixture(scope="module")
def cnda():
    """Import cnda module."""
    import cnda as cnda_module
    return cnda_module


class TestDeepCopy:
    """Comprehensive tests for deep copy (copy=True) semantics."""
    
    def test_to_numpy_deep_copy_pointer_independence(self, cnda):
        """Test to_numpy(copy=True): pointers are different."""
        # Vec2f
        arr_vec2f = cnda.ContiguousND_Vec2f([5, 5])
        np_copy = arr_vec2f.to_numpy(copy=True)
        
        cpp_ptr = arr_vec2f.data_ptr()
        numpy_ptr = np_copy.__array_interface__['data'][0]
        
        # Deep copy: pointers must be different
        assert cpp_ptr != numpy_ptr
        
        # Vec3f
        arr_vec3f = cnda.ContiguousND_Vec3f([4, 4])
        np_copy = arr_vec3f.to_numpy(copy=True)
        
        cpp_ptr = arr_vec3f.data_ptr()
        numpy_ptr = np_copy.__array_interface__['data'][0]
        assert cpp_ptr != numpy_ptr
        
        # Cell2D
        arr_cell2d = cnda.ContiguousND_Cell2D([6, 6])
        np_copy = arr_cell2d.to_numpy(copy=True)
        
        cpp_ptr = arr_cell2d.data_ptr()
        numpy_ptr = np_copy.__array_interface__['data'][0]
        assert cpp_ptr != numpy_ptr
        
        # Cell3D
        arr_cell3d = cnda.ContiguousND_Cell3D([3, 3, 3])
        np_copy = arr_cell3d.to_numpy(copy=True)
        
        cpp_ptr = arr_cell3d.data_ptr()
        numpy_ptr = np_copy.__array_interface__['data'][0]
        assert cpp_ptr != numpy_ptr
    
    def test_to_numpy_deep_copy_value_consistency(self, cnda):
        """Test to_numpy(copy=True): initial values are identical."""
        # Vec2f: comprehensive value check
        arr = cnda.ContiguousND_Vec2f([4, 5])
        
        # Fill with test data
        for i in range(4):
            for j in range(5):
                arr.set_x((i, j), float(i * 10 + j))
                arr.set_y((i, j), float(i * 100 + j * 10))
        
        # Create deep copy
        np_copy = arr.to_numpy(copy=True)
        
        # All values must match exactly
        for i in range(4):
            for j in range(5):
                assert np_copy[i, j]['x'] == pytest.approx(float(i * 10 + j))
                assert np_copy[i, j]['y'] == pytest.approx(float(i * 100 + j * 10))
        
        # Cell3D: test all field types
        grid = cnda.ContiguousND_Cell3D([3, 4])
        
        for i in range(3):
            for j in range(4):
                grid.set_u((i, j), float(i))
                grid.set_v((i, j), float(j))
                grid.set_w((i, j), float(i + j))
                grid.set_flag((i, j), i * 10 + j)
        
        np_grid = grid.to_numpy(copy=True)
        
        for i in range(3):
            for j in range(4):
                assert np_grid[i, j]['u'] == pytest.approx(float(i))
                assert np_grid[i, j]['v'] == pytest.approx(float(j))
                assert np_grid[i, j]['w'] == pytest.approx(float(i + j))
                assert np_grid[i, j]['flag'] == i * 10 + j
    
    def test_to_numpy_deep_copy_cpp_modifications_isolated(self, cnda):
        """Test to_numpy(copy=True): C++ modifications don't affect NumPy copy."""
        arr = cnda.ContiguousND_Vec3f([3, 3])
        
        # Set initial values
        arr.set_x((1, 1), 10.0)
        arr.set_y((1, 1), 20.0)
        arr.set_z((1, 1), 30.0)
        
        # Create deep copy
        np_copy = arr.to_numpy(copy=True)
        
        # Verify initial values in copy
        assert np_copy[1, 1]['x'] == pytest.approx(10.0)
        assert np_copy[1, 1]['y'] == pytest.approx(20.0)
        assert np_copy[1, 1]['z'] == pytest.approx(30.0)
        
        # Modify C++ array extensively
        arr.set_x((1, 1), 100.0)
        arr.set_y((1, 1), 200.0)
        arr.set_z((1, 1), 300.0)
        arr.set_x((0, 0), 999.0)
        arr.set_y((2, 2), 888.0)
        
        # NumPy copy should remain unchanged
        assert np_copy[1, 1]['x'] == pytest.approx(10.0)
        assert np_copy[1, 1]['y'] == pytest.approx(20.0)
        assert np_copy[1, 1]['z'] == pytest.approx(30.0)
        assert np_copy[0, 0]['x'] == pytest.approx(0.0)  # Still default
        assert np_copy[2, 2]['y'] == pytest.approx(0.0)  # Still default
    
    def test_to_numpy_deep_copy_numpy_modifications_isolated(self, cnda):
        """Test to_numpy(copy=True): NumPy modifications don't affect C++."""
        arr = cnda.ContiguousND_Cell2D([4, 4])
        
        # Set initial values
        arr.set_u((2, 2), 5.5)
        arr.set_v((2, 2), 6.5)
        arr.set_flag((2, 2), 42)
        
        # Create deep copy
        np_copy = arr.to_numpy(copy=True)
        
        # Modify NumPy copy
        np_copy[2, 2]['u'] = 100.0
        np_copy[2, 2]['v'] = 200.0
        np_copy[2, 2]['flag'] = 999
        np_copy[0, 0]['u'] = 777.0
        np_copy[3, 3]['flag'] = 123
        
        # C++ array should remain unchanged
        assert arr.get_u((2, 2)) == pytest.approx(5.5)
        assert arr.get_v((2, 2)) == pytest.approx(6.5)
        assert arr.get_flag((2, 2)) == 42
        assert arr.get_u((0, 0)) == pytest.approx(0.0)  # Still default
        assert arr.get_flag((3, 3)) == 0  # Still default
    
    def test_from_numpy_deep_copy_pointer_independence(self, cnda):
        """Test from_numpy(copy=True): pointers are different."""
        # Vec2f
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        np_arr = np.zeros((6, 6), dtype=vec2f_dtype)
        
        cnda_arr = cnda.from_numpy_Vec2f(np_arr, copy=True)
        
        numpy_ptr = np_arr.__array_interface__['data'][0]
        cpp_ptr = cnda_arr.data_ptr()
        assert numpy_ptr != cpp_ptr
        
        # Vec3f
        vec3f_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        np_arr = np.zeros((5, 5), dtype=vec3f_dtype)
        
        cnda_arr = cnda.from_numpy_Vec3f(np_arr, copy=True)
        
        numpy_ptr = np_arr.__array_interface__['data'][0]
        cpp_ptr = cnda_arr.data_ptr()
        assert numpy_ptr != cpp_ptr
        
        # Cell2D
        cell2d_dtype = np.dtype([('u', np.float32), ('v', np.float32), ('flag', np.int32)])
        np_arr = np.zeros((7, 7), dtype=cell2d_dtype)
        
        cnda_arr = cnda.from_numpy_Cell2D(np_arr, copy=True)
        
        numpy_ptr = np_arr.__array_interface__['data'][0]
        cpp_ptr = cnda_arr.data_ptr()
        assert numpy_ptr != cpp_ptr
        
        # Cell3D
        cell3d_dtype = np.dtype([('u', np.float32), ('v', np.float32), 
                                  ('w', np.float32), ('flag', np.int32)])
        np_arr = np.zeros((4, 4, 4), dtype=cell3d_dtype)
        
        cnda_arr = cnda.from_numpy_Cell3D(np_arr, copy=True)
        
        numpy_ptr = np_arr.__array_interface__['data'][0]
        cpp_ptr = cnda_arr.data_ptr()
        assert numpy_ptr != cpp_ptr
    
    def test_from_numpy_deep_copy_value_consistency(self, cnda):
        """Test from_numpy(copy=True): initial values are identical."""
        # Vec2f
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        np_arr = np.zeros((5, 6), dtype=vec2f_dtype)
        
        for i in range(5):
            for j in range(6):
                np_arr[i, j]['x'] = float(i * 5 + j)
                np_arr[i, j]['y'] = float(i * 50 + j * 5)
        
        cnda_arr = cnda.from_numpy_Vec2f(np_arr, copy=True)
        
        # All values must match
        for i in range(5):
            for j in range(6):
                assert cnda_arr.get_x((i, j)) == pytest.approx(float(i * 5 + j))
                assert cnda_arr.get_y((i, j)) == pytest.approx(float(i * 50 + j * 5))
        
        # Cell3D with all field types
        cell3d_dtype = np.dtype([('u', np.float32), ('v', np.float32), 
                                  ('w', np.float32), ('flag', np.int32)])
        np_grid = np.zeros((3, 3), dtype=cell3d_dtype)
        
        for i in range(3):
            for j in range(3):
                np_grid[i, j]['u'] = float(i * 1.5)
                np_grid[i, j]['v'] = float(j * 2.5)
                np_grid[i, j]['w'] = float((i + j) * 0.5)
                np_grid[i, j]['flag'] = i * 3 + j
        
        cnda_grid = cnda.from_numpy_Cell3D(np_grid, copy=True)
        
        for i in range(3):
            for j in range(3):
                assert cnda_grid.get_u((i, j)) == pytest.approx(float(i * 1.5))
                assert cnda_grid.get_v((i, j)) == pytest.approx(float(j * 2.5))
                assert cnda_grid.get_w((i, j)) == pytest.approx(float((i + j) * 0.5))
                assert cnda_grid.get_flag((i, j)) == i * 3 + j
    
    def test_from_numpy_deep_copy_numpy_modifications_isolated(self, cnda):
        """Test from_numpy(copy=True): NumPy modifications don't affect C++."""
        vec3f_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        np_arr = np.zeros((4, 4), dtype=vec3f_dtype)
        
        # Set initial values
        np_arr[1, 1]['x'] = 10.0
        np_arr[1, 1]['y'] = 20.0
        np_arr[1, 1]['z'] = 30.0
        
        # Create deep copy
        cnda_arr = cnda.from_numpy_Vec3f(np_arr, copy=True)
        
        # Verify C++ got the values
        assert cnda_arr.get_x((1, 1)) == pytest.approx(10.0)
        assert cnda_arr.get_y((1, 1)) == pytest.approx(20.0)
        assert cnda_arr.get_z((1, 1)) == pytest.approx(30.0)
        
        # Modify NumPy array extensively
        np_arr[1, 1]['x'] = 1000.0
        np_arr[1, 1]['y'] = 2000.0
        np_arr[1, 1]['z'] = 3000.0
        np_arr[0, 0]['x'] = 999.0
        np_arr[2, 2]['y'] = 888.0
        
        # C++ should remain unchanged
        assert cnda_arr.get_x((1, 1)) == pytest.approx(10.0)
        assert cnda_arr.get_y((1, 1)) == pytest.approx(20.0)
        assert cnda_arr.get_z((1, 1)) == pytest.approx(30.0)
        assert cnda_arr.get_x((0, 0)) == pytest.approx(0.0)
        assert cnda_arr.get_y((2, 2)) == pytest.approx(0.0)
    
    def test_from_numpy_deep_copy_cpp_modifications_isolated(self, cnda):
        """Test from_numpy(copy=True): C++ modifications don't affect NumPy."""
        cell2d_dtype = np.dtype([('u', np.float32), ('v', np.float32), ('flag', np.int32)])
        np_arr = np.zeros((5, 5), dtype=cell2d_dtype)
        
        # Set initial values
        np_arr[2, 3]['u'] = 5.5
        np_arr[2, 3]['v'] = 6.5
        np_arr[2, 3]['flag'] = 42
        
        # Create deep copy
        cnda_arr = cnda.from_numpy_Cell2D(np_arr, copy=True)
        
        # Modify C++ array
        cnda_arr.set_u((2, 3), 100.0)
        cnda_arr.set_v((2, 3), 200.0)
        cnda_arr.set_flag((2, 3), 999)
        cnda_arr.set_u((0, 0), 777.0)
        cnda_arr.set_flag((4, 4), 123)
        
        # NumPy should remain unchanged
        assert np_arr[2, 3]['u'] == pytest.approx(5.5)
        assert np_arr[2, 3]['v'] == pytest.approx(6.5)
        assert np_arr[2, 3]['flag'] == 42
        assert np_arr[0, 0]['u'] == pytest.approx(0.0)
        assert np_arr[4, 4]['flag'] == 0
    
    def test_deep_copy_multiple_copies_independence(self, cnda):
        """Test that multiple deep copies are all independent."""
        arr = cnda.ContiguousND_Vec2f([3, 3])
        arr.set_x((1, 1), 10.0)
        arr.set_y((1, 1), 20.0)
        
        # Create multiple copies
        copy1 = arr.to_numpy(copy=True)
        copy2 = arr.to_numpy(copy=True)
        copy3 = arr.to_numpy(copy=True)
        
        # All pointers should be different
        ptr_cpp = arr.data_ptr()
        ptr1 = copy1.__array_interface__['data'][0]
        ptr2 = copy2.__array_interface__['data'][0]
        ptr3 = copy3.__array_interface__['data'][0]
        
        assert ptr_cpp != ptr1
        assert ptr_cpp != ptr2
        assert ptr_cpp != ptr3
        assert ptr1 != ptr2
        assert ptr2 != ptr3
        assert ptr1 != ptr3
        
        # All should have same initial values
        assert copy1[1, 1]['x'] == pytest.approx(10.0)
        assert copy2[1, 1]['x'] == pytest.approx(10.0)
        assert copy3[1, 1]['x'] == pytest.approx(10.0)
        
        # Modify each independently
        arr.set_x((1, 1), 100.0)
        copy1[1, 1]['x'] = 200.0
        copy2[1, 1]['x'] = 300.0
        copy3[1, 1]['x'] = 400.0
        
        # Each should retain its own value
        assert arr.get_x((1, 1)) == pytest.approx(100.0)
        assert copy1[1, 1]['x'] == pytest.approx(200.0)
        assert copy2[1, 1]['x'] == pytest.approx(300.0)
        assert copy3[1, 1]['x'] == pytest.approx(400.0)
    
    def test_deep_copy_large_array_performance(self, cnda):
        """Test deep copy with larger arrays (correctness, not performance)."""
        # Create large array
        arr = cnda.ContiguousND_Cell3D([50, 50])
        
        # Fill with pattern
        for i in range(0, 50, 5):
            for j in range(0, 50, 5):
                arr.set_u((i, j), float(i))
                arr.set_v((i, j), float(j))
                arr.set_w((i, j), float(i + j))
                arr.set_flag((i, j), i * 100 + j)
        
        # Create deep copy
        np_copy = arr.to_numpy(copy=True)
        
        # Verify pointers are different
        assert arr.data_ptr() != np_copy.__array_interface__['data'][0]
        
        # Verify values match
        for i in range(0, 50, 5):
            for j in range(0, 50, 5):
                assert np_copy[i, j]['u'] == pytest.approx(float(i))
                assert np_copy[i, j]['v'] == pytest.approx(float(j))
                assert np_copy[i, j]['w'] == pytest.approx(float(i + j))
                assert np_copy[i, j]['flag'] == i * 100 + j
        
        # Modify original
        arr.set_u((10, 10), 9999.0)
        
        # Copy should be unaffected
        assert np_copy[10, 10]['u'] == pytest.approx(10.0)
    
    def test_roundtrip_numpy_cnda(self, cnda):
        """Test round-trip: NumPy -> CNDA -> NumPy."""
        # Create structured array
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        np_orig = np.zeros((5, 5), dtype=vec2f_dtype)
        
        # Fill with data
        for i in range(5):
            for j in range(5):
                np_orig[i, j]['x'] = float(i * 10 + j)
                np_orig[i, j]['y'] = float(i * 10 + j + 100)
        
        # Convert to CNDA
        cnda_arr = cnda.from_numpy_Vec2f(np_orig, copy=True)
        
        # Convert back to NumPy
        np_new = cnda_arr.to_numpy(copy=True)
        
        # Verify data preserved
        assert np_new.shape == np_orig.shape
        assert np.allclose(np_new['x'], np_orig['x'])
        assert np.allclose(np_new['y'], np_orig['y'])


