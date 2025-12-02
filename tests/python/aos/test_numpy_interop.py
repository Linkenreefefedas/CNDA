"""
Test suite for CNDA AoS NumPy interoperability.

This test suite covers:
1. Memory layout validation (dtype, shape, strides)
2. Zero-copy bidirectional sync (C++ ↔ NumPy)
3. from_numpy() compatibility and integration tests
"""

import pytest
import numpy as np

@pytest.fixture(scope="module")
def cnda():
    """Import cnda module."""
    import cnda as cnda_module
    return cnda_module


# ==============================================================================
# Memory Layout and Performance Tests
# ==============================================================================

class TestMemoryLayout:
    """Test memory layout properties of AoS arrays."""
    
    def test_aos_to_numpy_dtype_correctness(self, cnda):
        """Test AoS → NumPy: dtype correctness (field names, types, order)."""
        # Test Vec2f
        arr_vec2f = cnda.ContiguousND_Vec2f([3, 4])
        arr_vec2f.set_x((1, 2), 10.5)
        arr_vec2f.set_y((1, 2), 20.5)
        
        np_vec2f = arr_vec2f.to_numpy(copy=False)
        
        # Check dtype is structured
        assert np_vec2f.dtype.names is not None
        assert len(np_vec2f.dtype.names) == 2
        
        # Check field names and order
        assert np_vec2f.dtype.names == ('x', 'y')
        
        # Check field types
        assert np_vec2f.dtype['x'] == np.float32
        assert np_vec2f.dtype['y'] == np.float32
        
        # Check values are correct
        assert np_vec2f[1, 2]['x'] == pytest.approx(10.5)
        assert np_vec2f[1, 2]['y'] == pytest.approx(20.5)
        
        # Test Vec3f
        arr_vec3f = cnda.ContiguousND_Vec3f([2, 3])
        arr_vec3f.set_x((0, 1), 1.0)
        arr_vec3f.set_y((0, 1), 2.0)
        arr_vec3f.set_z((0, 1), 3.0)
        
        np_vec3f = arr_vec3f.to_numpy(copy=False)
        
        assert np_vec3f.dtype.names == ('x', 'y', 'z')
        assert np_vec3f.dtype['x'] == np.float32
        assert np_vec3f.dtype['y'] == np.float32
        assert np_vec3f.dtype['z'] == np.float32
        assert np_vec3f[0, 1]['x'] == pytest.approx(1.0)
        assert np_vec3f[0, 1]['y'] == pytest.approx(2.0)
        assert np_vec3f[0, 1]['z'] == pytest.approx(3.0)
        
        # Test Cell2D
        arr_cell2d = cnda.ContiguousND_Cell2D([2, 2])
        arr_cell2d.set_u((1, 0), 5.5)
        arr_cell2d.set_v((1, 0), 6.5)
        arr_cell2d.set_flag((1, 0), 42)
        
        np_cell2d = arr_cell2d.to_numpy(copy=False)
        
        assert np_cell2d.dtype.names == ('u', 'v', 'flag')
        assert np_cell2d.dtype['u'] == np.float32
        assert np_cell2d.dtype['v'] == np.float32
        assert np_cell2d.dtype['flag'] == np.int32
        assert np_cell2d[1, 0]['u'] == pytest.approx(5.5)
        assert np_cell2d[1, 0]['v'] == pytest.approx(6.5)
        assert np_cell2d[1, 0]['flag'] == 42
        
        # Test Cell3D
        arr_cell3d = cnda.ContiguousND_Cell3D([2, 2, 2])
        arr_cell3d.set_u((1, 1, 1), 1.0)
        arr_cell3d.set_v((1, 1, 1), 2.0)
        arr_cell3d.set_w((1, 1, 1), 3.0)
        arr_cell3d.set_flag((1, 1, 1), 99)
        
        np_cell3d = arr_cell3d.to_numpy(copy=False)
        
        assert np_cell3d.dtype.names == ('u', 'v', 'w', 'flag')
        assert np_cell3d.dtype['u'] == np.float32
        assert np_cell3d.dtype['v'] == np.float32
        assert np_cell3d.dtype['w'] == np.float32
        assert np_cell3d.dtype['flag'] == np.int32
        assert np_cell3d[1, 1, 1]['u'] == pytest.approx(1.0)
        assert np_cell3d[1, 1, 1]['v'] == pytest.approx(2.0)
        assert np_cell3d[1, 1, 1]['w'] == pytest.approx(3.0)
        assert np_cell3d[1, 1, 1]['flag'] == 99
    
    def test_aos_to_numpy_shape_consistency(self, cnda):
        """Test AoS → NumPy: shape consistency across dimensions."""
        # 1D array
        arr_1d = cnda.ContiguousND_Vec2f([10])
        np_1d = arr_1d.to_numpy(copy=False)
        assert arr_1d.shape == (10,)
        assert np_1d.shape == (10,)
        assert arr_1d.ndim == 1
        assert np_1d.ndim == 1
        
        # 2D array
        arr_2d = cnda.ContiguousND_Vec3f([3, 4])
        np_2d = arr_2d.to_numpy(copy=False)
        assert arr_2d.shape == (3, 4)
        assert np_2d.shape == (3, 4)
        assert arr_2d.ndim == 2
        assert np_2d.ndim == 2
        
        # 3D array
        arr_3d = cnda.ContiguousND_Cell2D([2, 3, 4])
        np_3d = arr_3d.to_numpy(copy=False)
        assert arr_3d.shape == (2, 3, 4)
        assert np_3d.shape == (2, 3, 4)
        assert arr_3d.ndim == 3
        assert np_3d.ndim == 3
        
        # 4D array
        arr_4d = cnda.ContiguousND_Cell3D([2, 2, 3, 4])
        np_4d = arr_4d.to_numpy(copy=False)
        assert arr_4d.shape == (2, 2, 3, 4)
        assert np_4d.shape == (2, 2, 3, 4)
        assert arr_4d.ndim == 4
        assert np_4d.ndim == 4
    
    def test_aos_to_numpy_strides_consistency(self, cnda):
        """Test AoS → NumPy: strides in bytes match row-major layout."""
        # Vec2f: sizeof = 8 bytes (2 * float32)
        arr_vec2f = cnda.ContiguousND_Vec2f([3, 4])
        np_vec2f = arr_vec2f.to_numpy(copy=False)
        
        # C++ strides are in element units: (4, 1)
        # NumPy strides should be in bytes: [4 * 8, 1 * 8] = [32, 8]
        expected_strides = (4 * 8, 1 * 8)
        assert np_vec2f.strides == expected_strides
        assert arr_vec2f.strides == (4, 1)  # C++ element-based strides (always tuple)
        
        # Vec3f: sizeof = 12 bytes (3 * float32)
        arr_vec3f = cnda.ContiguousND_Vec3f([2, 5])
        np_vec3f = arr_vec3f.to_numpy(copy=False)
        
        # C++ strides: (5, 1)
        # NumPy strides: [5 * 12, 1 * 12] = [60, 12]
        expected_strides = (5 * 12, 1 * 12)
        assert np_vec3f.strides == expected_strides
        assert arr_vec3f.strides == (5, 1)  # C++ element-based strides (always tuple)
        
        # Cell2D: sizeof = 12 bytes (2 * float32 + 1 * int32)
        arr_cell2d = cnda.ContiguousND_Cell2D([4, 6])
        np_cell2d = arr_cell2d.to_numpy(copy=False)
        
        # C++ strides: (6, 1)
        # NumPy strides: [6 * 12, 1 * 12] = [72, 12]
        expected_strides = (6 * 12, 1 * 12)
        assert np_cell2d.strides == expected_strides
        assert arr_cell2d.strides == (6, 1)  # C++ element-based strides (always tuple)
        
        # Cell3D: sizeof = 16 bytes (3 * float32 + 1 * int32)
        arr_cell3d = cnda.ContiguousND_Cell3D([2, 3, 4])
        np_cell3d = arr_cell3d.to_numpy(copy=False)
        
        # C++ strides: (12, 4, 1)
        # NumPy strides: [12 * 16, 4 * 16, 1 * 16] = [192, 64, 16]
        expected_strides = (12 * 16, 4 * 16, 1 * 16)
        assert np_cell3d.strides == expected_strides
        assert arr_cell3d.strides == (12, 4, 1)  # C++ element-based strides (always tuple)
    
    def test_aos_to_numpy_all_fields_accessible(self, cnda):
        """Test that all struct fields are correctly accessible from NumPy."""
        # Create and populate Vec2f array
        arr = cnda.ContiguousND_Vec2f([5, 5])
        for i in range(5):
            for j in range(5):
                arr.set_x((i, j), float(i * 10 + j))
                arr.set_y((i, j), float(i * 10 + j + 100))
        
        # Export to NumPy
        np_arr = arr.to_numpy(copy=False)
        
        # Verify all fields match
        for i in range(5):
            for j in range(5):
                assert np_arr[i, j]['x'] == pytest.approx(float(i * 10 + j))
                assert np_arr[i, j]['y'] == pytest.approx(float(i * 10 + j + 100))
                # Also verify via C++ getter
                assert arr.get_x((i, j)) == pytest.approx(float(i * 10 + j))
                assert arr.get_y((i, j)) == pytest.approx(float(i * 10 + j + 100))
        
        # Test Cell3D with all 4 fields
        grid = cnda.ContiguousND_Cell3D([3, 3])
        for i in range(3):
            for j in range(3):
                grid.set_u((i, j), float(i))
                grid.set_v((i, j), float(j))
                grid.set_w((i, j), float(i + j))
                grid.set_flag((i, j), i * 10 + j)
        
        np_grid = grid.to_numpy(copy=False)
        
        for i in range(3):
            for j in range(3):
                assert np_grid[i, j]['u'] == pytest.approx(float(i))
                assert np_grid[i, j]['v'] == pytest.approx(float(j))
                assert np_grid[i, j]['w'] == pytest.approx(float(i + j))
                assert np_grid[i, j]['flag'] == i * 10 + j
    
    def test_zero_copy_cpp_to_numpy_sync(self, cnda):
        """Test zero-copy: C++ modifications sync to NumPy view."""
        # Create C++ array and get NumPy view
        arr = cnda.ContiguousND_Vec2f([3, 4])
        np_arr = arr.to_numpy(copy=False)
        
        # Initial state
        arr.set_x((1, 2), 10.0)
        arr.set_y((1, 2), 20.0)
        
        # NumPy should immediately see C++ changes (zero-copy)
        assert np_arr[1, 2]['x'] == pytest.approx(10.0)
        assert np_arr[1, 2]['y'] == pytest.approx(20.0)
        
        # Modify via C++
        arr.set_x((1, 2), 100.0)
        arr.set_y((1, 2), 200.0)
        
        # NumPy should see updated values immediately
        assert np_arr[1, 2]['x'] == pytest.approx(100.0)
        assert np_arr[1, 2]['y'] == pytest.approx(200.0)
        
        # Test with Cell3D (multiple fields)
        grid = cnda.ContiguousND_Cell3D([2, 3])
        np_grid = grid.to_numpy(copy=False)
        
        grid.set_u((0, 1), 1.5)
        grid.set_v((0, 1), 2.5)
        grid.set_w((0, 1), 3.5)
        grid.set_flag((0, 1), 42)
        
        assert np_grid[0, 1]['u'] == pytest.approx(1.5)
        assert np_grid[0, 1]['v'] == pytest.approx(2.5)
        assert np_grid[0, 1]['w'] == pytest.approx(3.5)
        assert np_grid[0, 1]['flag'] == 42
        
        # Update again
        grid.set_u((0, 1), 10.0)
        grid.set_flag((0, 1), 99)
        
        assert np_grid[0, 1]['u'] == pytest.approx(10.0)
        assert np_grid[0, 1]['flag'] == 99
    
    def test_zero_copy_numpy_to_cpp_sync(self, cnda):
        """Test zero-copy: NumPy modifications sync to C++ view."""
        # Create C++ array and get NumPy view
        arr = cnda.ContiguousND_Vec3f([2, 3])
        np_arr = arr.to_numpy(copy=False)
        
        # Modify via NumPy
        np_arr[0, 1]['x'] = 5.5
        np_arr[0, 1]['y'] = 6.5
        np_arr[0, 1]['z'] = 7.5
        
        # C++ should see NumPy changes immediately (zero-copy)
        assert arr.get_x((0, 1)) == pytest.approx(5.5)
        assert arr.get_y((0, 1)) == pytest.approx(6.5)
        assert arr.get_z((0, 1)) == pytest.approx(7.5)
        
        # Modify again via NumPy
        np_arr[1, 2]['x'] = 100.0
        np_arr[1, 2]['y'] = 200.0
        np_arr[1, 2]['z'] = 300.0
        
        # C++ should see updated values
        assert arr.get_x((1, 2)) == pytest.approx(100.0)
        assert arr.get_y((1, 2)) == pytest.approx(200.0)
        assert arr.get_z((1, 2)) == pytest.approx(300.0)
        
        # Test with Cell2D (mixed types: float + int)
        grid = cnda.ContiguousND_Cell2D([3, 3])
        np_grid = grid.to_numpy(copy=False)
        
        np_grid[1, 1]['u'] = 10.5
        np_grid[1, 1]['v'] = 20.5
        np_grid[1, 1]['flag'] = 123
        
        assert grid.get_u((1, 1)) == pytest.approx(10.5)
        assert grid.get_v((1, 1)) == pytest.approx(20.5)
        assert grid.get_flag((1, 1)) == 123
    
    def test_zero_copy_bidirectional_sync(self, cnda):
        """Test zero-copy: bidirectional sync between C++ and NumPy."""
        arr = cnda.ContiguousND_Vec2f([4, 5])
        np_arr = arr.to_numpy(copy=False)
        
        # C++ writes, NumPy reads
        arr.set_x((2, 3), 1.0)
        arr.set_y((2, 3), 2.0)
        assert np_arr[2, 3]['x'] == pytest.approx(1.0)
        assert np_arr[2, 3]['y'] == pytest.approx(2.0)
        
        # NumPy writes, C++ reads
        np_arr[2, 3]['x'] = 10.0
        np_arr[2, 3]['y'] = 20.0
        assert arr.get_x((2, 3)) == pytest.approx(10.0)
        assert arr.get_y((2, 3)) == pytest.approx(20.0)
        
        # C++ writes again
        arr.set_x((2, 3), 100.0)
        assert np_arr[2, 3]['x'] == pytest.approx(100.0)
        
        # NumPy writes again
        np_arr[2, 3]['y'] = 200.0
        assert arr.get_y((2, 3)) == pytest.approx(200.0)
        
        # Test multiple elements
        for i in range(4):
            arr.set_x((i, 0), float(i))
            arr.set_y((i, 0), float(i * 10))
        
        for i in range(4):
            assert np_arr[i, 0]['x'] == pytest.approx(float(i))
            assert np_arr[i, 0]['y'] == pytest.approx(float(i * 10))
        
        for j in range(5):
            np_arr[0, j]['x'] = float(j * 100)
            np_arr[0, j]['y'] = float(j * 1000)
        
        for j in range(5):
            assert arr.get_x((0, j)) == pytest.approx(float(j * 100))
            assert arr.get_y((0, j)) == pytest.approx(float(j * 1000))
    
    def test_zero_copy_data_ptr_consistency(self, cnda):
        """Test that data_ptr matches NumPy array's underlying pointer."""
        # Vec2f test
        arr_vec2f = cnda.ContiguousND_Vec2f([10, 10])
        np_vec2f = arr_vec2f.to_numpy(copy=False)
        
        cpp_ptr = arr_vec2f.data_ptr()
        numpy_ptr = np_vec2f.__array_interface__['data'][0]
        
        # Both should point to the same memory location (zero-copy proof)
        assert cpp_ptr == numpy_ptr
        assert isinstance(cpp_ptr, int)
        assert cpp_ptr > 0
        
        # Vec3f test
        arr_vec3f = cnda.ContiguousND_Vec3f([5, 5])
        np_vec3f = arr_vec3f.to_numpy(copy=False)
        
        cpp_ptr = arr_vec3f.data_ptr()
        numpy_ptr = np_vec3f.__array_interface__['data'][0]
        assert cpp_ptr == numpy_ptr
        
        # Cell2D test
        arr_cell2d = cnda.ContiguousND_Cell2D([8, 8])
        np_cell2d = arr_cell2d.to_numpy(copy=False)
        
        cpp_ptr = arr_cell2d.data_ptr()
        numpy_ptr = np_cell2d.__array_interface__['data'][0]
        assert cpp_ptr == numpy_ptr
        
        # Cell3D test
        arr_cell3d = cnda.ContiguousND_Cell3D([4, 4, 4])
        np_cell3d = arr_cell3d.to_numpy(copy=False)
        
        cpp_ptr = arr_cell3d.data_ptr()
        numpy_ptr = np_cell3d.__array_interface__['data'][0]
        assert cpp_ptr == numpy_ptr
    
    def test_copy_mode_independence(self, cnda):
        """Test that copy=True creates independent memory."""
        arr = cnda.ContiguousND_Vec2f([3, 3])
        arr.set_x((1, 1), 10.0)
        arr.set_y((1, 1), 20.0)
        
        # Create a copy
        np_copy = arr.to_numpy(copy=True)
        
        # Pointers should be different
        cpp_ptr = arr.data_ptr()
        numpy_ptr = np_copy.__array_interface__['data'][0]
        assert cpp_ptr != numpy_ptr
        
        # Initial values should match
        assert np_copy[1, 1]['x'] == pytest.approx(10.0)
        assert np_copy[1, 1]['y'] == pytest.approx(20.0)
        
        # Modify C++ side
        arr.set_x((1, 1), 100.0)
        arr.set_y((1, 1), 200.0)
        
        # NumPy copy should NOT see changes (independent memory)
        assert np_copy[1, 1]['x'] == pytest.approx(10.0)
        assert np_copy[1, 1]['y'] == pytest.approx(20.0)
        
        # C++ should see its own changes
        assert arr.get_x((1, 1)) == pytest.approx(100.0)
        assert arr.get_y((1, 1)) == pytest.approx(200.0)
        
        # Modify NumPy copy
        np_copy[1, 1]['x'] = 999.0
        
        # C++ should NOT see NumPy changes
        assert arr.get_x((1, 1)) == pytest.approx(100.0)
    
    def test_data_ptr_accessibility(self, cnda):
        """Test that data_ptr is accessible."""
        arr = cnda.ContiguousND_Vec2f([10, 10])
        ptr = arr.data_ptr()
        assert isinstance(ptr, int)
        assert ptr > 0
    
    def test_contiguous_storage(self, cnda):
        """Verify contiguous storage via NumPy."""
        arr = cnda.ContiguousND_Vec2f([5, 5])
        
        # Fill with values
        for i in range(5):
            for j in range(5):
                arr.set_x((i, j), float(i * 5 + j))
                arr.set_y((i, j), float(i * 5 + j + 100))
        
        # Export to NumPy
        np_arr = arr.to_numpy(copy=False)
        
        # Should be C-contiguous
        assert np_arr.flags['C_CONTIGUOUS']
    
    def test_shape_strides_consistency(self, cnda):
        """Test that shape and strides are consistent."""
        arr = cnda.ContiguousND_Cell2D([4, 6])
        
        assert arr.shape == (4, 6)
        assert arr.ndim == 2
        assert arr.size == 24
        
        # Strides should be row-major
        strides = arr.strides
        assert len(strides) == 2
        assert strides[1] == 1  # Last dimension stride is 1
        assert strides[0] == 6  # First dimension stride is columns


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegration:
    """Integration tests for real-world use cases."""
    
    def test_fluid_simulation_workflow(self, cnda):
        """Test a simple fluid simulation workflow."""
        # Create 2D grid
        nx, ny = 20, 20
        grid = cnda.ContiguousND_Cell2D([nx, ny])
        
        # Initialize boundaries
        for i in range(nx):
            grid.set_flag((i, 0), -1)
            grid.set_flag((i, ny-1), -1)
        for j in range(ny):
            grid.set_flag((0, j), -1)
            grid.set_flag((nx-1, j), -1)
        
        # Initialize interior with velocity
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                grid.set_u((i, j), 1.0)
                grid.set_v((i, j), 0.0)
                grid.set_flag((i, j), 1)
        
        # Verify setup
        assert grid.get_flag((0, 0)) == -1
        assert grid.get_flag((10, 10)) == 1
        assert grid.get_u((10, 10)) == pytest.approx(1.0)
    
    def test_particle_array_workflow(self, cnda):
        """Test particle system workflow (1D array of particles)."""
        n_particles = 100
        
        # For now, just test that we can create the array
        # (Full Particle support would require additional bindings)
        # This is a placeholder for future implementation
        pass
    
    def test_from_numpy_zero_copy_dtype_compatibility(self, cnda):
        """Test from_numpy(copy=False): dtype compatibility with C++ AoS."""
        # Vec2f: dtype must match exactly
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        np_vec2f = np.zeros((3, 4), dtype=vec2f_dtype)
        np_vec2f[1, 2]['x'] = 10.0
        np_vec2f[1, 2]['y'] = 20.0
        
        cnda_vec2f = cnda.from_numpy_Vec2f(np_vec2f, copy=False)
        
        # Should accept and correctly interpret the structured dtype
        assert cnda_vec2f.shape == (3, 4)
        assert cnda_vec2f.get_x((1, 2)) == pytest.approx(10.0)
        assert cnda_vec2f.get_y((1, 2)) == pytest.approx(20.0)
        
        # Vec3f: 3 fields
        vec3f_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        np_vec3f = np.zeros((2, 3), dtype=vec3f_dtype)
        np_vec3f[0, 1]['x'] = 1.0
        np_vec3f[0, 1]['y'] = 2.0
        np_vec3f[0, 1]['z'] = 3.0
        
        cnda_vec3f = cnda.from_numpy_Vec3f(np_vec3f, copy=False)
        
        assert cnda_vec3f.shape == (2, 3)
        assert cnda_vec3f.get_x((0, 1)) == pytest.approx(1.0)
        assert cnda_vec3f.get_y((0, 1)) == pytest.approx(2.0)
        assert cnda_vec3f.get_z((0, 1)) == pytest.approx(3.0)
        
        # Cell2D: mixed types (float + int)
        cell2d_dtype = np.dtype([('u', np.float32), ('v', np.float32), ('flag', np.int32)])
        np_cell2d = np.zeros((4, 4), dtype=cell2d_dtype)
        np_cell2d[2, 3]['u'] = 5.5
        np_cell2d[2, 3]['v'] = 6.5
        np_cell2d[2, 3]['flag'] = 99
        
        cnda_cell2d = cnda.from_numpy_Cell2D(np_cell2d, copy=False)
        
        assert cnda_cell2d.shape == (4, 4)
        assert cnda_cell2d.get_u((2, 3)) == pytest.approx(5.5)
        assert cnda_cell2d.get_v((2, 3)) == pytest.approx(6.5)
        assert cnda_cell2d.get_flag((2, 3)) == 99
        
        # Cell3D: 4 fields
        cell3d_dtype = np.dtype([('u', np.float32), ('v', np.float32), 
                                  ('w', np.float32), ('flag', np.int32)])
        np_cell3d = np.zeros((2, 2, 2), dtype=cell3d_dtype)
        np_cell3d[1, 1, 1]['u'] = 10.0
        np_cell3d[1, 1, 1]['v'] = 20.0
        np_cell3d[1, 1, 1]['w'] = 30.0
        np_cell3d[1, 1, 1]['flag'] = 42
        
        cnda_cell3d = cnda.from_numpy_Cell3D(np_cell3d, copy=False)
        
        assert cnda_cell3d.shape == (2, 2, 2)
        assert cnda_cell3d.get_u((1, 1, 1)) == pytest.approx(10.0)
        assert cnda_cell3d.get_v((1, 1, 1)) == pytest.approx(20.0)
        assert cnda_cell3d.get_w((1, 1, 1)) == pytest.approx(30.0)
        assert cnda_cell3d.get_flag((1, 1, 1)) == 42
    
    def test_from_numpy_zero_copy_shape_preservation(self, cnda):
        """Test from_numpy(copy=False): shape preservation across dimensions."""
        # 1D array
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        np_1d = np.zeros((100,), dtype=vec2f_dtype)
        cnda_1d = cnda.from_numpy_Vec2f(np_1d, copy=False)
        
        assert np_1d.shape == (100,)
        assert cnda_1d.shape == (100,)
        assert cnda_1d.ndim == 1
        
        # 2D array
        np_2d = np.zeros((5, 10), dtype=vec2f_dtype)
        cnda_2d = cnda.from_numpy_Vec2f(np_2d, copy=False)
        
        assert np_2d.shape == (5, 10)
        assert cnda_2d.shape == (5, 10)
        assert cnda_2d.ndim == 2
        
        # 3D array
        np_3d = np.zeros((3, 4, 5), dtype=vec2f_dtype)
        cnda_3d = cnda.from_numpy_Vec2f(np_3d, copy=False)
        
        assert np_3d.shape == (3, 4, 5)
        assert cnda_3d.shape == (3, 4, 5)
        assert cnda_3d.ndim == 3
        
        # 4D array with Cell3D
        cell3d_dtype = np.dtype([('u', np.float32), ('v', np.float32), 
                                  ('w', np.float32), ('flag', np.int32)])
        np_4d = np.zeros((2, 3, 4, 5), dtype=cell3d_dtype)
        cnda_4d = cnda.from_numpy_Cell3D(np_4d, copy=False)
        
        assert np_4d.shape == (2, 3, 4, 5)
        assert cnda_4d.shape == (2, 3, 4, 5)
        assert cnda_4d.ndim == 4
    
    def test_from_numpy_zero_copy_pointer_consistency(self, cnda):
        """Test from_numpy(copy=False): pointer consistency (zero-copy proof)."""
        # Vec2f
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        np_vec2f = np.zeros((10, 10), dtype=vec2f_dtype)
        numpy_ptr_vec2f = np_vec2f.__array_interface__['data'][0]
        
        cnda_vec2f = cnda.from_numpy_Vec2f(np_vec2f, copy=False)
        cpp_ptr_vec2f = cnda_vec2f.data_ptr()
        
        # Zero-copy: pointers must be identical
        assert cpp_ptr_vec2f == numpy_ptr_vec2f
        
        # Vec3f
        vec3f_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        np_vec3f = np.zeros((8, 8), dtype=vec3f_dtype)
        numpy_ptr_vec3f = np_vec3f.__array_interface__['data'][0]
        
        cnda_vec3f = cnda.from_numpy_Vec3f(np_vec3f, copy=False)
        cpp_ptr_vec3f = cnda_vec3f.data_ptr()
        
        assert cpp_ptr_vec3f == numpy_ptr_vec3f
        
        # Cell2D
        cell2d_dtype = np.dtype([('u', np.float32), ('v', np.float32), ('flag', np.int32)])
        np_cell2d = np.zeros((6, 6), dtype=cell2d_dtype)
        numpy_ptr_cell2d = np_cell2d.__array_interface__['data'][0]
        
        cnda_cell2d = cnda.from_numpy_Cell2D(np_cell2d, copy=False)
        cpp_ptr_cell2d = cnda_cell2d.data_ptr()
        
        assert cpp_ptr_cell2d == numpy_ptr_cell2d
        
        # Cell3D
        cell3d_dtype = np.dtype([('u', np.float32), ('v', np.float32), 
                                  ('w', np.float32), ('flag', np.int32)])
        np_cell3d = np.zeros((4, 4, 4), dtype=cell3d_dtype)
        numpy_ptr_cell3d = np_cell3d.__array_interface__['data'][0]
        
        cnda_cell3d = cnda.from_numpy_Cell3D(np_cell3d, copy=False)
        cpp_ptr_cell3d = cnda_cell3d.data_ptr()
        
        assert cpp_ptr_cell3d == numpy_ptr_cell3d
    
    def test_from_numpy_zero_copy_numpy_to_cpp_sync(self, cnda):
        """Test from_numpy(copy=False): NumPy modifications visible in C++."""
        # Vec2f
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        np_arr = np.zeros((4, 5), dtype=vec2f_dtype)
        
        cnda_arr = cnda.from_numpy_Vec2f(np_arr, copy=False)
        
        # Modify NumPy array
        np_arr[2, 3]['x'] = 100.0
        np_arr[2, 3]['y'] = 200.0
        
        # C++ should see changes immediately (zero-copy)
        assert cnda_arr.get_x((2, 3)) == pytest.approx(100.0)
        assert cnda_arr.get_y((2, 3)) == pytest.approx(200.0)
        
        # Modify again
        np_arr[0, 0]['x'] = 999.0
        assert cnda_arr.get_x((0, 0)) == pytest.approx(999.0)
        
        # Cell3D with multiple fields
        cell3d_dtype = np.dtype([('u', np.float32), ('v', np.float32), 
                                  ('w', np.float32), ('flag', np.int32)])
        np_grid = np.zeros((3, 3), dtype=cell3d_dtype)
        
        cnda_grid = cnda.from_numpy_Cell3D(np_grid, copy=False)
        
        np_grid[1, 1]['u'] = 10.5
        np_grid[1, 1]['v'] = 20.5
        np_grid[1, 1]['w'] = 30.5
        np_grid[1, 1]['flag'] = 777
        
        assert cnda_grid.get_u((1, 1)) == pytest.approx(10.5)
        assert cnda_grid.get_v((1, 1)) == pytest.approx(20.5)
        assert cnda_grid.get_w((1, 1)) == pytest.approx(30.5)
        assert cnda_grid.get_flag((1, 1)) == 777
    
    def test_from_numpy_zero_copy_cpp_to_numpy_sync(self, cnda):
        """Test from_numpy(copy=False): C++ modifications visible in NumPy."""
        # Vec3f
        vec3f_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        np_arr = np.zeros((3, 4), dtype=vec3f_dtype)
        
        cnda_arr = cnda.from_numpy_Vec3f(np_arr, copy=False)
        
        # Modify via C++
        cnda_arr.set_x((1, 2), 50.0)
        cnda_arr.set_y((1, 2), 60.0)
        cnda_arr.set_z((1, 2), 70.0)
        
        # NumPy should see changes immediately (zero-copy)
        assert np_arr[1, 2]['x'] == pytest.approx(50.0)
        assert np_arr[1, 2]['y'] == pytest.approx(60.0)
        assert np_arr[1, 2]['z'] == pytest.approx(70.0)
        
        # Modify again
        cnda_arr.set_x((0, 0), 111.0)
        assert np_arr[0, 0]['x'] == pytest.approx(111.0)
        
        # Cell2D
        cell2d_dtype = np.dtype([('u', np.float32), ('v', np.float32), ('flag', np.int32)])
        np_grid = np.zeros((5, 5), dtype=cell2d_dtype)
        
        cnda_grid = cnda.from_numpy_Cell2D(np_grid, copy=False)
        
        cnda_grid.set_u((2, 2), 1.5)
        cnda_grid.set_v((2, 2), 2.5)
        cnda_grid.set_flag((2, 2), 42)
        
        assert np_grid[2, 2]['u'] == pytest.approx(1.5)
        assert np_grid[2, 2]['v'] == pytest.approx(2.5)
        assert np_grid[2, 2]['flag'] == 42
    
    def test_from_numpy_zero_copy_bidirectional_sync(self, cnda):
        """Test from_numpy(copy=False): bidirectional sync NumPy ↔ C++."""
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        np_arr = np.zeros((5, 5), dtype=vec2f_dtype)
        
        cnda_arr = cnda.from_numpy_Vec2f(np_arr, copy=False)
        
        # NumPy writes
        np_arr[1, 1]['x'] = 10.0
        np_arr[1, 1]['y'] = 20.0
        
        # C++ reads
        assert cnda_arr.get_x((1, 1)) == pytest.approx(10.0)
        assert cnda_arr.get_y((1, 1)) == pytest.approx(20.0)
        
        # C++ writes
        cnda_arr.set_x((1, 1), 100.0)
        cnda_arr.set_y((1, 1), 200.0)
        
        # NumPy reads
        assert np_arr[1, 1]['x'] == pytest.approx(100.0)
        assert np_arr[1, 1]['y'] == pytest.approx(200.0)
        
        # Batch test: NumPy writes multiple
        for i in range(5):
            np_arr[i, 0]['x'] = float(i * 10)
            np_arr[i, 0]['y'] = float(i * 20)
        
        # C++ reads all
        for i in range(5):
            assert cnda_arr.get_x((i, 0)) == pytest.approx(float(i * 10))
            assert cnda_arr.get_y((i, 0)) == pytest.approx(float(i * 20))
        
        # C++ writes multiple
        for j in range(5):
            cnda_arr.set_x((0, j), float(j * 100))
            cnda_arr.set_y((0, j), float(j * 200))
        
        # NumPy reads all
        for j in range(5):
            assert np_arr[0, j]['x'] == pytest.approx(float(j * 100))
            assert np_arr[0, j]['y'] == pytest.approx(float(j * 200))
    
    def test_from_numpy_zero_copy_roundtrip_consistency(self, cnda):
        """Test NumPy → C++(zero-copy) → NumPy consistency."""
        # Create original NumPy structured array
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        np_orig = np.zeros((4, 6), dtype=vec2f_dtype)
        
        for i in range(4):
            for j in range(6):
                np_orig[i, j]['x'] = float(i * 10 + j)
                np_orig[i, j]['y'] = float(i * 10 + j + 100)
        
        # Convert to CNDA (zero-copy)
        cnda_arr = cnda.from_numpy_Vec2f(np_orig, copy=False)
        
        # Export back to NumPy (zero-copy)
        np_view = cnda_arr.to_numpy(copy=False)
        
        # All three should share the same memory
        orig_ptr = np_orig.__array_interface__['data'][0]
        cnda_ptr = cnda_arr.data_ptr()
        view_ptr = np_view.__array_interface__['data'][0]
        
        assert orig_ptr == cnda_ptr
        assert cnda_ptr == view_ptr
        
        # Data should be identical
        assert np_view.shape == np_orig.shape
        assert np.allclose(np_view['x'], np_orig['x'])
        assert np.allclose(np_view['y'], np_orig['y'])
        
        # Modify original NumPy array
        np_orig[2, 3]['x'] = 999.0
        
        # All views should see the change
        assert cnda_arr.get_x((2, 3)) == pytest.approx(999.0)
        assert np_view[2, 3]['x'] == pytest.approx(999.0)
        
        # Modify via C++
        cnda_arr.set_y((1, 1), 888.0)
        
        # Both NumPy arrays should see it
        assert np_orig[1, 1]['y'] == pytest.approx(888.0)
        assert np_view[1, 1]['y'] == pytest.approx(888.0)
    
    def test_from_numpy_copy_mode_independence(self, cnda):
        """Test from_numpy(copy=True) creates independent memory."""
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        np_arr = np.zeros((3, 3), dtype=vec2f_dtype)
        np_arr[1, 1]['x'] = 10.0
        np_arr[1, 1]['y'] = 20.0
        
        # Create with copy=True
        cnda_arr = cnda.from_numpy_Vec2f(np_arr, copy=True)
        
        # Pointers should be different
        numpy_ptr = np_arr.__array_interface__['data'][0]
        cpp_ptr = cnda_arr.data_ptr()
        assert numpy_ptr != cpp_ptr
        
        # Initial values should match
        assert cnda_arr.get_x((1, 1)) == pytest.approx(10.0)
        assert cnda_arr.get_y((1, 1)) == pytest.approx(20.0)
        
        # Modify NumPy array
        np_arr[1, 1]['x'] = 100.0
        
        # C++ should NOT see changes (independent memory)
        assert cnda_arr.get_x((1, 1)) == pytest.approx(10.0)
        
        # Modify C++ array
        cnda_arr.set_y((1, 1), 200.0)
        
        # NumPy should NOT see changes
        assert np_arr[1, 1]['y'] == pytest.approx(20.0)
