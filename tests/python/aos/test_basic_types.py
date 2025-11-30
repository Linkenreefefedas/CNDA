"""
Test suite for CNDA Array-of-Structures (AoS) basic functionality.

This test suite covers:
1. Basic construction and properties of AoS types (Vec2f, Vec3f, Cell2D, Cell3D)
2. Field access and modification
3. Simple NumPy interoperability
4. Documentation examples
"""

import pytest
import numpy as np

@pytest.fixture(scope="module")
def cnda():
    """Import cnda module."""
    import cnda as cnda_module
    return cnda_module


# ==============================================================================
# Vec2f Tests
# ==============================================================================

class TestVec2f:
    """Test Vec2f (2D vector) AoS type."""
    
    def test_construction(self, cnda):
        """Test basic construction of Vec2f array."""
        arr = cnda.ContiguousND_Vec2f([3, 4])
        assert arr.ndim == 2
        assert arr.size == 12
        assert arr.shape == (3, 4)
    
    def test_field_access(self, cnda):
        """Test accessing individual fields via helper methods."""
        arr = cnda.ContiguousND_Vec2f([2, 2])
        
        # Set values using field setters
        arr.set_x((0, 0), 1.5)
        arr.set_y((0, 0), 2.5)
        
        # Get values using field getters
        assert arr.get_x((0, 0)) == pytest.approx(1.5)
        assert arr.get_y((0, 0)) == pytest.approx(2.5)
    
    def test_multiple_elements(self, cnda):
        """Test setting multiple elements."""
        arr = cnda.ContiguousND_Vec2f([3, 3])
        
        # Set multiple elements
        for i in range(3):
            for j in range(3):
                arr.set_x((i, j), float(i * 10 + j))
                arr.set_y((i, j), float(i * 10 + j + 100))
        
        # Verify
        assert arr.get_x((0, 0)) == pytest.approx(0.0)
        assert arr.get_y((0, 0)) == pytest.approx(100.0)
        assert arr.get_x((2, 2)) == pytest.approx(22.0)
        assert arr.get_y((2, 2)) == pytest.approx(122.0)
    
    def test_numpy_interop_structured_dtype(self, cnda):
        """Test NumPy interop with structured dtype."""
        # Create NumPy structured array
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        np_arr = np.zeros((2, 3), dtype=vec2f_dtype)
        
        # Set values
        np_arr[0, 0]['x'] = 1.0
        np_arr[0, 0]['y'] = 2.0
        np_arr[1, 2]['x'] = 3.0
        np_arr[1, 2]['y'] = 4.0
        
        # Convert to CNDA
        cnda_arr = cnda.from_numpy_Vec2f(np_arr, copy=True)
        
        assert cnda_arr.shape == (2, 3)
        assert cnda_arr.get_x((0, 0)) == pytest.approx(1.0)
        assert cnda_arr.get_y((0, 0)) == pytest.approx(2.0)
        assert cnda_arr.get_x((1, 2)) == pytest.approx(3.0)
        assert cnda_arr.get_y((1, 2)) == pytest.approx(4.0)
    
    def test_to_numpy_copy(self, cnda):
        """Test exporting Vec2f to NumPy with copy."""
        arr = cnda.ContiguousND_Vec2f([2, 2])
        arr.set_x((0, 0), 10.0)
        arr.set_y((0, 0), 20.0)
        
        np_arr = arr.to_numpy(copy=True)
        
        # Verify it's a structured array
        assert np_arr.dtype.names is not None
        assert 'x' in np_arr.dtype.names
        assert 'y' in np_arr.dtype.names
        
        # Verify values
        assert np_arr[0, 0]['x'] == pytest.approx(10.0)
        assert np_arr[0, 0]['y'] == pytest.approx(20.0)


# ==============================================================================
# Vec3f Tests
# ==============================================================================

class TestVec3f:
    """Test Vec3f (3D vector) AoS type."""
    
    def test_construction(self, cnda):
        """Test basic construction."""
        arr = cnda.ContiguousND_Vec3f([5, 5, 5])
        assert arr.ndim == 3
        assert arr.size == 125
    
    def test_field_access(self, cnda):
        """Test x, y, z field access."""
        arr = cnda.ContiguousND_Vec3f([2, 2, 2])
        
        arr.set_x((1, 1, 1), 1.5)
        arr.set_y((1, 1, 1), 2.5)
        arr.set_z((1, 1, 1), 3.5)
        
        assert arr.get_x((1, 1, 1)) == pytest.approx(1.5)
        assert arr.get_y((1, 1, 1)) == pytest.approx(2.5)
        assert arr.get_z((1, 1, 1)) == pytest.approx(3.5)
    
    def test_numpy_structured_dtype(self, cnda):
        """Test with NumPy structured dtype."""
        vec3f_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        np_arr = np.zeros((3, 3), dtype=vec3f_dtype)
        
        np_arr[1, 1]['x'] = 10.0
        np_arr[1, 1]['y'] = 20.0
        np_arr[1, 1]['z'] = 30.0
        
        cnda_arr = cnda.from_numpy_Vec3f(np_arr, copy=True)
        
        assert cnda_arr.get_x((1, 1)) == pytest.approx(10.0)
        assert cnda_arr.get_y((1, 1)) == pytest.approx(20.0)
        assert cnda_arr.get_z((1, 1)) == pytest.approx(30.0)


# ==============================================================================
# Cell2D Tests (Fluid Simulation Use Case)
# ==============================================================================

class TestCell2D:
    """Test Cell2D (2D simulation cell with velocities and flag)."""
    
    def test_construction(self, cnda):
        """Test construction of 2D grid."""
        grid = cnda.ContiguousND_Cell2D([100, 100])
        assert grid.shape == (100, 100)
        assert grid.size == 10000
    
    def test_field_access(self, cnda):
        """Test u, v, flag field access."""
        grid = cnda.ContiguousND_Cell2D([5, 5])
        
        # Set velocity field
        grid.set_u((2, 2), 1.5)
        grid.set_v((2, 2), 2.5)
        grid.set_flag((2, 2), 1)
        
        assert grid.get_u((2, 2)) == pytest.approx(1.5)
        assert grid.get_v((2, 2)) == pytest.approx(2.5)
        assert grid.get_flag((2, 2)) == 1
    
    def test_boundary_conditions(self, cnda):
        """Test setting boundary conditions."""
        grid = cnda.ContiguousND_Cell2D([10, 10])
        
        # Mark boundaries with flag = -1
        for i in range(10):
            grid.set_flag((0, i), -1)  # left
            grid.set_flag((9, i), -1)  # right
            grid.set_flag((i, 0), -1)  # bottom
            grid.set_flag((i, 9), -1)  # top
        
        # Set interior cells to fluid
        for i in range(1, 9):
            for j in range(1, 9):
                grid.set_flag((i, j), 1)
                grid.set_u((i, j), 0.0)
                grid.set_v((i, j), 0.0)
        
        # Verify boundaries
        assert grid.get_flag((0, 0)) == -1
        assert grid.get_flag((9, 9)) == -1
        
        # Verify interior
        assert grid.get_flag((5, 5)) == 1
    
    def test_numpy_structured_dtype(self, cnda):
        """Test NumPy interop with Cell2D structured dtype."""
        cell2d_dtype = np.dtype([('u', np.float32), ('v', np.float32), ('flag', np.int32)])
        np_arr = np.zeros((3, 3), dtype=cell2d_dtype)
        
        np_arr[1, 1]['u'] = 5.0
        np_arr[1, 1]['v'] = 10.0
        np_arr[1, 1]['flag'] = 2
        
        cnda_arr = cnda.from_numpy_Cell2D(np_arr, copy=True)
        
        assert cnda_arr.get_u((1, 1)) == pytest.approx(5.0)
        assert cnda_arr.get_v((1, 1)) == pytest.approx(10.0)
        assert cnda_arr.get_flag((1, 1)) == 2


# ==============================================================================
# Cell3D Tests
# ==============================================================================

class TestCell3D:
    """Test Cell3D (3D simulation cell)."""
    
    def test_construction(self, cnda):
        """Test 3D grid construction."""
        grid = cnda.ContiguousND_Cell3D([10, 10, 10])
        assert grid.shape == (10, 10, 10)
        assert grid.size == 1000
    
    def test_velocity_field(self, cnda):
        """Test 3D velocity field."""
        grid = cnda.ContiguousND_Cell3D([3, 3, 3])
        
        grid.set_u((1, 1, 1), 1.0)
        grid.set_v((1, 1, 1), 2.0)
        grid.set_w((1, 1, 1), 3.0)
        grid.set_flag((1, 1, 1), 1)
        
        assert grid.get_u((1, 1, 1)) == pytest.approx(1.0)
        assert grid.get_v((1, 1, 1)) == pytest.approx(2.0)
        assert grid.get_w((1, 1, 1)) == pytest.approx(3.0)
        assert grid.get_flag((1, 1, 1)) == 1
    
    def test_numpy_interop(self, cnda):
        """Test NumPy interop."""
        cell3d_dtype = np.dtype([('u', np.float32), ('v', np.float32), 
                                  ('w', np.float32), ('flag', np.int32)])
        np_arr = np.zeros((2, 2, 2), dtype=cell3d_dtype)
        
        np_arr[0, 0, 0]['u'] = 1.0
        np_arr[0, 0, 0]['v'] = 2.0
        np_arr[0, 0, 0]['w'] = 3.0
        np_arr[0, 0, 0]['flag'] = 5
        
        cnda_arr = cnda.from_numpy_Cell3D(np_arr, copy=True)
        
        assert cnda_arr.get_u((0, 0, 0)) == pytest.approx(1.0)
        assert cnda_arr.get_v((0, 0, 0)) == pytest.approx(2.0)
        assert cnda_arr.get_w((0, 0, 0)) == pytest.approx(3.0)
        assert cnda_arr.get_flag((0, 0, 0)) == 5


# ==============================================================================
# Documentation Examples
# ==============================================================================

class TestDocumentationExamples:
    """Test examples from documentation."""
    
    def test_vec2f_basic_example(self, cnda):
        """Example: Basic Vec2f usage."""
        # Create 2D array of 2D vectors
        vectors = cnda.ContiguousND_Vec2f([10, 10])
        
        # Set a vector at position (5, 5)
        vectors.set_x((5, 5), 1.5)
        vectors.set_y((5, 5), 2.5)
        
        # Read it back
        x = vectors.get_x((5, 5))
        y = vectors.get_y((5, 5))
        
        assert x == pytest.approx(1.5)
        assert y == pytest.approx(2.5)
    
    def test_cell2d_fluid_example(self, cnda):
        """Example: Fluid dynamics grid."""
        # Create velocity field
        grid = cnda.ContiguousND_Cell2D([50, 50])
        
        # Set initial conditions
        for i in range(10, 40):
            for j in range(10, 40):
                grid.set_u((i, j), 1.0)  # x-velocity
                grid.set_v((i, j), 0.5)  # y-velocity
                grid.set_flag((i, j), 1)  # fluid cell
        
        # Verify
        assert grid.get_u((25, 25)) == pytest.approx(1.0)
        assert grid.get_v((25, 25)) == pytest.approx(0.5)
        assert grid.get_flag((25, 25)) == 1
