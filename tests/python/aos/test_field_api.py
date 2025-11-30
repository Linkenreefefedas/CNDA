"""
Test suite for CNDA AoS field API consistency.

This test suite ensures field getter/setter APIs correctly map to struct fields, preventing field mix-up bugs.
"""

import pytest
import numpy as np


@pytest.fixture(scope="module")
def cnda():
    """Import cnda module."""
    import cnda as cnda_module
    return cnda_module


class TestFieldAPIConsistency:
    """
    Comprehensive tests to ensure field getter/setter APIs correctly map
    to the corresponding struct fields, preventing field mix-up bugs.
    
    These tests validate:
    1. get_x() reads the x field, not y
    2. set_y() writes the y field, not x
    3. NumPy structured dtype field order matches C++ struct layout
    4. Field values are independent and not corrupted
    """
    
    def test_vec2f_field_consistency(self, cnda):
        """Verify Vec2f get/set APIs map correctly to x, y fields."""
        arr = cnda.ContiguousND_Vec2f([2, 2])
        
        # Set distinctive values for x and y
        arr.set_x((0, 0), 11.0)
        arr.set_y((0, 0), 22.0)
        arr.set_x((1, 1), 33.0)
        arr.set_y((1, 1), 44.0)
        
        # Verify x is x, not y
        assert arr.get_x((0, 0)) == pytest.approx(11.0)
        assert arr.get_x((1, 1)) == pytest.approx(33.0)
        
        # Verify y is y, not x
        assert arr.get_y((0, 0)) == pytest.approx(22.0)
        assert arr.get_y((1, 1)) == pytest.approx(44.0)
        
        # Cross-check: x != y
        assert arr.get_x((0, 0)) != arr.get_y((0, 0))
        assert arr.get_x((1, 1)) != arr.get_y((1, 1))
        
        # Verify via NumPy structured dtype
        np_view = arr.to_numpy(copy=False)
        assert np_view[0, 0]['x'] == pytest.approx(11.0)
        assert np_view[0, 0]['y'] == pytest.approx(22.0)
        assert np_view[1, 1]['x'] == pytest.approx(33.0)
        assert np_view[1, 1]['y'] == pytest.approx(44.0)
    
    def test_vec3f_field_consistency(self, cnda):
        """Verify Vec3f get/set APIs map correctly to x, y, z fields."""
        arr = cnda.ContiguousND_Vec3f([3, 2])
        
        # Set unique values for each field
        arr.set_x((0, 0), 10.0)
        arr.set_y((0, 0), 20.0)
        arr.set_z((0, 0), 30.0)
        
        arr.set_x((2, 1), 100.0)
        arr.set_y((2, 1), 200.0)
        arr.set_z((2, 1), 300.0)
        
        # Verify each field is distinct
        assert arr.get_x((0, 0)) == pytest.approx(10.0)
        assert arr.get_y((0, 0)) == pytest.approx(20.0)
        assert arr.get_z((0, 0)) == pytest.approx(30.0)
        
        assert arr.get_x((2, 1)) == pytest.approx(100.0)
        assert arr.get_y((2, 1)) == pytest.approx(200.0)
        assert arr.get_z((2, 1)) == pytest.approx(300.0)
        
        # Verify no field mix-up
        assert arr.get_x((0, 0)) != arr.get_y((0, 0))
        assert arr.get_y((0, 0)) != arr.get_z((0, 0))
        assert arr.get_x((0, 0)) != arr.get_z((0, 0))
        
        # Cross-validate with NumPy
        np_view = arr.to_numpy(copy=False)
        assert np_view[0, 0]['x'] == pytest.approx(10.0)
        assert np_view[0, 0]['y'] == pytest.approx(20.0)
        assert np_view[0, 0]['z'] == pytest.approx(30.0)
        assert np_view.dtype.names == ('x', 'y', 'z')
    
    def test_cell2d_field_consistency(self, cnda):
        """Verify Cell2D get/set APIs map correctly to u, v, flag fields."""
        arr = cnda.ContiguousND_Cell2D([2, 2, 2])
        
        # Set distinct values
        arr.set_u((0, 0, 0), 5.5)
        arr.set_v((0, 0, 0), 6.5)
        arr.set_flag((0, 0, 0), 42)
        
        arr.set_u((1, 1, 1), 15.5)
        arr.set_v((1, 1, 1), 16.5)
        arr.set_flag((1, 1, 1), 99)
        
        # Verify u, v, flag are distinct
        assert arr.get_u((0, 0, 0)) == pytest.approx(5.5)
        assert arr.get_v((0, 0, 0)) == pytest.approx(6.5)
        assert arr.get_flag((0, 0, 0)) == 42
        
        assert arr.get_u((1, 1, 1)) == pytest.approx(15.5)
        assert arr.get_v((1, 1, 1)) == pytest.approx(16.5)
        assert arr.get_flag((1, 1, 1)) == 99
        
        # Verify float fields are not mixed
        assert arr.get_u((0, 0, 0)) != arr.get_v((0, 0, 0))
        
        # Cross-validate with NumPy
        np_view = arr.to_numpy(copy=False)
        assert np_view[0, 0, 0]['u'] == pytest.approx(5.5)
        assert np_view[0, 0, 0]['v'] == pytest.approx(6.5)
        assert np_view[0, 0, 0]['flag'] == 42
        assert np_view.dtype.names == ('u', 'v', 'flag')
    
    def test_cell3d_field_consistency(self, cnda):
        """Verify Cell3D get/set APIs map correctly to u, v, w, flag fields."""
        arr = cnda.ContiguousND_Cell3D([2, 2])
        
        # Set unique values
        arr.set_u((0, 0), 1.0)
        arr.set_v((0, 0), 2.0)
        arr.set_w((0, 0), 3.0)
        arr.set_flag((0, 0), 111)
        
        arr.set_u((1, 1), 10.0)
        arr.set_v((1, 1), 20.0)
        arr.set_w((1, 1), 30.0)
        arr.set_flag((1, 1), 222)
        
        # Verify each field is correct
        assert arr.get_u((0, 0)) == pytest.approx(1.0)
        assert arr.get_v((0, 0)) == pytest.approx(2.0)
        assert arr.get_w((0, 0)) == pytest.approx(3.0)
        assert arr.get_flag((0, 0)) == 111
        
        assert arr.get_u((1, 1)) == pytest.approx(10.0)
        assert arr.get_v((1, 1)) == pytest.approx(20.0)
        assert arr.get_w((1, 1)) == pytest.approx(30.0)
        assert arr.get_flag((1, 1)) == 222
        
        # Verify no mix-up
        assert arr.get_u((0, 0)) != arr.get_v((0, 0))
        assert arr.get_v((0, 0)) != arr.get_w((0, 0))
        assert arr.get_u((0, 0)) != arr.get_w((0, 0))
        
        # Cross-validate with NumPy
        np_view = arr.to_numpy(copy=False)
        assert np_view[0, 0]['u'] == pytest.approx(1.0)
        assert np_view[0, 0]['v'] == pytest.approx(2.0)
        assert np_view[0, 0]['w'] == pytest.approx(3.0)
        assert np_view[0, 0]['flag'] == 111
        assert np_view.dtype.names == ('u', 'v', 'w', 'flag')
    
    def test_particle_field_consistency(self, cnda):
        """Verify Particle get/set APIs map correctly to all 7 fields."""
        arr = cnda.ContiguousND_Particle([2])
        
        # Set unique values for position, velocity, mass
        arr.set_x((0,), 1.0)
        arr.set_y((0,), 2.0)
        arr.set_z((0,), 3.0)
        arr.set_vx((0,), 10.0)
        arr.set_vy((0,), 20.0)
        arr.set_vz((0,), 30.0)
        arr.set_mass((0,), 100.0)
        
        arr.set_x((1,), 4.0)
        arr.set_y((1,), 5.0)
        arr.set_z((1,), 6.0)
        arr.set_vx((1,), 40.0)
        arr.set_vy((1,), 50.0)
        arr.set_vz((1,), 60.0)
        arr.set_mass((1,), 200.0)
        
        # Verify all fields are distinct
        assert arr.get_x((0,)) == pytest.approx(1.0)
        assert arr.get_y((0,)) == pytest.approx(2.0)
        assert arr.get_z((0,)) == pytest.approx(3.0)
        assert arr.get_vx((0,)) == pytest.approx(10.0)
        assert arr.get_vy((0,)) == pytest.approx(20.0)
        assert arr.get_vz((0,)) == pytest.approx(30.0)
        assert arr.get_mass((0,)) == pytest.approx(100.0)
        
        # Verify position != velocity
        assert arr.get_x((0,)) != arr.get_vx((0,))
        assert arr.get_y((0,)) != arr.get_vy((0,))
        assert arr.get_z((0,)) != arr.get_vz((0,))
        
        # Cross-validate with NumPy
        np_view = arr.to_numpy(copy=False)
        assert np_view[0]['x'] == pytest.approx(1.0)
        assert np_view[0]['vx'] == pytest.approx(10.0)
        assert np_view[0]['mass'] == pytest.approx(100.0)
        assert np_view.dtype.names == ('x', 'y', 'z', 'vx', 'vy', 'vz', 'mass')
    
    def test_materialpoint_field_consistency(self, cnda):
        """Verify MaterialPoint get/set APIs map correctly to density, temp, pressure, id."""
        arr = cnda.ContiguousND_MaterialPoint([3])
        
        # Set unique values
        arr.set_density((0,), 1000.0)
        arr.set_temperature((0,), 273.15)
        arr.set_pressure((0,), 101325.0)
        arr.set_id((0,), 1)
        
        arr.set_density((2,), 2000.0)
        arr.set_temperature((2,), 373.15)
        arr.set_pressure((2,), 202650.0)
        arr.set_id((2,), 999)
        
        # Verify fields are distinct
        assert arr.get_density((0,)) == pytest.approx(1000.0)
        assert arr.get_temperature((0,)) == pytest.approx(273.15)
        assert arr.get_pressure((0,)) == pytest.approx(101325.0)
        assert arr.get_id((0,)) == 1
        
        assert arr.get_density((2,)) == pytest.approx(2000.0)
        assert arr.get_temperature((2,)) == pytest.approx(373.15)
        assert arr.get_pressure((2,)) == pytest.approx(202650.0)
        assert arr.get_id((2,)) == 999
        
        # Verify no mix-up
        assert arr.get_density((0,)) != arr.get_temperature((0,))
        assert arr.get_temperature((0,)) != arr.get_pressure((0,))
        
        # Cross-validate with NumPy
        np_view = arr.to_numpy(copy=False)
        assert np_view[0]['density'] == pytest.approx(1000.0)
        assert np_view[0]['temperature'] == pytest.approx(273.15)
        assert np_view[0]['pressure'] == pytest.approx(101325.0)
        assert np_view[0]['id'] == 1
        assert np_view.dtype.names == ('density', 'temperature', 'pressure', 'id')
    
    def test_numpy_to_cnda_field_order_consistency(self, cnda):
        """
        Verify that NumPy → CNDA conversion maintains correct field mapping.
        
        This test ensures that when creating CNDA arrays from NumPy structured
        arrays, the field order and values are correctly preserved.
        """
        # Test Vec2f
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        np_arr = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=vec2f_dtype)
        cnda_arr = cnda.from_numpy_Vec2f(np_arr, copy=True)
        
        assert cnda_arr.get_x((0,)) == pytest.approx(1.0)
        assert cnda_arr.get_y((0,)) == pytest.approx(2.0)
        assert cnda_arr.get_x((1,)) == pytest.approx(3.0)
        assert cnda_arr.get_y((1,)) == pytest.approx(4.0)
        
        # Test Cell3D
        cell3d_dtype = np.dtype([('u', np.float32), ('v', np.float32), 
                                  ('w', np.float32), ('flag', np.int32)])
        np_arr = np.array([(1.0, 2.0, 3.0, 10), (4.0, 5.0, 6.0, 20)], 
                          dtype=cell3d_dtype)
        cnda_arr = cnda.from_numpy_Cell3D(np_arr, copy=True)
        
        assert cnda_arr.get_u((0,)) == pytest.approx(1.0)
        assert cnda_arr.get_v((0,)) == pytest.approx(2.0)
        assert cnda_arr.get_w((0,)) == pytest.approx(3.0)
        assert cnda_arr.get_flag((0,)) == 10
        
        assert cnda_arr.get_u((1,)) == pytest.approx(4.0)
        assert cnda_arr.get_v((1,)) == pytest.approx(5.0)
        assert cnda_arr.get_w((1,)) == pytest.approx(6.0)
        assert cnda_arr.get_flag((1,)) == 20
    
    def test_bidirectional_field_sync_consistency(self, cnda):
        """
        Test that bidirectional zero-copy sync maintains field consistency.
        
        This ensures that when modifying fields through C++ or NumPy,
        the correct field is modified and other fields remain unchanged.
        """
        # C++ → NumPy field isolation
        arr = cnda.ContiguousND_Vec3f([3])
        arr.set_x((0,), 1.0)
        arr.set_y((0,), 2.0)
        arr.set_z((0,), 3.0)
        
        np_view = arr.to_numpy(copy=False)
        
        # Modify x via NumPy
        np_view[0]['x'] = 100.0
        
        # Verify x changed, y and z unchanged
        assert arr.get_x((0,)) == pytest.approx(100.0)
        assert arr.get_y((0,)) == pytest.approx(2.0)  # Should NOT change
        assert arr.get_z((0,)) == pytest.approx(3.0)  # Should NOT change
        
        # NumPy → C++ field isolation
        cell2d_dtype = np.dtype([('u', np.float32), ('v', np.float32), ('flag', np.int32)])
        np_arr = np.zeros((2,), dtype=cell2d_dtype)
        np_arr[0] = (5.0, 6.0, 42)
        
        cnda_arr = cnda.from_numpy_Cell2D(np_arr, copy=False)
        
        # Modify v via C++
        cnda_arr.set_v((0,), 99.0)
        
        # Verify v changed, u and flag unchanged
        assert np_arr[0]['u'] == pytest.approx(5.0)   # Should NOT change
        assert np_arr[0]['v'] == pytest.approx(99.0)  # Changed
        assert np_arr[0]['flag'] == 42                # Should NOT change
    
    def test_field_independence_across_elements(self, cnda):
        """
        Verify that modifying one element's field doesn't affect other elements.
        """
        arr = cnda.ContiguousND_Vec2f([10])
        
        # Initialize all elements
        for i in range(10):
            arr.set_x((i,), float(i))
            arr.set_y((i,), float(i * 10))
        
        # Modify element 5
        arr.set_x((5,), 999.0)
        arr.set_y((5,), 888.0)
        
        # Verify element 5 changed
        assert arr.get_x((5,)) == pytest.approx(999.0)
        assert arr.get_y((5,)) == pytest.approx(888.0)
        
        # Verify other elements unchanged
        assert arr.get_x((0,)) == pytest.approx(0.0)
        assert arr.get_y((0,)) == pytest.approx(0.0)
        assert arr.get_x((4,)) == pytest.approx(4.0)
        assert arr.get_y((4,)) == pytest.approx(40.0)
        assert arr.get_x((6,)) == pytest.approx(6.0)
        assert arr.get_y((6,)) == pytest.approx(60.0)
        assert arr.get_x((9,)) == pytest.approx(9.0)
        assert arr.get_y((9,)) == pytest.approx(90.0)
    
    def test_dtype_field_order_matches_cpp_struct_order(self, cnda):
        """
        Critical test: Verify NumPy dtype field order exactly matches C++ struct field order.
        
        This is essential because the zero-copy mechanism relies on identical memory layout.
        If field order is wrong, data will be misinterpreted.
        """
        # Vec2f: C++ order is {x, y}
        arr = cnda.ContiguousND_Vec2f([1])
        np_view = arr.to_numpy(copy=False)
        assert np_view.dtype.names == ('x', 'y'), "Vec2f field order must be (x, y)"
        
        # Vec3f: C++ order is {x, y, z}
        arr = cnda.ContiguousND_Vec3f([1])
        np_view = arr.to_numpy(copy=False)
        assert np_view.dtype.names == ('x', 'y', 'z'), "Vec3f field order must be (x, y, z)"
        
        # Cell2D: C++ order is {u, v, flag}
        arr = cnda.ContiguousND_Cell2D([1])
        np_view = arr.to_numpy(copy=False)
        assert np_view.dtype.names == ('u', 'v', 'flag'), "Cell2D field order must be (u, v, flag)"
        
        # Cell3D: C++ order is {u, v, w, flag}
        arr = cnda.ContiguousND_Cell3D([1])
        np_view = arr.to_numpy(copy=False)
        assert np_view.dtype.names == ('u', 'v', 'w', 'flag'), "Cell3D field order must be (u, v, w, flag)"
        
        # Particle: C++ order is {x, y, z, vx, vy, vz, mass}
        arr = cnda.ContiguousND_Particle([1])
        np_view = arr.to_numpy(copy=False)
        assert np_view.dtype.names == ('x', 'y', 'z', 'vx', 'vy', 'vz', 'mass'), \
            "Particle field order must be (x, y, z, vx, vy, vz, mass)"
        
        # MaterialPoint: C++ order is {density, temperature, pressure, id}
        arr = cnda.ContiguousND_MaterialPoint([1])
        np_view = arr.to_numpy(copy=False)
        assert np_view.dtype.names == ('density', 'temperature', 'pressure', 'id'), \
            "MaterialPoint field order must be (density, temperature, pressure, id)"
