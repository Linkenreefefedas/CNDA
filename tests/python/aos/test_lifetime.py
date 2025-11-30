"""
Test suite for CNDA AoS lifetime management.

This test suite covers lifetime and ownership management for zero-copy AoS arrays.
"""

import pytest
import numpy as np
import gc


@pytest.fixture(scope="module")
def cnda():
    """Import cnda module."""
    import cnda as cnda_module
    return cnda_module


class TestAoSLifetimeManagement:
    """Test lifetime and ownership management for AoS types with zero-copy."""
    
    def test_to_numpy_cpp_deleted_numpy_remains_valid_vec2f(self, cnda):
        """Test to_numpy(copy=False): NumPy view valid after C++ object deleted (Vec2f)."""
        import gc
        
        # Create C++ array
        arr = cnda.ContiguousND_Vec2f([3, 4])
        arr.set_x((1, 2), 10.5)
        arr.set_y((1, 2), 20.5)
        arr.set_x((0, 0), 1.0)
        arr.set_y((0, 0), 2.0)
        
        # Get data pointer before deletion
        data_ptr = arr.data_ptr()
        
        # Create zero-copy NumPy view
        np_view = arr.to_numpy(copy=False)
        
        # Verify initial values
        assert np_view[1, 2]['x'] == pytest.approx(10.5)
        assert np_view[1, 2]['y'] == pytest.approx(20.5)
        assert np_view[0, 0]['x'] == pytest.approx(1.0)
        assert np_view[0, 0]['y'] == pytest.approx(2.0)
        
        # Delete C++ object
        del arr
        gc.collect()
        
        # NumPy view should still be valid (capsule keeps buffer alive)
        assert np_view[1, 2]['x'] == pytest.approx(10.5)
        assert np_view[1, 2]['y'] == pytest.approx(20.5)
        assert np_view[0, 0]['x'] == pytest.approx(1.0)
        assert np_view[0, 0]['y'] == pytest.approx(2.0)
        
        # Pointer should remain the same
        assert np_view.__array_interface__['data'][0] == data_ptr
        
        # Modifications should still work
        np_view[2, 3]['x'] = 99.0
        assert np_view[2, 3]['x'] == pytest.approx(99.0)
    
    def test_to_numpy_cpp_deleted_numpy_remains_valid_vec3f(self, cnda):
        """Test to_numpy(copy=False): NumPy view valid after C++ deleted (Vec3f)."""
        import gc
        
        arr = cnda.ContiguousND_Vec3f([2, 3])
        arr.set_x((0, 1), 5.5)
        arr.set_y((0, 1), 6.5)
        arr.set_z((0, 1), 7.5)
        
        data_ptr = arr.data_ptr()
        np_view = arr.to_numpy(copy=False)
        
        # Delete C++
        del arr
        gc.collect()
        
        # NumPy should remain valid
        assert np_view[0, 1]['x'] == pytest.approx(5.5)
        assert np_view[0, 1]['y'] == pytest.approx(6.5)
        assert np_view[0, 1]['z'] == pytest.approx(7.5)
        assert np_view.__array_interface__['data'][0] == data_ptr
    
    def test_to_numpy_cpp_deleted_numpy_remains_valid_cell2d(self, cnda):
        """Test to_numpy(copy=False): NumPy view valid after C++ deleted (Cell2D)."""
        import gc
        
        grid = cnda.ContiguousND_Cell2D([4, 5])
        grid.set_u((2, 3), 10.0)
        grid.set_v((2, 3), 20.0)
        grid.set_flag((2, 3), 42)
        
        data_ptr = grid.data_ptr()
        np_view = grid.to_numpy(copy=False)
        
        # Delete C++
        del grid
        gc.collect()
        
        # NumPy should remain valid (all fields)
        assert np_view[2, 3]['u'] == pytest.approx(10.0)
        assert np_view[2, 3]['v'] == pytest.approx(20.0)
        assert np_view[2, 3]['flag'] == 42
        assert np_view.__array_interface__['data'][0] == data_ptr
    
    def test_to_numpy_cpp_deleted_numpy_remains_valid_cell3d(self, cnda):
        """Test to_numpy(copy=False): NumPy view valid after C++ deleted (Cell3D)."""
        import gc
        
        grid = cnda.ContiguousND_Cell3D([3, 3, 3])
        grid.set_u((1, 1, 1), 1.5)
        grid.set_v((1, 1, 1), 2.5)
        grid.set_w((1, 1, 1), 3.5)
        grid.set_flag((1, 1, 1), 99)
        
        data_ptr = grid.data_ptr()
        np_view = grid.to_numpy(copy=False)
        
        # Delete C++
        del grid
        gc.collect()
        
        # NumPy should remain valid (all 4 fields)
        assert np_view[1, 1, 1]['u'] == pytest.approx(1.5)
        assert np_view[1, 1, 1]['v'] == pytest.approx(2.5)
        assert np_view[1, 1, 1]['w'] == pytest.approx(3.5)
        assert np_view[1, 1, 1]['flag'] == 99
        assert np_view.__array_interface__['data'][0] == data_ptr
    
    def test_to_numpy_multiple_views_after_cpp_deleted(self, cnda):
        """Test multiple NumPy views remain valid after C++ deleted."""
        import gc
        
        arr = cnda.ContiguousND_Vec2f([3, 3])
        arr.set_x((1, 1), 10.0)
        arr.set_y((1, 1), 20.0)
        
        # Create multiple views
        view1 = arr.to_numpy(copy=False)
        view2 = arr.to_numpy(copy=False)
        view3 = arr.to_numpy(copy=False)
        
        # Delete C++
        del arr
        gc.collect()
        
        # All views should remain valid
        assert view1[1, 1]['x'] == pytest.approx(10.0)
        assert view2[1, 1]['x'] == pytest.approx(10.0)
        assert view3[1, 1]['x'] == pytest.approx(10.0)
        
        # Modification in one view visible in all
        view1[1, 1]['y'] = 99.0
        assert view2[1, 1]['y'] == pytest.approx(99.0)
        assert view3[1, 1]['y'] == pytest.approx(99.0)
    
    def test_from_numpy_numpy_deleted_cpp_remains_valid_vec2f(self, cnda):
        """Test from_numpy(copy=False): C++ view valid after NumPy deleted (Vec2f)."""
        import gc
        
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        np_arr = np.zeros((4, 5), dtype=vec2f_dtype)
        np_arr[2, 3]['x'] = 15.5
        np_arr[2, 3]['y'] = 25.5
        np_arr[0, 0]['x'] = 1.0
        np_arr[0, 0]['y'] = 2.0
        
        # Get data pointer before deletion
        data_ptr = np_arr.__array_interface__['data'][0]
        
        # Create zero-copy C++ view
        cnda_arr = cnda.from_numpy_Vec2f(np_arr, copy=False)
        
        # Verify initial values
        assert cnda_arr.get_x((2, 3)) == pytest.approx(15.5)
        assert cnda_arr.get_y((2, 3)) == pytest.approx(25.5)
        assert cnda_arr.get_x((0, 0)) == pytest.approx(1.0)
        assert cnda_arr.get_y((0, 0)) == pytest.approx(2.0)
        
        # Delete NumPy array
        del np_arr
        gc.collect()
        
        # C++ view should still be valid (shared_ptr keeps buffer alive)
        assert cnda_arr.get_x((2, 3)) == pytest.approx(15.5)
        assert cnda_arr.get_y((2, 3)) == pytest.approx(25.5)
        assert cnda_arr.get_x((0, 0)) == pytest.approx(1.0)
        assert cnda_arr.get_y((0, 0)) == pytest.approx(2.0)
        
        # Pointer should remain the same
        assert cnda_arr.data_ptr() == data_ptr
        
        # Modifications should still work
        cnda_arr.set_x((1, 1), 88.0)
        assert cnda_arr.get_x((1, 1)) == pytest.approx(88.0)
    
    def test_from_numpy_numpy_deleted_cpp_remains_valid_vec3f(self, cnda):
        """Test from_numpy(copy=False): C++ view valid after NumPy deleted (Vec3f)."""
        import gc
        
        vec3f_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        np_arr = np.zeros((3, 4), dtype=vec3f_dtype)
        np_arr[1, 2]['x'] = 7.5
        np_arr[1, 2]['y'] = 8.5
        np_arr[1, 2]['z'] = 9.5
        
        data_ptr = np_arr.__array_interface__['data'][0]
        cnda_arr = cnda.from_numpy_Vec3f(np_arr, copy=False)
        
        # Delete NumPy
        del np_arr
        gc.collect()
        
        # C++ should remain valid
        assert cnda_arr.get_x((1, 2)) == pytest.approx(7.5)
        assert cnda_arr.get_y((1, 2)) == pytest.approx(8.5)
        assert cnda_arr.get_z((1, 2)) == pytest.approx(9.5)
        assert cnda_arr.data_ptr() == data_ptr
    
    def test_from_numpy_numpy_deleted_cpp_remains_valid_cell2d(self, cnda):
        """Test from_numpy(copy=False): C++ view valid after NumPy deleted (Cell2D)."""
        import gc
        
        cell2d_dtype = np.dtype([('u', np.float32), ('v', np.float32), ('flag', np.int32)])
        np_arr = np.zeros((5, 6), dtype=cell2d_dtype)
        np_arr[3, 4]['u'] = 12.5
        np_arr[3, 4]['v'] = 13.5
        np_arr[3, 4]['flag'] = 77
        
        data_ptr = np_arr.__array_interface__['data'][0]
        cnda_arr = cnda.from_numpy_Cell2D(np_arr, copy=False)
        
        # Delete NumPy
        del np_arr
        gc.collect()
        
        # C++ should remain valid (all fields)
        assert cnda_arr.get_u((3, 4)) == pytest.approx(12.5)
        assert cnda_arr.get_v((3, 4)) == pytest.approx(13.5)
        assert cnda_arr.get_flag((3, 4)) == 77
        assert cnda_arr.data_ptr() == data_ptr
    
    def test_from_numpy_numpy_deleted_cpp_remains_valid_cell3d(self, cnda):
        """Test from_numpy(copy=False): C++ view valid after NumPy deleted (Cell3D)."""
        import gc
        
        cell3d_dtype = np.dtype([('u', np.float32), ('v', np.float32), 
                                  ('w', np.float32), ('flag', np.int32)])
        np_arr = np.zeros((4, 4, 4), dtype=cell3d_dtype)
        np_arr[2, 2, 2]['u'] = 5.0
        np_arr[2, 2, 2]['v'] = 6.0
        np_arr[2, 2, 2]['w'] = 7.0
        np_arr[2, 2, 2]['flag'] = 123
        
        data_ptr = np_arr.__array_interface__['data'][0]
        cnda_arr = cnda.from_numpy_Cell3D(np_arr, copy=False)
        
        # Delete NumPy
        del np_arr
        gc.collect()
        
        # C++ should remain valid (all 4 fields)
        assert cnda_arr.get_u((2, 2, 2)) == pytest.approx(5.0)
        assert cnda_arr.get_v((2, 2, 2)) == pytest.approx(6.0)
        assert cnda_arr.get_w((2, 2, 2)) == pytest.approx(7.0)
        assert cnda_arr.get_flag((2, 2, 2)) == 123
        assert cnda_arr.data_ptr() == data_ptr
    
    def test_from_numpy_multiple_cpp_views_after_numpy_deleted(self, cnda):
        """Test multiple C++ views remain valid after NumPy deleted."""
        import gc
        
        vec2f_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        np_arr = np.zeros((3, 3), dtype=vec2f_dtype)
        np_arr[1, 1]['x'] = 50.0
        np_arr[1, 1]['y'] = 60.0
        
        # Create multiple C++ views
        view1 = cnda.from_numpy_Vec2f(np_arr, copy=False)
        view2 = cnda.from_numpy_Vec2f(np_arr, copy=False)
        view3 = cnda.from_numpy_Vec2f(np_arr, copy=False)
        
        # Delete NumPy
        del np_arr
        gc.collect()
        
        # All C++ views should remain valid
        assert view1.get_x((1, 1)) == pytest.approx(50.0)
        assert view2.get_x((1, 1)) == pytest.approx(50.0)
        assert view3.get_x((1, 1)) == pytest.approx(50.0)
        
        # Modification in one view visible in all
        view1.set_y((1, 1), 999.0)
        assert view2.get_y((1, 1)) == pytest.approx(999.0)
        assert view3.get_y((1, 1)) == pytest.approx(999.0)
    
    def test_roundtrip_lifetime_numpy_to_cpp_to_numpy(self, cnda):
        """Test roundtrip: NumPy → C++ → NumPy with intermediate deletions."""
        import gc
        
        vec3f_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        np_orig = np.zeros((3, 3), dtype=vec3f_dtype)
        np_orig[1, 1]['x'] = 10.0
        np_orig[1, 1]['y'] = 20.0
        np_orig[1, 1]['z'] = 30.0
        
        original_ptr = np_orig.__array_interface__['data'][0]
        
        # NumPy → C++ (zero-copy)
        cnda_arr = cnda.from_numpy_Vec3f(np_orig, copy=False)
        
        # Delete original NumPy
        del np_orig
        gc.collect()
        
        # C++ → NumPy (zero-copy)
        np_final = cnda_arr.to_numpy(copy=False)
        
        # Delete intermediate C++
        del cnda_arr
        gc.collect()
        
        # Final NumPy should still be valid
        assert np_final[1, 1]['x'] == pytest.approx(10.0)
        assert np_final[1, 1]['y'] == pytest.approx(20.0)
        assert np_final[1, 1]['z'] == pytest.approx(30.0)
        
        # Should still point to same memory
        assert np_final.__array_interface__['data'][0] == original_ptr
    
    def test_roundtrip_lifetime_cpp_to_numpy_to_cpp(self, cnda):
        """Test roundtrip: C++ → NumPy → C++ with intermediate deletions."""
        import gc
        
        arr_orig = cnda.ContiguousND_Cell2D([4, 4])
        arr_orig.set_u((2, 2), 7.5)
        arr_orig.set_v((2, 2), 8.5)
        arr_orig.set_flag((2, 2), 55)
        
        original_ptr = arr_orig.data_ptr()
        
        # C++ → NumPy (zero-copy)
        np_arr = arr_orig.to_numpy(copy=False)
        
        # Delete original C++
        del arr_orig
        gc.collect()
        
        # NumPy → C++ (zero-copy)
        arr_final = cnda.from_numpy_Cell2D(np_arr, copy=False)
        
        # Delete intermediate NumPy
        del np_arr
        gc.collect()
        
        # Final C++ should still be valid
        assert arr_final.get_u((2, 2)) == pytest.approx(7.5)
        assert arr_final.get_v((2, 2)) == pytest.approx(8.5)
        assert arr_final.get_flag((2, 2)) == 55
        
        # Should still point to same memory
        assert arr_final.data_ptr() == original_ptr
    
    def test_lifetime_with_modifications_after_deletion(self, cnda):
        """Test that modifications work correctly after source deletion."""
        import gc
        
        # Test Vec2f: C++ → NumPy
        arr = cnda.ContiguousND_Vec2f([3, 3])
        arr.set_x((0, 0), 1.0)
        np_view = arr.to_numpy(copy=False)
        del arr
        gc.collect()
        
        # Modify NumPy view after C++ deleted
        np_view[0, 0]['x'] = 100.0
        np_view[1, 1]['y'] = 200.0
        assert np_view[0, 0]['x'] == pytest.approx(100.0)
        assert np_view[1, 1]['y'] == pytest.approx(200.0)
        
        # Test Cell3D: NumPy → C++
        cell3d_dtype = np.dtype([('u', np.float32), ('v', np.float32), 
                                  ('w', np.float32), ('flag', np.int32)])
        np_arr = np.zeros((2, 2, 2), dtype=cell3d_dtype)
        np_arr[0, 0, 0]['u'] = 5.0
        cnda_view = cnda.from_numpy_Cell3D(np_arr, copy=False)
        del np_arr
        gc.collect()
        
        # Modify C++ view after NumPy deleted
        cnda_view.set_v((0, 0, 0), 50.0)
        cnda_view.set_flag((1, 1, 1), 999)
        assert cnda_view.get_v((0, 0, 0)) == pytest.approx(50.0)
        assert cnda_view.get_flag((1, 1, 1)) == 999
    
    def test_no_use_after_free_stress_test(self, cnda):
        """Stress test: rapid creation/deletion to detect use-after-free."""
        import gc
        
        for i in range(50):
            # Vec2f test
            arr = cnda.ContiguousND_Vec2f([10, 10])
            arr.set_x((5, 5), float(i))
            np_view = arr.to_numpy(copy=False)
            del arr
            assert np_view[5, 5]['x'] == pytest.approx(float(i))
            
            # Cell3D test
            cell3d_dtype = np.dtype([('u', np.float32), ('v', np.float32), 
                                      ('w', np.float32), ('flag', np.int32)])
            np_arr = np.zeros((5, 5), dtype=cell3d_dtype)
            np_arr[2, 2]['flag'] = i
            cnda_view = cnda.from_numpy_Cell3D(np_arr, copy=False)
            del np_arr
            assert cnda_view.get_flag((2, 2)) == i
            
            if i % 10 == 0:
                gc.collect()
        
        gc.collect()


