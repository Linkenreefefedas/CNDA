"""
Test suite for ownership and lifetime management in CNDA Python bindings.

This test suite validates capsule-based lifetime management including:
1. NumPy → CNDA zero-copy lifetime
2. CNDA → NumPy view lifetime
3. Capsule invalidation and reference counting
4. Early deallocation prevention
5. Multiple views referencing same buffer
6. Round-trip lifetime scenarios
7. Exception safety and cleanup
8. Double-free protection
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
# Capsule Ownership & Lifetime Safety Tests
# ==============================================================================

class TestCapsuleOwnershipFromNumpy:
    """Test capsule ownership for NumPy→C++ zero-copy buffers."""
    
    def test_numpy_array_deleted_before_access(self, np, cnda):
        """Test that C++ array remains valid after NumPy array is deleted."""
        x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        data_ptr = x.ctypes.data
        
        # Create zero-copy view
        arr = cnda.from_numpy_f32(x, copy=False)
        
        # Delete NumPy array
        del x
        
        # C++ array should still be accessible and valid
        assert arr[0] == 10.0
        assert arr[1] == 20.0
        assert arr[2] == 30.0
        assert arr.data_ptr() == data_ptr
    
    def test_garbage_collection_doesnt_invalidate_buffer(self, np, cnda):
        """Test that explicit garbage collection doesn't invalidate buffer."""
        import gc
        
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        data_ptr = x.ctypes.data
        
        # Create zero-copy view
        arr = cnda.from_numpy_f32(x, copy=False)
        
        # Delete NumPy array reference
        del x
        
        # Force garbage collection
        gc.collect()
        
        # C++ array should still be valid
        assert arr[0, 0] == 1.0
        assert arr[1, 1] == 4.0
        assert arr.data_ptr() == data_ptr
    
    def test_multiple_references_lifetime(self, np, cnda):
        """Test lifetime management with multiple references."""
        x = np.array([100.0, 200.0], dtype=np.float32)
        data_ptr = x.ctypes.data
        
        # Create multiple C++ views
        arr1 = cnda.from_numpy_f32(x, copy=False)
        arr2 = cnda.from_numpy_f32(x, copy=False)
        
        # Delete original NumPy array
        del x
        
        # Both C++ arrays should still be valid
        assert arr1[0] == 100.0
        assert arr2[1] == 200.0
        assert arr1.data_ptr() == data_ptr
        assert arr2.data_ptr() == data_ptr
    
    def test_capsule_protects_from_use_after_free(self, np, cnda):
        """Test that capsule mechanism prevents use-after-free."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        arr = cnda.from_numpy_f32(x, copy=False)
        
        # Delete NumPy array and trigger cleanup
        del x
        import gc
        gc.collect()
        
        # Accessing C++ array should not crash or exhibit undefined behavior
        # The capsule keeps the underlying NumPy data alive
        value = arr[1, 1]
        assert value == 4.0
    
    def test_zero_copy_to_numpy_lifetime(self, np, cnda):
        """Test that NumPy array keeps C++ ContiguousND alive via capsule."""
        arr = cnda.ContiguousND_f32([2, 3])
        arr[0, 0] = 1.5
        arr[1, 2] = 9.9
        data_ptr = arr.data_ptr()
        
        # Export to NumPy with zero-copy
        np_arr = arr.to_numpy(copy=False)
        
        # Delete C++ object reference
        del arr
        
        import gc
        gc.collect()
        
        # NumPy array should still be valid (capsule keeps C++ data alive)
        assert np_arr[0, 0] == 1.5
        assert np_arr[1, 2] == 9.9
        assert np_arr.ctypes.data == data_ptr
    
    def test_modified_numpy_persists_after_deletion(self, np, cnda):
        """Test that modifications to NumPy persist even after original array deleted."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr = cnda.from_numpy_f32(x, copy=False)
        
        # Modify via C++
        arr[0] = 99.0
        
        # Delete original NumPy reference
        del x
        
        import gc
        gc.collect()
        
        # Verify modification persists
        assert arr[0] == 99.0


class TestCapsuleOwnershipToCpp:
    """Test capsule ownership for C++→NumPy zero-copy buffers."""
    
    def test_cpp_object_deletion_keeps_numpy_valid(self, np, cnda):
        """Test that NumPy array remains valid after C++ object deleted."""
        arr = cnda.ContiguousND_f32([2, 2])
        arr[0, 0] = 42.0
        arr[1, 1] = 99.0
        
        # Export with zero-copy
        np_arr = arr.to_numpy(copy=False)
        data_ptr = np_arr.ctypes.data
        
        # Delete C++ object
        del arr
        
        import gc
        gc.collect()
        
        # NumPy should remain valid
        assert np_arr[0, 0] == 42.0
        assert np_arr[1, 1] == 99.0
        assert np_arr.ctypes.data == data_ptr
    
    def test_multiple_numpy_views_same_cpp_object(self, np, cnda):
        """Test multiple NumPy views of same C++ object."""
        arr = cnda.ContiguousND_f32([3, 3])
        arr[0, 0] = 1.0
        arr[2, 2] = 9.0
        
        # Create multiple NumPy views
        np_arr1 = arr.to_numpy(copy=False)
        np_arr2 = arr.to_numpy(copy=False)
        
        # Delete C++ reference
        del arr
        
        import gc
        gc.collect()
        
        # Both NumPy arrays should remain valid
        assert np_arr1[0, 0] == 1.0
        assert np_arr2[2, 2] == 9.0


# ==============================================================================
# Capsule Lifecycle & Refcount Tests
# ==============================================================================

class TestCapsuleLifecycle:
    """Test capsule invalidation and reference counting behavior."""
    
    def test_capsule_refcount_numpy_to_cnda(self, np, cnda):
        """Test reference counting for NumPy→CNDA zero-copy."""
        import sys
        
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        initial_refcount = sys.getrefcount(x)
        
        # Create zero-copy view - should increase refcount
        arr = cnda.from_numpy_f32(x, copy=False)
        refcount_with_view = sys.getrefcount(x)
        
        # Refcount should have increased (capsule holds reference)
        assert refcount_with_view > initial_refcount
        
        # Delete C++ view
        del arr
        import gc
        gc.collect()
        
        # Refcount should return to normal
        final_refcount = sys.getrefcount(x)
        assert final_refcount == initial_refcount
    
    def test_capsule_refcount_cnda_to_numpy(self, np, cnda):
        """Test reference counting for CNDA→NumPy zero-copy."""
        import sys
        
        arr = cnda.ContiguousND_f32([2, 3])
        arr[0, 0] = 5.0
        
        # Export to NumPy with zero-copy
        np_arr = arr.to_numpy(copy=False)
        
        # Both objects should be alive
        assert arr[0, 0] == 5.0
        assert np_arr[0, 0] == 5.0
        
        # Delete C++ object
        del arr
        import gc
        gc.collect()
        
        # NumPy array should keep data alive
        assert np_arr[0, 0] == 5.0
    
    def test_capsule_keeps_buffer_during_modification(self, np, cnda):
        """Test that capsule protects buffer during active modifications."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        arr = cnda.from_numpy_f32(x, copy=False)
        
        # Modify through C++ view
        arr[0, 0] = 99.0
        
        # Delete NumPy reference
        del x
        import gc
        gc.collect()
        
        # Continue modifications
        arr[1, 1] = 88.0
        
        # Values should persist
        assert arr[0, 0] == 99.0
        assert arr[1, 1] == 88.0


# ==============================================================================
# Round-Trip Lifetime Tests
# ==============================================================================

class TestRoundTripLifetime:
    """Test multiple conversion cycles and round-trip scenarios."""
    
    def test_numpy_cnda_numpy_roundtrip(self, np, cnda):
        """Test NumPy→CNDA→NumPy full cycle maintains data integrity."""
        # Original NumPy array
        x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        original_data = x1.copy()
        
        # NumPy → CNDA (zero-copy)
        arr = cnda.from_numpy_f32(x1, copy=False)
        
        # CNDA → NumPy (zero-copy)
        x2 = arr.to_numpy(copy=False)
        
        # Verify data integrity
        assert np.array_equal(x2, original_data)
        
        # Delete intermediate references
        del x1, arr
        import gc
        gc.collect()
        
        # Final NumPy array should still be valid
        assert x2[0, 0] == 1.0
        assert x2[1, 1] == 4.0
    
    def test_cnda_numpy_cnda_roundtrip(self, np, cnda):
        """Test CNDA→NumPy→CNDA full cycle."""
        # Original CNDA array
        arr1 = cnda.ContiguousND_f32([2, 3])
        arr1[0, 0] = 10.0
        arr1[1, 2] = 20.0
        
        # CNDA → NumPy (zero-copy)
        np_arr = arr1.to_numpy(copy=False)
        
        # NumPy → CNDA (zero-copy)
        arr2 = cnda.from_numpy_f32(np_arr, copy=False)
        
        # Verify data integrity
        assert arr2[0, 0] == 10.0
        assert arr2[1, 2] == 20.0
        
        # Delete intermediate references
        del arr1, np_arr
        import gc
        gc.collect()
        
        # Final CNDA array should still be valid
        assert arr2[0, 0] == 10.0
        assert arr2[1, 2] == 20.0
    
    def test_multiple_roundtrips(self, np, cnda):
        """Test 3+ conversion cycles maintain data integrity."""
        # Start with NumPy
        data = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        
        # Cycle 1: NumPy → CNDA
        arr1 = cnda.from_numpy_f32(data, copy=False)
        assert arr1[0] == 100.0
        
        # Cycle 2: CNDA → NumPy
        data2 = arr1.to_numpy(copy=False)
        assert data2[1] == 200.0
        
        # Cycle 3: NumPy → CNDA
        arr2 = cnda.from_numpy_f32(data2, copy=False)
        assert arr2[2] == 300.0
        
        # Cycle 4: CNDA → NumPy
        data3 = arr2.to_numpy(copy=False)
        assert data3[0] == 100.0
        
        # Clean up intermediate references
        del data, arr1, data2, arr2
        import gc
        gc.collect()
        
        # Final array should still be valid
        assert data3[0] == 100.0
        assert data3[1] == 200.0
        assert data3[2] == 300.0
    
    def test_roundtrip_with_modifications(self, np, cnda):
        """Test round-trip with modifications at each stage."""
        # Start with NumPy
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        # NumPy → CNDA
        arr = cnda.from_numpy_f32(x, copy=False)
        arr[0] = 10.0  # Modify in CNDA
        
        # CNDA → NumPy
        y = arr.to_numpy(copy=False)
        y[1] = 20.0  # Modify in NumPy
        
        # NumPy → CNDA
        arr2 = cnda.from_numpy_f32(y, copy=False)
        arr2[2] = 30.0  # Modify in CNDA again
        
        # Verify all modifications persisted
        final = arr2.to_numpy(copy=False)
        assert final[0] == 10.0
        assert final[1] == 20.0
        assert final[2] == 30.0


# ==============================================================================
# Memory Leak Prevention & Stress Tests
# ==============================================================================

class TestMemoryLeakPrevention:
    """Stress tests for memory leak detection and prevention."""
    
    def test_repeated_numpy_to_cnda_conversions(self, np, cnda):
        """Test repeated NumPy→CNDA conversions don't leak memory."""
        import gc
        
        for i in range(100):
            x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            arr = cnda.from_numpy_f32(x, copy=False)
            assert arr[0, 0] == 1.0
            del x, arr
            
            if i % 10 == 0:
                gc.collect()
        
        # Final cleanup
        gc.collect()
    
    def test_repeated_cnda_to_numpy_conversions(self, np, cnda):
        """Test repeated CNDA→NumPy conversions don't leak memory."""
        import gc
        
        for i in range(100):
            arr = cnda.ContiguousND_f32([3, 3])
            arr[0, 0] = float(i)
            np_arr = arr.to_numpy(copy=False)
            assert np_arr[0, 0] == float(i)
            del arr, np_arr
            
            if i % 10 == 0:
                gc.collect()
        
        # Final cleanup
        gc.collect()
    
    def test_repeated_roundtrip_conversions(self, np, cnda):
        """Test repeated round-trip conversions don't leak memory."""
        import gc
        
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        for i in range(50):
            arr = cnda.from_numpy_f32(x, copy=False)
            x = arr.to_numpy(copy=False)
            
            if i % 10 == 0:
                gc.collect()
        
        assert x[0] == 1.0
        gc.collect()
    
    def test_large_array_cleanup(self, np, cnda):
        """Test that large arrays are properly cleaned up."""
        import gc
        
        # Create and delete large array multiple times
        for _ in range(10):
            x = np.zeros((1000, 1000), dtype=np.float32)
            arr = cnda.from_numpy_f32(x, copy=True)
            del x, arr
            gc.collect()
    
    def test_exception_during_conversion_cleanup(self, np, cnda):
        """Test that resources are cleaned up when conversion fails."""
        import gc
        
        # Test with unsupported dtype
        try:
            x = np.array([[1, 2]], dtype=np.uint8)
            arr = cnda.from_numpy(x, copy=False)
        except TypeError:
            pass  # Expected
        
        gc.collect()
        
        # Test with non-contiguous array
        try:
            x = np.array([[1.0, 2.0]], dtype=np.float32, order='F')
            arr = cnda.from_numpy_f32(x, copy=False)
        except ValueError:
            pass  # Expected
        
        gc.collect()


# ==============================================================================
# Double-Free Protection Tests
# ==============================================================================

class TestDoubleFreeProtection:
    """Test double-free prevention in complex ownership scenarios."""
    
    def test_overlapping_views_no_double_free(self, np, cnda):
        """Test multiple overlapping views don't cause double-free."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        
        # Create multiple C++ views of same data
        arr1 = cnda.from_numpy_f32(x, copy=False)
        arr2 = cnda.from_numpy_f32(x, copy=False)
        arr3 = cnda.from_numpy_f32(x, copy=False)
        
        # All views should be valid
        assert arr1[0] == 1.0
        assert arr2[1] == 2.0
        assert arr3[2] == 3.0
        
        # Delete in various orders
        del arr2
        assert arr1[0] == 1.0
        assert arr3[2] == 3.0
        
        del arr1
        assert arr3[3] == 4.0
        
        del arr3, x
        import gc
        gc.collect()
    
    def test_cross_referenced_views_cleanup(self, np, cnda):
        """Test cleanup of cross-referenced views."""
        # NumPy → CNDA → NumPy → CNDA
        x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        arr1 = cnda.from_numpy_f32(x1, copy=False)
        x2 = arr1.to_numpy(copy=False)
        arr2 = cnda.from_numpy_f32(x2, copy=False)
        
        # Delete in reverse order
        del arr2
        import gc
        gc.collect()
        
        assert x2[0, 0] == 1.0
        
        del x2
        gc.collect()
        
        assert arr1[0, 0] == 1.0
        
        del arr1
        gc.collect()
        
        assert x1[0, 0] == 1.0
    
    def test_circular_reference_cleanup(self, np, cnda):
        """Test cleanup when circular references exist."""
        x = np.array([10.0, 20.0], dtype=np.float32)
        arr = cnda.from_numpy_f32(x, copy=False)
        
        # Create circular-like reference pattern
        y = arr.to_numpy(copy=False)
        arr2 = cnda.from_numpy_f32(y, copy=False)
        
        # Delete all references
        del x, arr, y, arr2
        
        import gc
        gc.collect()
        
        # Should not cause any issues (no double-free or leaks)


# ==============================================================================
# Exception Safety Tests
# ==============================================================================

class TestExceptionSafety:
    """Test resource cleanup during exception scenarios."""
    
    def test_exception_in_from_numpy_dtype_error(self, np, cnda):
        """Test cleanup when from_numpy raises TypeError."""
        import gc
        
        x = np.array([[1, 2]], dtype=np.uint16)
        initial_refcount = None
        
        try:
            import sys
            initial_refcount = sys.getrefcount(x)
            arr = cnda.from_numpy(x, copy=False)
            assert False, "Should have raised TypeError"
        except TypeError as e:
            assert "Unsupported dtype" in str(e)
        
        gc.collect()
        
        # Refcount should be back to normal
        if initial_refcount is not None:
            import sys
            assert sys.getrefcount(x) == initial_refcount
    
    def test_exception_in_from_numpy_layout_error(self, np, cnda):
        """Test cleanup when from_numpy raises ValueError."""
        import gc
        
        x = np.array([[1.0, 2.0]], dtype=np.float32, order='F')
        
        try:
            arr = cnda.from_numpy_f32(x, copy=False)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            # Check for either error message variant
            assert "C-contiguous" in str(e) or "strides" in str(e)
        
        gc.collect()
        
        # NumPy array should still be valid
        assert x[0, 0] == 1.0
    
    def test_no_leak_on_repeated_exceptions(self, np, cnda):
        """Test no memory leak when exceptions raised repeatedly."""
        import gc
        
        for _ in range(50):
            try:
                x = np.array([[1, 2]], dtype=np.uint8)
                arr = cnda.from_numpy(x, copy=False)
            except TypeError:
                pass
            
            try:
                x = np.array([[1.0, 2.0]], dtype=np.float32, order='F')
                arr = cnda.from_numpy_f32(x, copy=False)
            except ValueError:
                pass
        
        gc.collect()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
