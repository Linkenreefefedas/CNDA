"""
Example 5: Error Handling Best Practices

Demonstrates how to handle CNDA errors gracefully and implement
robust error handling in production code.
"""

import numpy as np
import cnda

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def safe_from_numpy(arr, copy=False, auto_fix=True):
    """
    Wrapper around cnda.from_numpy with automatic error handling.
    
    Parameters
    ----------
    arr : numpy.ndarray
        Input array
    copy : bool
        Whether to force copy
    auto_fix : bool
        If True, automatically fix common issues
    
    Returns
    -------
    cnda.ContiguousND_*
        CNDA array
    """
    try:
        # First attempt: strict zero-copy or copy as requested
        return cnda.from_numpy(arr, copy=copy)
        
    except TypeError as e:
        # Unsupported dtype
        print(f"TypeError: {e}")
        
        if auto_fix:
            # Try to convert to supported dtype
            if arr.dtype == np.uint8:
                print("→ Auto-fix: Converting uint8 to int32")
                arr = arr.astype(np.int32)
            elif arr.dtype == np.float16:
                print("→ Auto-fix: Converting float16 to float32")
                arr = arr.astype(np.float32)
            elif arr.dtype in [np.uint16, np.uint32, np.uint64]:
                print(f"→ Auto-fix: Converting {arr.dtype} to int64")
                arr = arr.astype(np.int64)
            else:
                print(f"→ Cannot auto-fix dtype: {arr.dtype}")
                raise
            
            return cnda.from_numpy(arr, copy=copy)
        else:
            raise
    
    except ValueError as e:
        # Layout mismatch (non-contiguous, Fortran order, etc.)
        print(f"ValueError: {e}")
        
        if auto_fix and not copy:
            print("→ Auto-fix: Forcing copy=True for layout conversion")
            return cnda.from_numpy(arr, copy=True)
        else:
            raise

def main():
    print_section("Example 5: Error Handling Best Practices")
    
    # ============================================================
    # Error Type 1: TypeError (Unsupported dtype)
    # ============================================================
    
    print_section("Error Type 1: TypeError (Unsupported dtype)")
    
    print("\nAttempt 1: uint8 array (unsupported)")
    x1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    print(f"Input dtype: {x1.dtype}")
    
    try:
        arr1 = cnda.from_numpy(x1, copy=False)
        print("Unexpected success!")
    except TypeError as e:
        print(f" Caught TypeError: {e}")
        print("Common causes:")
        print("- uint8, uint16, uint32, uint64")
        print("- float16, complex64, complex128")
        print("- string, object dtypes")
    
    print("\nManual fix: Convert to supported dtype")
    x1_fixed = x1.astype(np.int32)
    arr1_fixed = cnda.from_numpy(x1_fixed, copy=False)
    print(f"Success with dtype={x1_fixed.dtype}")
    
    print("\nAutomatic fix using wrapper:")
    arr1_auto = safe_from_numpy(x1, copy=False, auto_fix=True)
    print("Success with auto-fix")
    
    # ============================================================
    # Error Type 2: ValueError (Layout mismatch)
    # ============================================================
    
    print_section("Error Type 2: ValueError (Layout mismatch)")
    
    print("\nAttempt 1: Fortran-order array")
    x2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='F')
    print(f"C-contiguous: {x2.flags['C_CONTIGUOUS']}")
    print(f"F-contiguous: {x2.flags['F_CONTIGUOUS']}")
    
    try:
        arr2 = cnda.from_numpy(x2, copy=False)
        print("Unexpected success!")
    except ValueError as e:
        print(f"Caught ValueError: {e}")
        print("Common causes:")
        print("- Fortran order (column-major)")
        print("- Non-contiguous slices")
        print("- Transposed arrays")
        print("- Non-standard strides")
    
    print("\nManual fix: Use copy=True")
    arr2_fixed = cnda.from_numpy(x2, copy=True)
    print("Success with copy=True")
    
    print("\nAutomatic fix using wrapper:")
    arr2_auto = safe_from_numpy(x2, copy=False, auto_fix=True)
    print(" Success with auto-fix")
    
    print("\nAttempt 2: Non-contiguous slice")
    x3 = np.arange(20, dtype=np.float32).reshape(5, 4)
    x3_slice = x3[::2, :]
    print(f"Slice shape: {x3_slice.shape}")
    print(f"C-contiguous: {x3_slice.flags['C_CONTIGUOUS']}")
    
    try:
        arr3 = cnda.from_numpy(x3_slice, copy=False)
        print("Unexpected success!")
    except ValueError as e:
        print(f"Caught ValueError: {e}")
    
    print("\nAutomatic fix:")
    arr3_auto = safe_from_numpy(x3_slice, copy=False, auto_fix=True)
    print("Success with auto-fix")
    
    # ============================================================
    # Error Type 3: IndexError (Out of bounds)
    # ============================================================
    
    print_section("Error Type 3: IndexError (Bounds checking)")
    
    arr = cnda.ContiguousND_f32([3, 4])
    print(f"Array shape: {arr.shape}")
    
    print("\nAttempt 1: Valid access")
    try:
        arr[2, 3] = 42.0
        print(f"arr[2, 3] = {arr[2, 3]} (valid)")
    except IndexError as e:
        print(f"IndexError: {e}")
    
    print("\nAttempt 2: Out of bounds")
    try:
        val = arr[3, 0]  # Index 3 >= shape[0]=3
        print(f"Unexpected success: {val}")
    except IndexError as e:
        print(f"Caught IndexError: {e}")
        print("Cause: Index 3 is out of bounds for axis 0 (size 3)")
    
    print("\nAttempt 3: Wrong number of indices")
    try:
        val = arr[0]  # Only 1 index for 2D array
        print(f"Unexpected success: {val}")
    except IndexError as e:
        print(f"Caught IndexError: {e}")
        print("Cause: 2D array requires 2 indices")
    
    print("\nSafe access with bounds checking:")
    def safe_access(arr, i, j):
        try:
            return arr[i, j]
        except IndexError as e:
            print(f"IndexError: {e}")
            return None
    
    val = safe_access(arr, 2, 3)
    print(f"safe_access(arr, 2, 3) = {val}")
    
    val = safe_access(arr, 3, 0)
    print(f"safe_access(arr, 3, 0) = {val} (handled)")
    
    # ============================================================
    # Error Type 4: RuntimeError (Memory/lifetime issues)
    # ============================================================
    
    print_section("Error Type 4: RuntimeError (Memory issues)")
    
    print("\nNote: RuntimeError typically occurs with:")
    print("- Corrupted capsule objects")
    print("- Invalid memory access")
    print("- Lifetime management issues")
    print("\nThese are rare in normal usage due to automatic lifetime management.")
    
    # ============================================================
    # Production-Ready Error Handler
    # ============================================================
    
    print_section("Production-Ready Error Handler Example")
    
    def robust_from_numpy(arr, copy=False, verbose=True):
        """
        Production-ready wrapper with comprehensive error handling.
        """
        if verbose:
            print(f"\nProcessing array: shape={arr.shape}, dtype={arr.dtype}")
        
        # Check 1: Dtype compatibility
        supported_dtypes = [np.float32, np.float64, np.int32, np.int64]
        if arr.dtype not in supported_dtypes:
            if verbose:
                print(f"Unsupported dtype: {arr.dtype}")
            
            # Auto-convert to nearest supported type
            if arr.dtype in [np.uint8, np.uint16, np.int16]:
                arr = arr.astype(np.int32)
                if verbose:
                    print(f"→ Converted to int32")
            elif arr.dtype in [np.uint32, np.int64]:
                arr = arr.astype(np.int64)
                if verbose:
                    print(f"→ Converted to int64")
            elif arr.dtype == np.float16:
                arr = arr.astype(np.float32)
                if verbose:
                    print(f"→ Converted to float32")
            else:
                raise TypeError(f"Cannot convert {arr.dtype} to supported type")
        
        # Check 2: Contiguity
        if not copy and not arr.flags['C_CONTIGUOUS']:
            if verbose:
                print("Array is not C-contiguous")
                print("→ Forcing copy=True")
            copy = True
        
        # Attempt conversion
        try:
            result = cnda.from_numpy(arr, copy=copy)
            if verbose:
                print(f"Success! copy={copy}")
            return result
        except Exception as e:
            if verbose:
                print(f"Failed: {e}")
            raise
    
    # Test the robust handler
    print("\nTest 1: uint8 non-contiguous slice")
    test1 = np.arange(100, dtype=np.uint8).reshape(10, 10)[::2, :]
    result1 = robust_from_numpy(test1, copy=False, verbose=True)
    
    print("\nTest 2: float16 Fortran-order")
    test2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16, order='F')
    result2 = robust_from_numpy(test2, copy=False, verbose=True)
    
    print("\nTest 3: Valid float32 C-contiguous")
    test3 = np.arange(12, dtype=np.float32).reshape(3, 4)
    result3 = robust_from_numpy(test3, copy=False, verbose=True)
    
    # ============================================================
    # Summary
    # ============================================================
    
    print_section("Error Handling Summary")
    print("""
Exception Types:
  1. TypeError      → Unsupported dtype
  2. ValueError     → Layout/contiguity mismatch
  3. IndexError     → Out-of-bounds access or rank mismatch
  4. RuntimeError   → Memory/lifetime issues (rare)

Best Practices:
   Always use try-except when copy=False
   Check dtype before conversion
   Check C-contiguity for zero-copy
   Use wrapper functions for auto-fixing
   Validate indices before access
   Provide clear error messages

Production Checklist:
  1. Validate input dtype
  2. Check array contiguity
  3. Handle errors gracefully
  4. Log issues for debugging
  5. Provide fallback behavior
    """)
    
    print("Robust error handling ensures reliable production code!")

if __name__ == "__main__":
    main()
