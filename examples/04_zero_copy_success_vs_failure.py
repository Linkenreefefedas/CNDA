"""
Example 4: Zero-Copy Success vs Failure Cases

Demonstrates when zero-copy works and when it fails, with clear explanations.
Understanding these rules is crucial for efficient CNDA usage.
"""

import numpy as np
import cnda

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def main():
    print_section("Example 4: Zero-Copy Success vs Failure")
    
    # ============================================================
    # SUCCESS CASES
    # ============================================================
    
    print_section("SUCCESS CASE 1: Standard C-contiguous array")
    x1 = np.arange(12, dtype=np.float32).reshape(3, 4)
    print(f"Dtype: {x1.dtype}")
    print(f"C-contiguous: {x1.flags['C_CONTIGUOUS']}")
    print(f"Shape: {x1.shape}")
    
    try:
        arr1 = cnda.from_numpy(x1, copy=False)
        y1 = arr1.to_numpy(copy=False)
        print(f"Success! Same memory: {y1.ctypes.data == x1.ctypes.data}")
    except Exception as e:
        print(f"Failed: {e}")
    
    print_section("SUCCESS CASE 2: Explicit C-order specification")
    x2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order='C')
    print(f"Dtype: {x2.dtype}")
    print(f"C-contiguous: {x2.flags['C_CONTIGUOUS']}")
    
    try:
        arr2 = cnda.from_numpy(x2, copy=False)
        print("Success! Zero-copy achieved")
    except Exception as e:
        print(f" Failed: {e}")
    
    print_section("SUCCESS CASE 3: Supported int32 dtype")
    x3 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    print(f"Dtype: {x3.dtype}")
    
    try:
        arr3 = cnda.from_numpy(x3, copy=False)
        print("Success! int32 is supported")
    except Exception as e:
        print(f"Failed: {e}")
    
    # ============================================================
    # FAILURE CASES
    # ============================================================
    
    print_section("FAILURE CASE 1: Unsupported dtype (uint8)")
    x4 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    print(f"Dtype: {x4.dtype}")
    print("Supported dtypes: float32, float64, int32, int64")
    
    try:
        arr4 = cnda.from_numpy(x4, copy=False)
        print("Unexpected success!")
    except TypeError as e:
        print(f"Expected failure: {e}")
        print("→ Solution: Convert to supported dtype")
        x4_fixed = x4.astype(np.int32)
        arr4_fixed = cnda.from_numpy(x4_fixed, copy=False)
        print(f"Fixed: converted to {x4_fixed.dtype}")
    
    print_section(" FAILURE CASE 2: Fortran order (column-major)")
    x5 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='F')
    print(f"Dtype: {x5.dtype}")
    print(f"C-contiguous: {x5.flags['C_CONTIGUOUS']}")
    print(f"F-contiguous: {x5.flags['F_CONTIGUOUS']}")
    
    try:
        arr5 = cnda.from_numpy(x5, copy=False)
        print("Unexpected success!")
    except ValueError as e:
        print(f" Expected failure: {e}")
        print("→ Solution 1: Use copy=True")
        arr5_copy = cnda.from_numpy(x5, copy=True)
        print("Fixed: used explicit copy")
        
        print("→ Solution 2: Convert to C-order")
        x5_c = np.ascontiguousarray(x5)
        arr5_c = cnda.from_numpy(x5_c, copy=False)
        print(f"Fixed: made C-contiguous (but x5_c is already a copy)")
    
    print_section(" FAILURE CASE 3: Non-contiguous slice")
    x6 = np.arange(20, dtype=np.float32).reshape(5, 4)
    x6_slice = x6[::2, :]  # Every other row
    print(f"Original shape: {x6.shape}")
    print(f"Slice shape: {x6_slice.shape}")
    print(f"Slice C-contiguous: {x6_slice.flags['C_CONTIGUOUS']}")
    print(f"Slice strides: {x6_slice.strides} (non-standard!)")
    
    try:
        arr6 = cnda.from_numpy(x6_slice, copy=False)
        print("Unexpected success!")
    except ValueError as e:
        print(f" Expected failure: {e}")
        print("→ Solution 1: Use copy=True")
        arr6_copy = cnda.from_numpy(x6_slice, copy=True)
        print("Fixed: used explicit copy")
        
        print("→ Solution 2: Make contiguous first")
        x6_contig = np.ascontiguousarray(x6_slice)
        arr6_contig = cnda.from_numpy(x6_contig, copy=False)
        print("Fixed: made contiguous (but already a copy)")
    
    print_section("FAILURE CASE 4: Transposed array")
    x7 = np.arange(12, dtype=np.float32).reshape(3, 4)
    x7_T = x7.T  # Transpose
    print(f"Original shape: {x7.shape}, strides: {x7.strides}")
    print(f"Transposed shape: {x7_T.shape}, strides: {x7_T.strides}")
    print(f"Transposed C-contiguous: {x7_T.flags['C_CONTIGUOUS']}")
    
    try:
        arr7 = cnda.from_numpy(x7_T, copy=False)
        print("Unexpected success!")
    except ValueError as e:
        print(f"Expected failure: {e}")
        print("→ Solution: Use copy=True or make contiguous")
        arr7_copy = cnda.from_numpy(x7_T, copy=True)
        print("Fixed: used explicit copy")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    print_section("SUMMARY: Zero-Copy Requirements")
    print("""
Zero-copy requires ALL of these conditions:

1.  Supported dtype:
   - float32 (np.float32)
   - float64 (np.float64)
   - int32 (np.int32)
   - int64 (np.int64)

2.  C-contiguous layout:
   - flags['C_CONTIGUOUS'] must be True
   - Standard row-major strides
   - No gaps or non-standard spacing

3.  copy=False explicitly set
   - from_numpy(arr, copy=False)

If any condition fails:
   → Use copy=True to force layout conversion
   → Or convert array to supported format first
    """)
    
    print("Understanding these rules ensures efficient memory usage!")

if __name__ == "__main__":
    main()
