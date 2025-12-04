"""
Example 1: Python Round-Trip (NumPy ↔ CNDA ↔ NumPy)

Demonstrates zero-copy memory sharing between NumPy and CNDA.
This is the most basic and common use case.
"""

import numpy as np
import cnda

def main():
    print("=" * 60)
    print("Example 1: Python Round-Trip (Zero-Copy)")
    print("=" * 60)
    
    # Create NumPy array
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    print("\n1. Original NumPy array:")
    print(x)
    print(f"Memory address: {hex(x.ctypes.data)}")
    
    # NumPy → CNDA (zero-copy)
    arr = cnda.from_numpy(x, copy=False)
    print("\n2. Converted to CNDA (zero-copy)")
    print(f"Shape: {arr.shape}")
    print(f"Strides: {arr.strides}")
    
    # Modify via CNDA
    arr[1, 2] = 999.0
    print("\n3. Modified arr[1, 2] = 999.0 via CNDA")
    
    # Change is visible in NumPy (same memory!)
    print(f"NumPy x[1, 2] = {x[1, 2]} (changed!)")
    
    # CNDA → NumPy (zero-copy)
    y = arr.to_numpy(copy=False)
    print("\n4. Converted back to NumPy (zero-copy)")
    print(f"Memory address: {hex(y.ctypes.data)}")
    print(f"Same memory? {y.ctypes.data == x.ctypes.data}")
    
    # Modify via NumPy
    y[0, 0] = 111.0
    print("\n5. Modified y[0, 0] = 111.0 via NumPy")
    print(f"CNDA arr[0, 0] = {arr[0, 0]} (changed!)")
    
    print("\n All three objects share the same memory!")
    print("x, arr, and y point to the same buffer")

if __name__ == "__main__":
    main()
