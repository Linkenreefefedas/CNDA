"""
Example 2: C++ Array → NumPy Export

Demonstrates creating arrays on the C++ side and exporting to NumPy.
Common pattern: allocate in C++, compute in C++, visualize in Python.
"""

import numpy as np
import cnda
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("Example 2: C++ Array → NumPy Export")
    print("=" * 60)
    
    # Create array on C++ side
    print("\n1. Create 100×100 array in C++")
    arr = cnda.ContiguousND_f32([100, 100])
    print(f"Created: {arr.shape} array")
    
    # Fill with data via NumPy (simulating C++ computation)
    data = arr.to_numpy(copy=False)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    data[:] = np.sin(np.sqrt(X**2 + Y**2))
    
    print("2. Filled with sin(r) pattern")
    print(f"Min value: {data.min():.3f}")
    print(f"Max value: {data.max():.3f}")
    
    # Export to NumPy (zero-copy)
    result = arr.to_numpy(copy=False)
    print("\n3. Exported to NumPy (zero-copy)")
    print(f"Same memory? {result.ctypes.data == data.ctypes.data}")
    
    # Visualization
    print("\n4. Visualizing...")
    plt.figure(figsize=(8, 6))
    plt.imshow(result, cmap='viridis', extent=[-5, 5, -5, 5], origin='lower')
    plt.colorbar(label='sin(r)')
    plt.title('C++ Array → NumPy Visualization')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('cpp_to_numpy_result.png', dpi=150, bbox_inches='tight')
    print("Saved to: cpp_to_numpy_result.png")
    
    # Copy for independent lifetime
    print("\n5. Create independent copy")
    independent = arr.to_numpy(copy=True)
    print(f"Different memory? {independent.ctypes.data != data.ctypes.data}")
    
    # Modify copy - original unchanged
    independent[0, 0] = -999.0
    print(f"\n6. Modified copy[0, 0] = -999.0")
    print(f"Original arr[0, 0] = {arr[0, 0]} (unchanged)")
    
    print("\n C++ arrays can be easily exported to NumPy!")

if __name__ == "__main__":
    main()
