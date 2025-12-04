# CNDA Quickstart Guide

> **Get started with CNDA in 5 minutes**

## What is CNDA?

**CNDA** (Contiguous N-Dimensional Array) is a lightweight C++11/Python library that provides cache-friendly multi-dimensional arrays with **zero-copy NumPy interoperability** and explicit memory ownership semantics.

---

## When to Use CNDA

### CNDA is Great For:

- **C++/Python interoperability** - Seamlessly move data between C++ and Python with explicit copy/zero-copy control
- **Predictable memory behavior** - Know exactly when data is copied vs. shared
- **Structured types (AoS)** - Store multiple values per grid point (e.g., velocity + pressure)
- **Minimal dependencies** - Header-only C++ core with no external dependencies
- **Scientific computing pipelines** - When you need both C++ performance and Python convenience
- **Cache-friendly layouts** - Row-major contiguous memory for optimal CPU cache usage

### When NOT to Use CNDA

- **Pure NumPy workflows** - If you're only using Python, stick with NumPy
- **Complex linear algebra** - Use Eigen, BLAS, or NumPy's linalg
- **GPU computing** - Use CuPy, PyTorch, or TensorFlow
- **Production data science** - Use Pandas, Polars, or Dask for tabular data
- **Need advanced array operations** - CNDA focuses on memory layout and interop, not computation

**Bottom line**: CNDA is infrastructure, not a computation library. Use it when you need reliable data exchange between C++ and Python with clear ownership rules.

---

## 5-Minute Tutorial

### Installation

**Prerequisites**: Python 3.9+, CMake 3.18+, C++11 compiler

```bash
# Quick install
pip install .

# Or for development
pip install -e .
```

See [INSTALLATION.md](INSTALLATION.md) for detailed instructions.

**Quick verification:**
```bash
python -c "import cnda; print('CNDA version:', cnda.__version__)"
```

---

### Python Quickstart

#### Example 1: Create and Access Arrays

```python
import cnda

# Create a 3x4 float32 array
arr = cnda.ContiguousND_f32([3, 4])

# Set values using tuple indexing
arr[0, 0] = 1.0
arr[1, 2] = 42.5
arr[2, 3] = 99.9

# Get values
print(arr[1, 2])  # Output: 42.5

# Inspect metadata
print(f"Shape: {arr.shape}")      # (3, 4)
print(f"Strides: {arr.strides}")  # (4, 1) - row-major
print(f"Dimensions: {arr.ndim}")  # 2
print(f"Total size: {arr.size}")  # 12
```

#### Example 2: Zero-Copy NumPy Interop

```python
import numpy as np
import cnda

# NumPy → CNDA (zero-copy)
x = np.arange(12, dtype=np.float32).reshape(3, 4)
arr = cnda.from_numpy(x, copy=False)

# Modify via CNDA
arr[1, 2] = 999.0

# Changes are visible in NumPy (same memory!)
print(x[1, 2])  # Output: 999.0

# CNDA → NumPy (zero-copy)
y = arr.to_numpy(copy=False)
print(y.ctypes.data == x.ctypes.data)  # True - same buffer!
```

#### Example 3: Different Data Types

```python
import cnda

# Available types: f32, f64, i32, i64
f32 = cnda.ContiguousND_f32([2, 3])     # float (32-bit)
f64 = cnda.ContiguousND_f64([2, 3])     # double (64-bit)
i32 = cnda.ContiguousND_i32([2, 3])     # int32
i64 = cnda.ContiguousND_i64([2, 3])     # int64
```

#### Example 4: Working with Structured Types (AoS)

```python
import numpy as np
import cnda

# Define a structured dtype (Array-of-Structs)
cell_dtype = np.dtype([
    ('u', '<f4'),      # velocity x
    ('v', '<f4'),      # velocity y
    ('flag', '<i4')    # status flag
], align=True)

# Create structured array
grid = np.zeros((100, 100), dtype=cell_dtype, order='C')

# Zero-copy to CNDA (if layout matches)
cnda_grid = cnda.from_numpy(grid, copy=False)

# Export back to NumPy
out = cnda_grid.to_numpy(copy=False)

# Access fields
out['u'][50, 50] = 1.5
out['v'][50, 50] = 2.0
out['flag'][50, 50] = 1
```

---

### C++ Quickstart

The C++ core is **header-only** - just include the headers!

#### Example 1: Basic Usage

```cpp
#include "cnda/contiguous_nd.hpp"
#include <iostream>

int main() {
    // Create a 3x4 float array
    cnda::ContiguousND<float> arr({3, 4});
    
    // Set values using operator()
    arr(0, 0) = 1.0f;
    arr(1, 2) = 42.5f;
    arr(2, 3) = 99.9f;
    
    // Get values
    std::cout << "arr(1, 2) = " << arr(1, 2) << "\n";  // 42.5
    
    // Inspect metadata
    std::cout << "ndim: " << arr.ndim() << "\n";       // 2
    std::cout << "size: " << arr.size() << "\n";       // 12
    std::cout << "shape: (" << arr.shape()[0] << ", " 
              << arr.shape()[1] << ")\n";              // (3, 4)
    
    return 0;
}
```

**Compile:**
```bash
# Linux/macOS
g++ -std=c++11 -I./include example.cpp -o example

# Windows (MSVC)
cl /std:c++11 /EHsc /I.\include example.cpp
```

#### Example 2: Multi-Dimensional Arrays

```cpp
#include "cnda/contiguous_nd.hpp"

int main() {
    // 1D array
    cnda::ContiguousND<double> vec({10});
    vec(5) = 3.14;
    
    // 3D array
    cnda::ContiguousND<int32_t> cube({2, 3, 4});
    cube(1, 2, 3) = 123;
    
    // 4D array
    cnda::ContiguousND<int64_t> hyper({2, 3, 4, 5});
    hyper(1, 2, 3, 4) = 9999;
    
    return 0;
}
```

#### Example 3: Iteration and Raw Access

```cpp
#include "cnda/contiguous_nd.hpp"
#include <algorithm>
#include <numeric>

int main() {
    cnda::ContiguousND<float> arr({100, 200});
    
    // Raw pointer access (for performance)
    float* data = arr.data();
    std::size_t total_size = arr.size();
    
    // Fill with zeros
    std::fill(data, data + total_size, 0.0f);
    
    // Or use standard algorithms
    std::iota(data, data + total_size, 0.0f);  // 0, 1, 2, ...
    
    return 0;
}
```

#### Example 4: Structured Types (AoS)

```cpp
#include "cnda/aos_types.hpp"
#include <iostream>

int main() {
    // Define a struct (must be POD)
    struct Cell2D {
        float u;       // velocity x
        float v;       // velocity y
        int32_t flag;  // status
    };
    
    // Create array of structs
    cnda::ContiguousND<Cell2D> grid({100, 100});
    
    // Access and modify
    grid(50, 50).u = 1.5f;
    grid(50, 50).v = 2.0f;
    grid(50, 50).flag = 1;
    
    // Verify
    std::cout << "Cell(50,50): u=" << grid(50, 50).u 
              << " v=" << grid(50, 50).v
              << " flag=" << grid(50, 50).flag << "\n";
    
    return 0;
}
```

---

## Core Concepts

### Shape and Strides

**Shape** defines the dimensions of the array:
```python
arr = cnda.ContiguousND_f32([3, 4, 5])
print(arr.shape)  # (3, 4, 5)
```

**Strides** define how many elements to skip to move along each dimension:
```python
print(arr.strides)  # (20, 5, 1) for row-major layout
```

**Why strides matter**: They enable efficient indexing without computing offsets repeatedly.

```python
# Accessing arr[i, j, k] computes offset as:
offset = i * strides[0] + j * strides[1] + k * strides[2]
       = i * 20         + j * 5          + k * 1
```

### Row-Major Layout

CNDA uses **row-major (C-style) layout**, same as NumPy's default:

```
Shape: (3, 4)
Memory: [0,0] [0,1] [0,2] [0,3] [1,0] [1,1] [1,2] [1,3] [2,0] [2,1] [2,2] [2,3]
Index:    0     1     2     3     4     5     6     7     8     9    10    11
```

**Why row-major?**
- CPU cache-friendly for most access patterns
- Compatible with NumPy's default
- Natural for C/C++ nested loops: `for i, for j, for k`

### Zero-Copy vs Copy

**Zero-copy** shares the same memory buffer:
```python
x = np.array([[1, 2], [3, 4]], dtype=np.float32)
arr = cnda.from_numpy(x, copy=False)  # Points to same memory
```

**Copy** creates independent memory:
```python
arr = cnda.from_numpy(x, copy=True)  # New memory allocated
```

**When is zero-copy safe?**
1. Dtype matches (e.g., `float32` → `ContiguousND_f32`)
2. Array is C-contiguous (row-major)
3. Lifetime is managed (pybind11 keeps producer alive)

**When to use `copy=True`?**
- Need independent lifetime (producer will be deleted)
- Array layout doesn't match (Fortran-order, non-contiguous)
- Modifying one shouldn't affect the other

---

## Common Patterns

### Pattern 1: C++ Computation, Python Visualization

```python
# C++ does heavy computation, Python visualizes
import cnda
import matplotlib.pyplot as plt

# Allocate on C++ side
result = cnda.ContiguousND_f64([512, 512])

# ... C++ fills data via bindings (not shown) ...

# Zero-copy to NumPy for plotting
data = result.to_numpy(copy=False)
plt.imshow(data, cmap='viridis')
plt.colorbar()
plt.show()
```

### Pattern 2: NumPy Preprocessing, C++ Processing

```python
import numpy as np
import cnda

# Prepare data in NumPy
data = np.random.randn(1000, 1000).astype(np.float32)
data = data / data.max()  # Normalize

# Zero-copy to C++ for processing
cnda_data = cnda.from_numpy(data, copy=False)

# ... Pass to C++ algorithm (not shown) ...

# Result stays in same buffer
result = cnda_data.to_numpy(copy=False)
```

### Pattern 3: Structured Data Processing

```python
import numpy as np
import cnda

# Particle system: position + velocity + mass
particle_dtype = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),   # position
    ('vx', '<f4'), ('vy', '<f4'), ('vz', '<f4'), # velocity
    ('mass', '<f4')
], align=True)

particles = np.zeros(10000, dtype=particle_dtype)
particles['mass'] = 1.0

# Zero-copy to C++ for physics update
cnda_particles = cnda.from_numpy(particles, copy=False)

# ... C++ updates positions and velocities ...

# Changes visible in original NumPy array
print(particles['x'][:5])  # Updated positions
```

---

## Error Handling

CNDA provides clear error messages:

```python
import numpy as np
import cnda

# TypeError: Unsupported dtype
try:
    x = np.array([[1, 2]], dtype=np.uint8)
    arr = cnda.from_numpy(x)
except TypeError as e:
    print(e)  # "Unsupported dtype: uint8"

# ValueError: Layout mismatch
try:
    x = np.array([[1.0, 2.0]], dtype=np.float32, order='F')  # Fortran
    arr = cnda.from_numpy(x, copy=False)
except ValueError as e:
    print(e)  # "from_numpy with copy=False requires C-contiguous array"

# IndexError: Out of bounds
try:
    arr = cnda.ContiguousND_f32([3, 4])
    val = arr[3, 0]  # Index 3 is out of bounds
except IndexError as e:
    print(e)  # "Index out of bounds"
```

---

## Performance Tips

### Do's

1. **Use zero-copy when possible**
   ```python
   arr = cnda.from_numpy(x, copy=False)  # Fast
   ```

2. **Preallocate arrays**
   ```cpp
   cnda::ContiguousND<float> result({1000, 1000});  // Reserve memory once
   ```

3. **Contiguous access patterns**
   ```cpp
   // Good: row-major iteration
   for (size_t i = 0; i < rows; ++i)
       for (size_t j = 0; j < cols; ++j)
           arr(i, j) = compute(i, j);
   ```

4. **Use raw pointers for tight loops**
   ```cpp
   float* data = arr.data();
   for (size_t i = 0; i < arr.size(); ++i)
       data[i] *= 2.0f;  // Faster than arr(...)
   ```

### Don'ts

1. **Avoid unnecessary copies**
   ```python
   arr = cnda.from_numpy(x, copy=True)  # Slow if not needed
   ```

2. **Don't use Python loops for computation**
   ```python
   # Bad: Python loop is slow
   for i in range(rows):
       for j in range(cols):
           arr[i, j] = compute(i, j)
   
   # Good: Use NumPy or C++
   data = arr.to_numpy(copy=False)
   data[:] = vectorized_compute(data)
   ```

3. **Avoid column-major iteration**
   ```cpp
   // Bad: cache-unfriendly
   for (size_t j = 0; j < cols; ++j)
       for (size_t i = 0; i < rows; ++i)
           arr(i, j) = ...
   ```

---

## Next Steps

### Learn More

**Documentation:**
- **[INSTALLATION.md](INSTALLATION.md)** - Detailed installation guide for all platforms
- **[PYTHON_USER_GUIDE.md](PYTHON_USER_GUIDE.md)** - Complete Python API reference and best practices
- **[CPP_USER_GUIDE.md](CPP_USER_GUIDE.md)** - Complete C++ API reference and performance tips
- **[BENCHMARKING.md](BENCHMARKING.md)** - Performance benchmarks and optimization guide

**Code Examples:**
- **[examples/](../examples/)** - Runnable Python examples:
  - `01_python_roundtrip.py` - Zero-copy NumPy ↔ CNDA basics
  - `02_cpp_to_numpy.py` - C++ array export for visualization
  - `03_aos_struct.py` - Structured types for fluid simulation
  - `04_zero_copy_success_vs_failure.py` - When zero-copy works/fails
  - `05_error_handling.py` - Robust error handling patterns

### Run Examples

```bash
# Run interactive examples (recommended for learning)
cd examples
python 01_python_roundtrip.py
python 02_cpp_to_numpy.py
python 03_aos_struct.py

# Run test suite
cd tests/python
pytest test_python_bindings.py -v

# Run benchmarks (requires pytest-benchmark)
cd benchmarks
pytest bench_numpy_interop.py --benchmark-only
```

### Explore the Code

```
CNDA/
├── include/cnda/           # C++ headers (header-only)
│   ├── contiguous_nd.hpp   # Main container class
│   └── aos_types.hpp       # Structured types (Vec2f, Cell2D, etc.)
├── python/                 # Python bindings (pybind11)
│   ├── module.cpp          # Main binding code
│   ├── aos_types.cpp       # AoS bindings
│   └── utils.hpp           # Zero-copy validation helpers
├── examples/               # Runnable examples (NEW!)
│   ├── 01_python_roundtrip.py
│   ├── 02_cpp_to_numpy.py
│   ├── 03_aos_struct.py
│   ├── 04_zero_copy_success_vs_failure.py
│   ├── 05_error_handling.py
│   └── README.md           # Examples guide
├── benchmarks/             # Performance benchmarks
│   ├── bench_core.cpp      # C++ core operations
│   ├── bench_comparison.cpp # vs std::vector
│   ├── bench_aos.cpp       # Array-of-Structs
│   └── bench_numpy_interop.py # NumPy interop
├── docs/                   # Documentation
│   ├── QUICKSTART.md       # This file
│   ├── INSTALLATION.md     # Installation guide
│   ├── PYTHON_USER_GUIDE.md # Python API
│   ├── CPP_USER_GUIDE.md   # C++ API
│   └── BENCHMARKING.md     # Performance guide
└── tests/                  # Test suite
    ├── cpp/                # C++ tests
    └── python/             # Python tests
```

---

## Performance at a Glance

**Zero-copy overhead** (measured):
- Small arrays (1 KB): **1.7-3.4 µs**
- `to_numpy` (100 MB): **1.8 µs** (size-independent!)
- `from_numpy` (100 MB): **14 ms** (validation scales with size)
- Deep copy (100 MB): **20-23 ms**

**Recommendation**: Always use `copy=False` when safe - `to_numpy` overhead is negligible regardless of size.

See [BENCHMARKING.md](BENCHMARKING.md) for detailed performance analysis.

---

## Quick Reference Card

```python
# Python API
import cnda
import numpy as np

# Create: ContiguousND_{f32,f64,i32,i64}([shape])
arr = cnda.ContiguousND_f32([3, 4])

# Access: arr[i, j] = val / val = arr[i, j]
# Metadata: arr.shape, arr.strides, arr.ndim, arr.size
# NumPy: cnda.from_numpy(arr, copy=False/True)
#        arr.to_numpy(copy=False/True)
```

```cpp
// C++ API
#include "cnda/contiguous_nd.hpp"

// Create: ContiguousND<T>({shape})
cnda::ContiguousND<float> arr({3, 4});

// Access: arr(i, j) = val / val = arr(i, j)
// Metadata: arr.ndim(), arr.size(), arr.shape(), arr.strides()
// Raw: arr.data() → T* pointer
```

---

**Version**: 0.1.0 | **Last Updated**: December 2024 | **License**: See [LICENSE](../LICENSE)
