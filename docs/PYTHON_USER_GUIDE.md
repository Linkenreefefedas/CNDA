# CNDA Python User Guide

A comprehensive guide to using CNDA's Python bindings with NumPy interoperability.

## Table of Contents

- [Overview](#overview)
- [Zero-Copy Memory Sharing](#zero-copy-memory-sharing)
- [API Reference](#api-reference)
- [Structured Types (AoS)](#structured-types-aos)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Performance Considerations](#performance-considerations)

---

## Overview

CNDA provides Python bindings via pybind11 that enable seamless interoperability with NumPy arrays. The key feature is **explicit control over zero-copy vs. copy semantics**.

### Core Philosophy

- **Explicit is better than implicit** - `copy=False` vs `copy=True` makes intent clear
- **Zero-copy when safe** - Share memory when dtype, layout, and lifetime are compatible
- **Clear error messages** - Know immediately why zero-copy failed

### Quick Example

```python
import numpy as np
import cnda

# NumPy → CNDA (zero-copy)
x = np.arange(12, dtype=np.float32).reshape(3, 4)
arr = cnda.from_numpy(x, copy=False)

# CNDA → NumPy (zero-copy)
y = arr.to_numpy(copy=False)

# Verify same memory
assert y.ctypes.data == x.ctypes.data  # True!
```

---

## Zero-Copy Memory Sharing

### What is Zero-Copy?

**Zero-copy** means that CNDA and NumPy share the **same underlying memory buffer** without duplication. Changes made through one interface are immediately visible through the other.

```python
import numpy as np
import cnda

x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
arr = cnda.from_numpy(x, copy=False)  # Zero-copy

# Modify via CNDA
arr[0, 0] = 999.0

# Change is visible in NumPy (same memory!)
print(x[0, 0])  # Output: 999.0
```

### When is Zero-Copy Possible?

Zero-copy requires **three conditions** to be met:

#### 1. Dtype Compatibility

The NumPy dtype must match the CNDA type:

| NumPy dtype | CNDA Type | Zero-Copy? |
|-------------|-----------|------------|
| `np.float32` | `ContiguousND_f32` | Yes |
| `np.float64` | `ContiguousND_f64` | Yes |
| `np.int32` | `ContiguousND_i32` | Yes |
| `np.int64` | `ContiguousND_i64` | Yes |
| `np.uint8` | Any | No - unsupported |
| `np.float16` | Any | No - unsupported |
| `np.complex128` | Any | No - unsupported |

**Example - Correct dtype:**
```python
x = np.array([[1.0, 2.0]], dtype=np.float32)
arr = cnda.from_numpy(x, copy=False)  # Works
```

**Example - Wrong dtype:**
```python
x = np.array([[1, 2]], dtype=np.uint8)
try:
    arr = cnda.from_numpy(x, copy=False)
except TypeError as e:
    print(e)  # "Unsupported dtype: uint8"
```

#### 2. C-Contiguous Layout (Row-Major)

The array must be stored in **row-major (C-style)** order with standard strides.

**What are strides?**  
Strides define how many **bytes** to skip to move along each dimension. For a shape `(3, 4)` with `float32` (4 bytes each):
- Standard C-contiguous strides: `(16, 4)` - skip 16 bytes for next row, 4 bytes for next column

**Example - C-contiguous (works):**
```python
x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='C')
print(x.flags['C_CONTIGUOUS'])  # True
arr = cnda.from_numpy(x, copy=False)  # Works
```

**Example - Fortran-order (fails):**
```python
x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='F')
print(x.flags['C_CONTIGUOUS'])  # False
try:
    arr = cnda.from_numpy(x, copy=False)
except ValueError as e:
    print(e)  # "from_numpy with copy=False requires C-contiguous array"
```

**Example - Non-contiguous slice (fails):**
```python
x = np.arange(20, dtype=np.float32).reshape(5, 4)
y = x[::2, :]  # Every other row - creates non-contiguous view
print(y.flags['C_CONTIGUOUS'])  # False
try:
    arr = cnda.from_numpy(y, copy=False)
except ValueError as e:
    print(e)  # "from_numpy with copy=False requires C-contiguous array"
```

**Solution - Use copy=True or make contiguous:**
```python
# Option 1: Explicit copy
arr = cnda.from_numpy(y, copy=True)  # Works

# Option 2: Make contiguous first
y_contig = np.ascontiguousarray(y)
arr = cnda.from_numpy(y_contig, copy=False)  # Works (but y_contig is already a copy)
```

#### 3. Lifetime Management

The memory must remain valid as long as the CNDA object exists. This is **automatically handled** by pybind11's capsule mechanism:

```python
def create_array():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    arr = cnda.from_numpy(x, copy=False)
    return arr  # x's reference count is kept alive via capsule

arr = create_array()  # Safe - capsule keeps NumPy array alive
print(arr[0, 0])     # Memory still valid
```

### Why Layout/Stride Restrictions?

**Memory Layout Matters for Performance:**

Row-major layout (C-style):
```
Shape: (3, 4)
Memory: [row0] [row1] [row2]
Layout: [0,0][0,1][0,2][0,3] [1,0][1,1][1,2][1,3] [2,0][2,1][2,2][2,3]
```

This layout is:
- **Cache-friendly** - accessing consecutive elements hits CPU cache
- **Predictable** - strides can be computed at construction time
- **Compatible** - matches NumPy's default and C/C++ row-major convention

**Non-standard strides break assumptions:**
```python
x = np.arange(20).reshape(5, 4)
y = x[::2, :]  # Skip every other row

# y's strides: (32, 4) instead of standard (16, 4)
# This means rows are NOT contiguous in memory!
```

CNDA's core assumes **standard contiguous layout** for:
1. O(1) offset computation
2. Cache-friendly iteration
3. Safe pointer arithmetic
4. Binary compatibility with C++ structs

**When strides are non-standard, you MUST copy:**
```python
arr = cnda.from_numpy(non_contiguous, copy=True)  # Forces re-layout
```

### When to Use copy=True?

Use `copy=True` when:

1. **Array is not C-contiguous** (Fortran order or non-contiguous slice)
2. **Need independent lifetime** (data outlives original array)
3. **Modifying one shouldn't affect the other** (independent modifications)
4. **One-time copy is cheaper** (for many operations on non-contiguous data)

```python
# Example: Independent copy
x = np.array([[1.0, 2.0]], dtype=np.float32)
arr = cnda.from_numpy(x, copy=True)
arr[0, 0] = 999.0
print(x[0, 0])  # Still 1.0 - not affected
```

---

## API Reference

### Creating Arrays

#### Constructor

```python
cnda.ContiguousND_<dtype>(shape)
```

Create a new CNDA array with the given shape.

**Parameters:**
- `shape` : list or tuple of integers
  - Dimensions of the array
  - Example: `[3, 4]` for a 3×4 matrix

**Returns:**
- `ContiguousND_<dtype>` : New array with uninitialized values

**Available Types:**

| Constructor | C++ Type | NumPy Equivalent |
|-------------|----------|------------------|
| `ContiguousND_f32` | `float` | `np.float32` |
| `ContiguousND_f64` | `double` | `np.float64` |
| `ContiguousND_i32` | `int32_t` | `np.int32` |
| `ContiguousND_i64` | `int64_t` | `np.int64` |

**Type Aliases:**
- `ContiguousND_float` → `ContiguousND_f32`
- `ContiguousND_double` → `ContiguousND_f64`
- `ContiguousND_int32` → `ContiguousND_i32`
- `ContiguousND_int64` → `ContiguousND_i64`

**Examples:**

```python
import cnda

# Create different array types
arr_f32 = cnda.ContiguousND_f32([3, 4])       # 3×4 float32
arr_f64 = cnda.ContiguousND_f64([10, 20, 30]) # 10×20×30 float64
arr_i32 = cnda.ContiguousND_i32([100])        # 1D int32 array
arr_i64 = cnda.ContiguousND_i64([2, 3, 4, 5]) # 4D int64 array

# Using aliases
arr = cnda.ContiguousND_float([3, 4])   # Same as ContiguousND_f32
arr = cnda.ContiguousND_double([3, 4])  # Same as ContiguousND_f64
```

### Properties

#### `shape`

```python
arr.shape -> tuple
```

Returns the dimensions of the array.

**Example:**
```python
arr = cnda.ContiguousND_f32([3, 4, 5])
print(arr.shape)  # Output: (3, 4, 5)
```

#### `strides`

```python
arr.strides -> tuple
```

Returns the stride (in **elements**, not bytes) for each dimension.

**Example:**
```python
arr = cnda.ContiguousND_f32([3, 4, 5])
print(arr.strides)  # Output: (20, 5, 1)
# To access arr[i,j,k]: offset = i*20 + j*5 + k*1
```

#### `ndim`

```python
arr.ndim -> int
```

Returns the number of dimensions.

**Example:**
```python
arr = cnda.ContiguousND_f32([3, 4, 5])
print(arr.ndim)  # Output: 3
```

#### `size`

```python
arr.size -> int
```

Returns the total number of elements (product of shape).

**Example:**
```python
arr = cnda.ContiguousND_f32([3, 4, 5])
print(arr.size)  # Output: 60 (3 × 4 × 5)
```

### Indexing

#### Get Element

```python
value = arr[i, j, k, ...]
```

**Example:**
```python
arr = cnda.ContiguousND_f32([3, 4])
arr[1, 2] = 42.0
val = arr[1, 2]
print(val)  # Output: 42.0
```

#### Set Element

```python
arr[i, j, k, ...] = value
```

**Example:**
```python
arr = cnda.ContiguousND_f32([3, 4])
arr[0, 0] = 1.0
arr[1, 2] = 42.5
arr[2, 3] = 99.9
```

**Bounds Checking:**

Python bindings **always perform bounds checking**:

```python
arr = cnda.ContiguousND_f32([3, 4])
try:
    val = arr[3, 0]  # Index 3 is out of bounds for axis 0 (size 3)
except IndexError as e:
    print(e)  # "Index out of bounds"
```

**Rank Mismatch:**

```python
arr = cnda.ContiguousND_f32([3, 4])
try:
    val = arr[0]  # Wrong: 2D array needs 2 indices
except IndexError as e:
    print(e)  # "Number of indices does not match ndim"
```

### NumPy Interoperability

#### `from_numpy(arr, copy=False)`

```python
cnda.from_numpy(arr, copy=False) -> ContiguousND_*
```

Create a CNDA array from a NumPy array.

**Parameters:**
- `arr` : `numpy.ndarray`
  - Source NumPy array
- `copy` : `bool`, optional (default=`False`)
  - If `False`: attempts zero-copy (raises error if incompatible)
  - If `True`: always creates an independent copy

**Returns:**
- `ContiguousND_*` : CNDA array (type suffix matches dtype)

**Raises:**
- `TypeError` : Unsupported dtype
- `ValueError` : Layout mismatch (when `copy=False`)

**Type-Specific Variants:**

For explicit type control, use:
- `cnda.from_numpy_f32(arr, copy=False)`
- `cnda.from_numpy_f64(arr, copy=False)`
- `cnda.from_numpy_i32(arr, copy=False)`
- `cnda.from_numpy_i64(arr, copy=False)`

**Examples:**

**Zero-copy (strict):**
```python
import numpy as np
import cnda

x = np.arange(12, dtype=np.float32).reshape(3, 4)
arr = cnda.from_numpy(x, copy=False)  # Zero-copy

# Verify shared memory
y = arr.to_numpy(copy=False)
assert y.ctypes.data == x.ctypes.data  # Same pointer
```

**Explicit copy:**
```python
x = np.arange(12, dtype=np.float32).reshape(3, 4)
arr = cnda.from_numpy(x, copy=True)  # Independent copy

arr[0, 0] = 999.0
print(x[0, 0])  # Still 0.0 - not affected
```

**Auto-detect dtype:**
```python
x32 = np.array([[1.0, 2.0]], dtype=np.float32)
x64 = np.array([[1.0, 2.0]], dtype=np.float64)

arr32 = cnda.from_numpy(x32)  # Returns ContiguousND_f32
arr64 = cnda.from_numpy(x64)  # Returns ContiguousND_f64
```

**Handling errors:**

For dtype or layout issues, use `copy=True` to force conversion:

```python
# Unsupported dtype - convert first
x = np.array([[1, 2]], dtype=np.uint8)
x = x.astype(np.int32)
arr = cnda.from_numpy(x)

# Non-contiguous - use copy=True
y = x[::2, :]
arr = cnda.from_numpy(y, copy=True)
```

See [Error Handling](#error-handling) for detailed error types and solutions.

#### `to_numpy(copy=False)`

```python
arr.to_numpy(copy=False) -> numpy.ndarray
```

Export CNDA array to NumPy.

**Parameters:**
- `copy` : `bool`, optional (default=`False`)
  - If `False`: returns zero-copy view (shares memory)
  - If `True`: returns independent copy

**Returns:**
- `numpy.ndarray` : NumPy array

**Lifetime Management:**

With `copy=False`, a **capsule deleter** automatically manages lifetime - the
NumPy array keeps the CNDA object alive. With `copy=True`, you get an
independent copy with separate lifetime.

```python
# Zero-copy: capsule manages lifetime
def create_array():
    arr = cnda.ContiguousND_f32([3, 4])
    arr[0, 0] = 42.0
    return arr.to_numpy(copy=False)  # Safe - capsule keeps arr alive

y = create_array()
print(y[0, 0])  # Valid - capsule manages lifetime

# Independent copy: separate lifetime
arr = cnda.ContiguousND_f32([3, 4])
y = arr.to_numpy(copy=True)
del arr  # y unaffected - has its own memory
```

**Examples:**

**Zero-copy view:**
```python
import cnda
import numpy as np

arr = cnda.ContiguousND_f32([3, 4])
arr[1, 2] = 42.0

y = arr.to_numpy(copy=False)  # Zero-copy
print(y[1, 2])  # 42.0

# Modify via NumPy
y[1, 2] = 999.0

# Change visible in CNDA (same memory)
print(arr[1, 2])  # 999.0
```

**Independent copy:**
```python
arr = cnda.ContiguousND_f32([3, 4])
arr[1, 2] = 42.0

y = arr.to_numpy(copy=True)  # Independent copy
y[1, 2] = 999.0

print(arr[1, 2])  # Still 42.0 - not affected
```

---

## Structured Types (AoS)

CNDA supports **Array-of-Structs (AoS)** layouts via NumPy's structured dtypes.

### What is AoS?

**Array-of-Structs** stores multiple related values per grid point as a single struct:

```
AoS Layout (Array-of-Structs):
Cell[0]: {u=1.0, v=2.0, flag=1}
Cell[1]: {u=1.5, v=2.5, flag=1}
Cell[2]: {u=2.0, v=3.0, flag=0}
...

Memory: [u0,v0,flag0] [u1,v1,flag1] [u2,v2,flag2] ...
```

**Benefits:**
- Cache-friendly for accessing all fields of one cell
- Natural for physics simulations (each cell = one entity)
- Matches C++ struct layout

**Alternative - SoA (Struct-of-Arrays):**
```
Memory: [u0,u1,u2,...] [v0,v1,v2,...] [flag0,flag1,flag2,...]
```
- Better for SIMD/vectorization when accessing one field across many cells
- Not supported in CNDA v0.1 (future work)

### Creating Structured Arrays

#### Define NumPy Structured Dtype

```python
import numpy as np

# Simple 2D cell with velocity + flag
cell_dtype = np.dtype([
    ('u', '<f4'),      # velocity x (float32, little-endian)
    ('v', '<f4'),      # velocity y (float32)
    ('flag', '<i4')    # status flag (int32)
], align=True)  # IMPORTANT: align=True for C++ compatibility
```

**Dtype Format:**
- `'<f4'` = little-endian float32
- `'<f8'` = little-endian float64
- `'<i4'` = little-endian int32
- `'<i8'` = little-endian int64
- `align=True` = add padding for natural alignment (matches C++)

#### Create Structured Array

```python
# Create 100×100 grid of Cell2D structs
grid = np.zeros((100, 100), dtype=cell_dtype, order='C')

# Set field values
grid['u'][50, 50] = 1.5
grid['v'][50, 50] = 2.0
grid['flag'][50, 50] = 1
```

### Binary Compatibility with C++ Structs

For zero-copy interop, the NumPy dtype **must match** the C++ struct layout:

**C++ Side (aos_types.hpp):**
```cpp
struct Cell2D {
    float       u;     // 4 bytes
    float       v;     // 4 bytes
    std::int32_t flag; // 4 bytes
};
// Total: 12 bytes, no padding needed
```

**Python Side:**
```python
cell_dtype = np.dtype([
    ('u', '<f4'),      # 4 bytes
    ('v', '<f4'),      # 4 bytes
    ('flag', '<i4')    # 4 bytes
], align=True)
# Total: 12 bytes, matches C++ layout
```

### Alignment and Padding

**Why `align=True` is Critical:**

Without `align=True`, NumPy packs fields tightly (9 bytes), but C++ compilers
add padding for alignment (12 bytes). Always use `align=True` to match C++:

```python
# ALWAYS use align=True for C++ interop
dtype = np.dtype([
    ('x', '<f4'),    # 4 bytes
    ('flag', '<i1'), # 1 byte + 3 bytes padding (with align=True)
    ('y', '<f4')     # 4 bytes
], align=True)  # Total: 12 bytes, matches C++ struct
```

### Ensuring Struct Compatibility

**Checklist for Python ↔ C++ struct interop:**

1. Use `align=True` in NumPy dtype
2. Match field order exactly (Python list order = C++ struct order)
3. Match field types: `'<f4'`↔`float`, `'<f8'`↔`double`, `'<i4'`↔`int32_t`, `'<i8'`↔`int64_t`
4. C++ struct must be POD (Plain Old Data)
5. Verify sizes match: `dtype.itemsize == sizeof(C++ struct)`

```python
# Verify compatibility
dtype = np.dtype([('u', '<f4'), ('v', '<f4'), ('flag', '<i4')], align=True)
print(f"Struct size: {dtype.itemsize} bytes")  # Should match C++
```

### AoS Example: Fluid Simulation Grid

```python
import numpy as np
import cnda

# Define fluid cell struct
fluid_dtype = np.dtype([
    ('u', '<f4'),        # velocity x
    ('v', '<f4'),        # velocity y
    ('pressure', '<f4'), # pressure
    ('flag', '<i4')      # boundary/fluid flag
], align=True)

# Create 512×512 grid
nx, ny = 512, 512
grid = np.zeros((nx, ny), dtype=fluid_dtype, order='C')

# Initialize boundary conditions
grid['flag'][0, :] = -1    # Bottom wall
grid['flag'][-1, :] = -1   # Top wall
grid['flag'][:, 0] = -1    # Left wall
grid['flag'][:, -1] = -1   # Right wall

# Initialize flow
grid['u'][1:-1, 1:-1] = 1.0   # Uniform flow
grid['pressure'][:, :] = 1.0  # Atmospheric pressure

# Zero-copy to CNDA (if layout matches C++ struct)
try:
    cnda_grid = cnda.from_numpy(grid, copy=False)
    print("Zero-copy successful - Python and C++ share memory")
except (TypeError, ValueError) as e:
    print(f"Zero-copy failed: {e}")
    print("Using explicit copy instead")
    cnda_grid = cnda.from_numpy(grid, copy=True)

# Pass to C++ for simulation step (via custom bindings)
# ... C++ updates grid ...

# Export back to Python
result = cnda_grid.to_numpy(copy=False)

# Access results
print(f"Max velocity: {np.max(result['u'])}")
print(f"Min pressure: {np.min(result['pressure'])}")
```

### AoS Field Access API

For specific AoS types (`Vec2f`, `Cell2D`, `Particle`), CNDA provides 
dedicated field getter/setter methods for convenience.

#### Vec2f API

**Struct Layout:**
```cpp
struct Vec2f {
    float x;
    float y;
};
```

**Python Methods:**

```python
arr = cnda.ContiguousND_Vec2f([100, 100])

# Get individual fields
x_val = arr.get_x(i, j)
y_val = arr.get_y(i, j)

# Set individual fields
arr.set_x(i, j, 1.5)
arr.set_y(i, j, 2.0)
```

**Example:**
```python
import cnda

# Create 10x10 grid of 2D vectors
vectors = cnda.ContiguousND_Vec2f([10, 10])

# Initialize with unit vectors
for i in range(10):
    for j in range(10):
        vectors.set_x(i, j, 1.0)
        vectors.set_y(i, j, 0.0)

# Read back
x = vectors.get_x(5, 5)  # 1.0
y = vectors.get_y(5, 5)  # 0.0
```

#### Cell2D API

**Struct Layout:**
```cpp
struct Cell2D {
    float u;
    float v;
    int32_t flag;
};
```

**Python Methods:**

```python
arr = cnda.ContiguousND_Cell2D([100, 100])

# Get fields
u_val = arr.get_u(i, j)
v_val = arr.get_v(i, j)
flag_val = arr.get_flag(i, j)

# Set fields
arr.set_u(i, j, 1.5)
arr.set_v(i, j, 2.0)
arr.set_flag(i, j, 1)
```

**Example:**
```python
import cnda

# Create fluid simulation grid
grid = cnda.ContiguousND_Cell2D([256, 256])

# Set boundary cells
for i in range(256):
    grid.set_flag(0, i, -1)      # Bottom wall
    grid.set_flag(255, i, -1)    # Top wall
    grid.set_flag(i, 0, -1)      # Left wall
    grid.set_flag(i, 255, -1)    # Right wall

# Initialize interior with uniform flow
for i in range(1, 255):
    for j in range(1, 255):
        grid.set_u(i, j, 1.0)
        grid.set_v(i, j, 0.0)
        grid.set_flag(i, j, 0)  # Fluid cell

# Read velocity at center
u_center = grid.get_u(128, 128)  # 1.0
v_center = grid.get_v(128, 128)  # 0.0
```

#### Particle API

**Struct Layout:**
```cpp
struct Particle {
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
};
```

**Python Methods:**

```python
arr = cnda.ContiguousND_Particle([1000])

# Position getters/setters
x = arr.get_x(i)
y = arr.get_y(i)
z = arr.get_z(i)

arr.set_x(i, 1.0)
arr.set_y(i, 2.0)
arr.set_z(i, 3.0)

# Velocity getters/setters
vx = arr.get_vx(i)
vy = arr.get_vy(i)
vz = arr.get_vz(i)

arr.set_vx(i, 0.1)
arr.set_vy(i, 0.2)
arr.set_vz(i, 0.3)
```

**Example:**
```python
import cnda
import random

# Create particle system
n_particles = 10000
particles = cnda.ContiguousND_Particle([n_particles])

# Initialize with random positions and velocities
for i in range(n_particles):
    particles.set_x(i, random.uniform(-10, 10))
    particles.set_y(i, random.uniform(-10, 10))
    particles.set_z(i, random.uniform(-10, 10))
    
    particles.set_vx(i, random.uniform(-1, 1))
    particles.set_vy(i, random.uniform(-1, 1))
    particles.set_vz(i, random.uniform(-1, 1))

# Simple time integration (Euler)
dt = 0.01
for i in range(n_particles):
    x = particles.get_x(i)
    y = particles.get_y(i)
    z = particles.get_z(i)
    
    vx = particles.get_vx(i)
    vy = particles.get_vy(i)
    vz = particles.get_vz(i)
    
    # Update positions
    particles.set_x(i, x + vx * dt)
    particles.set_y(i, y + vy * dt)
    particles.set_z(i, z + vz * dt)
```

**Performance Note:**

For bulk operations, prefer NumPy field access over individual getters/setters:

```python
# SLOW: Field access in Python loop
for i in range(n_particles):
    x = particles.get_x(i)
    particles.set_x(i, x + 1.0)

# FAST: NumPy vectorized field access
data = particles.to_numpy(copy=False)
data['x'] += 1.0
```

### Common AoS Pitfalls

#### Pitfall 1: Field Order Mismatch

```python
# Python
dtype = np.dtype([('v', '<f4'), ('u', '<f4'), ('flag', '<i4')], align=True)

# C++
struct Cell2D {
    float u;  // ← Order doesn't match!
    float v;
    int32_t flag;
};
```

**Result:** Zero-copy may succeed, but data is interpreted incorrectly!

**Solution:** Match field order exactly.

#### Pitfall 3: Non-Contiguous Array

```python
grid = np.zeros((100, 100), dtype=cell_dtype, order='C')
subset = grid[::2, ::2]  # Every other cell - non-contiguous!

try:
    arr = cnda.from_numpy(subset, copy=False)
except ValueError:
    # Fails - subset is not contiguous
    arr = cnda.from_numpy(subset, copy=True)  # Must copy
```

---

## Error Handling

CNDA provides clear, specific exception types for different error conditions.

### Exception Types

| Python Exception | Condition | C++ Exception |
|-----------------|-----------|---------------|
| `TypeError` | Unsupported dtype | `pybind11::type_error` |
| `ValueError` | Layout/shape mismatch | `std::invalid_argument` |
| `RuntimeError` | Memory/lifetime issues | `std::runtime_error` |
| `IndexError` | Out-of-bounds or rank mismatch | `std::out_of_range` |

### Common Errors and Solutions

#### TypeError: Unsupported Dtype

**Error:**
```python
x = np.array([[1, 2]], dtype=np.uint8)
arr = cnda.from_numpy(x)
# TypeError: Unsupported dtype: uint8
```

**Cause:** CNDA only supports `float32`, `float64`, `int32`, `int64`.

**Solution:**
```python
# Convert to supported dtype
x = x.astype(np.int32)
arr = cnda.from_numpy(x)  # Works
```

#### ValueError: Non-Contiguous Array

**Error:**
```python
x = np.arange(20, dtype=np.float32).reshape(5, 4)
y = x[::2, :]  # Non-contiguous
arr = cnda.from_numpy(y, copy=False)
# ValueError: from_numpy with copy=False requires C-contiguous array
```

**Cause:** Array has non-standard strides.

**Solution:**
```python
# Use copy=True to force re-layout
arr = cnda.from_numpy(y, copy=True)
```

#### ValueError: Fortran Order

**Error:**
```python
x = np.array([[1.0, 2.0]], dtype=np.float32, order='F')
arr = cnda.from_numpy(x, copy=False)
# ValueError: from_numpy with copy=False requires C-contiguous array
```

**Cause:** Array is Fortran-ordered (column-major).

**Solution:**
```python
# Use copy=True to convert to C-order
arr = cnda.from_numpy(x, copy=True)
```

#### IndexError: Out of Bounds

**Error:**
```python
arr = cnda.ContiguousND_f32([3, 4])
val = arr[3, 0]  # Index 3 is out of bounds for axis 0 (size 3)
# IndexError: Index out of bounds
```

**Cause:** Index exceeds array dimensions.

**Solution:**
```python
# Valid indices for shape (3, 4) are [0-2, 0-3]
val = arr[2, 3]  # Last valid element
```

#### IndexError: Rank Mismatch

**Error:**
```python
arr = cnda.ContiguousND_f32([3, 4])
val = arr[0]  # Wrong: 2D array needs 2 indices
# IndexError: Number of indices does not match ndim
```

**Solution:**
```python
# Provide correct number of indices
val = arr[0, 0]  # Correct for 2D array
```

### Error Handling Best Practices

```python
import numpy as np
import cnda

def safe_from_numpy(arr, copy=False):
    """Wrapper with error handling and auto-retry."""
    try:
        return cnda.from_numpy(arr, copy=copy)
    except TypeError as e:
        print(f"Unsupported dtype: {arr.dtype}")
        # Try converting to float32
        if arr.dtype != np.float32:
            print("Warning: Converting to float32...")
            arr = arr.astype(np.float32)
            return cnda.from_numpy(arr, copy=copy)
        else:
            raise
    except ValueError as e:
        if not copy and not arr.flags['C_CONTIGUOUS']:
            print("Warning: Array is not contiguous, forcing copy...")
            return cnda.from_numpy(arr, copy=True)
        else:
            raise

# Usage
x = np.array([[1, 2]], dtype=np.uint8)
arr = safe_from_numpy(x)  # Auto-converts and retries
```

---

## Best Practices

### 1. Prefer Zero-Copy When Possible

```python
# GOOD: Zero-copy when safe
x = np.arange(12, dtype=np.float32).reshape(3, 4)
arr = cnda.from_numpy(x, copy=False)

# BAD: Unnecessary copy
arr = cnda.from_numpy(x, copy=True)  # Wasteful if zero-copy is possible
```

### 2. Use Explicit Dtypes

```python
# GOOD: Explicit dtype
x = np.arange(12, dtype=np.float32).reshape(3, 4)

# BAD: Implicit dtype (may be float64 on some platforms)
x = np.arange(12.0).reshape(3, 4)
```

### 3. Check Contiguity Before Zero-Copy

```python
import numpy as np

x = get_array_from_somewhere()

# Check before attempting zero-copy
if x.flags['C_CONTIGUOUS'] and x.dtype in [np.float32, np.float64, np.int32, np.int64]:
    arr = cnda.from_numpy(x, copy=False)  # Safe
else:
    arr = cnda.from_numpy(x, copy=True)   # Fallback
```

### 4. Document Copy Semantics

Always document whether functions use zero-copy or copy, especially when
performance-critical.

### 5. Validate Structured Types

Verify NumPy dtype size matches C++ struct:

```python
dtype = np.dtype([('u', '<f4'), ('v', '<f4'), ('flag', '<i4')], align=True)
assert dtype.itemsize == 12, f"Size mismatch: {dtype.itemsize} != 12"
```

---

## Performance Considerations

### Zero-Copy Overhead

**Zero-copy overhead:** < 1 microsecond (capsule creation)

```python
import numpy as np
import cnda
import time

x = np.arange(1_000_000, dtype=np.float32)

# Measure zero-copy overhead
start = time.perf_counter()
arr = cnda.from_numpy(x, copy=False)
elapsed = time.perf_counter() - start
print(f"Zero-copy overhead: {elapsed*1e6:.2f} µs")  # ~0.5 µs
```

### Copy Overhead

**Copy overhead:** Proportional to data size (~1 GB/s)

```python
x = np.arange(10_000_000, dtype=np.float32)  # 40 MB

start = time.perf_counter()
arr = cnda.from_numpy(x, copy=True)
elapsed = time.perf_counter() - start
print(f"Copy time: {elapsed*1000:.2f} ms")  # ~40 ms for 40 MB
```

### When Copy is Faster

If you'll access the data many times and the source is non-contiguous, **one-time copy is faster** than repeated stride calculations:

```python
x = np.arange(100_000).reshape(1000, 100)
y = x[::2, :]  # Non-contiguous

# Option 1: Copy once
arr = cnda.from_numpy(y, copy=True)  # One-time cost
# ... many operations on arr ...

# Option 2: No copy (slower overall if many operations)
# Would need to handle non-contiguous strides on every access
```

### Avoid Python Loops

```python
import numpy as np
import cnda

arr = cnda.ContiguousND_f32([1000, 1000])

# SLOW: Python loop
for i in range(1000):
    for j in range(1000):
        arr[i, j] = i + j  # ~seconds

# FAST: NumPy vectorization
data = arr.to_numpy(copy=False)
data[:] = np.arange(1000)[:, None] + np.arange(1000)[None, :]  # ~milliseconds
```

---

## See Also

- **[QUICKSTART.md](QUICKSTART.md)** - Quick introduction
- **[CPP_API.md](CPP_API.md)** - C++ API reference
- **[INSTALLATION.md](INSTALLATION.md)** - Installation guide

---

**Version**: 0.1.0 | **Last Updated**: December 2024
