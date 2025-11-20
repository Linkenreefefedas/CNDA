# Python Bindings for CNDA

## Overview

CNDA provides Python bindings via pybind11, exposing the `ContiguousND<T>` class template for multiple data types. The bindings support construction, indexing, and metadata access with zero-copy semantics.

## Installation

### Prerequisites

- Python 3.9 or later
- CMake 3.18 or later
- C++11 compatible compiler
- pybind11 2.6.0+

### Method 1: Install using pip (Recommended)

```bash
# Install in your environment
pip install .

# Or install in development mode
pip install -e .
```

This will automatically:
- Detect Python 3.9+ installation
- Find or install pybind11
- Build the C++ extension module
- Install the `cnda` package

### Method 2: Build with CMake

```bash
# Install pybind11 first
pip install "pybind11[global]"

# Configure and build
mkdir build
cd build
cmake ..
cmake --build .

# Add to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/python  # Linux/Mac
# or
$env:PYTHONPATH="$env:PYTHONPATH;$(pwd)\python"  # Windows PowerShell
```

**Note:** Python bindings are always built. CMake will fail if Python 3.9+ or pybind11 are not found.

## Available Types

The following dtype variants are available:

| Python Type | C++ Type | Description |
|-------------|----------|-------------|
| `ContiguousND_f32` | `ContiguousND<float>` | 32-bit floating point |
| `ContiguousND_f64` | `ContiguousND<double>` | 64-bit floating point |
| `ContiguousND_i32` | `ContiguousND<std::int32_t>` | 32-bit signed integer |
| `ContiguousND_i64` | `ContiguousND<std::int64_t>` | 64-bit signed integer |

Type aliases are also available:
- `ContiguousND_float` → `ContiguousND_f32`
- `ContiguousND_double` → `ContiguousND_f64`
- `ContiguousND_int32` → `ContiguousND_i32`
- `ContiguousND_int64` → `ContiguousND_i64`

## Usage Examples

### Basic Construction and Access

```python
import cnda

# Create a 3x4 float array
arr = cnda.ContiguousND_f32([3, 4])

# Check shape and strides
print(arr.shape)    # (3, 4)
print(arr.strides)  # (4, 1) - row-major layout
print(arr.ndim)     # 2
print(arr.size)     # 12

# Set values using tuple indexing
arr[0, 0] = 1.0
arr[1, 2] = 42.5
arr[2, 3] = 99.9

# Get values
val = arr[1, 2]
print(val)  # 42.5
```

### Multi-dimensional Arrays

```python
import cnda

# 1D array
arr1d = cnda.ContiguousND_f64([10])
arr1d[5] = 3.14
print(arr1d[5])

# 3D array
arr3d = cnda.ContiguousND_i32([2, 3, 4])
arr3d[1, 2, 3] = 123
print(arr3d[1, 2, 3])

# 4D array
arr4d = cnda.ContiguousND_i64([2, 3, 4, 5])
print(arr4d.shape)    # (2, 3, 4, 5)
print(arr4d.strides)  # (60, 20, 5, 1)
```

### Working with Different Data Types

```python
import cnda

# Float arrays
f32 = cnda.ContiguousND_f32([2, 3])
f64 = cnda.ContiguousND_f64([2, 3])

# Integer arrays
i32 = cnda.ContiguousND_i32([2, 3])
i64 = cnda.ContiguousND_i64([2, 3])

# Using aliases
arr_float = cnda.ContiguousND_float([2, 3])
arr_double = cnda.ContiguousND_double([2, 3])
```

### Checking Version

```python
import cnda

print(cnda.__version__)
```

## API Reference

### Constructor

```python
arr = ContiguousND_<dtype>(shape)
```

**Parameters:**
- `shape`: List or tuple of integers specifying array dimensions

**Example:**
```python
arr = cnda.ContiguousND_f32([3, 4, 5])
```

### Properties

#### `shape`
Returns a tuple of integers representing the array dimensions.

```python
shape = arr.shape  # Returns: tuple
```

#### `strides`
Returns a tuple of integers representing the stride (in elements, not bytes) for each dimension. Row-major layout.

```python
strides = arr.strides  # Returns: tuple
```

#### `ndim`
Returns the number of dimensions.

```python
n = arr.ndim  # Returns: int
```

#### `size`
Returns the total number of elements (product of shape).

```python
total = arr.size  # Returns: int
```

### Indexing

#### Get element
```python
value = arr[i, j, k, ...]
```

#### Set element
```python
arr[i, j, k, ...] = value
```

**Notes:**
- Number of indices must match `ndim`
- Out-of-bounds access raises `RuntimeError`
- Indices are 0-based

### Alternative indexing with `__call__`

```python
# Alternative syntax (also supported)
value = arr(i, j, k, ...)
```

## Testing

### Run Python tests

```bash
# Using pytest
pip install pytest
pytest tests/test_python_bindings.py -v

# Or run directly
python tests/test_python_bindings.py
```

## Error Handling

The bindings use specific Python exception types for different error conditions:

### Error Semantics

CNDA follows a strict error semantic policy that maps C++ exceptions to appropriate Python exception types:

| Python Exception | Error Condition | Example |
|-----------------|-----------------|---------|
| `TypeError` | Unsupported dtype mismatch | Passing uint8 array to `from_numpy()` |
| `ValueError` | Shape/layout mismatch | Non-C-contiguous array with `copy=False` |
| `RuntimeError` | Invalid lifetime/ownership | Capsule or memory management errors |
| `IndexError` | Out-of-bounds or rank mismatch | Accessing `arr[3, 0]` on shape `[3, 4]` |

### Exception Examples

```python
import cnda
import numpy as np

# TypeError: Unsupported dtype
try:
    x = np.array([[1, 2]], dtype=np.uint8)
    arr = cnda.from_numpy(x, copy=False)
except TypeError as e:
    print(f"TypeError: {e}")
    # Output: TypeError: Unsupported dtype: uint8

# ValueError: Layout mismatch
try:
    x = np.array([[1.0, 2.0]], dtype=np.float32, order='F')
    arr = cnda.from_numpy_f32(x, copy=False)
except ValueError as e:
    print(f"ValueError: {e}")
    # Output: ValueError: from_numpy with copy=False requires C-contiguous array

# IndexError: Out of bounds
try:
    arr = cnda.ContiguousND_f32([3, 4])
    val = arr[3, 0]
except IndexError as e:
    print(f"IndexError: {e}")
    # Output: IndexError: Index out of bounds

# IndexError: Rank mismatch
try:
    arr = cnda.ContiguousND_f32([3, 4])
    val = arr[0]  # Wrong number of indices
except IndexError as e:
    print(f"IndexError: {e}")
    # Output: IndexError: Number of indices does not match ndim
```

### C++ Exception Mapping

The Python bindings automatically translate C++ exceptions to Python exceptions:

- `std::out_of_range` → `IndexError`
- `std::invalid_argument` → `ValueError`
- `std::runtime_error` → `RuntimeError`
- `pybind11::type_error` → `TypeError`

## Memory Layout

CNDA uses **row-major** (C-style) memory layout:

```python
import cnda

arr = cnda.ContiguousND_f32([3, 4])
# Shape: (3, 4)
# Strides: (4, 1)
# Memory layout: [0,0] [0,1] [0,2] [0,3] [1,0] [1,1] ...
```

This is compatible with NumPy's default layout and allows for potential zero-copy interop.