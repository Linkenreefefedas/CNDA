# CNDA C++ User Guide

A comprehensive guide to using CNDA's header-only C++11 core library for efficient multi-dimensional array management.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [ContiguousND Template Class](#contiguousnd-template-class)
- [Memory Layout and Guarantees](#memory-layout-and-guarantees)
- [Ownership and Lifetime](#ownership-and-lifetime)
- [Structured Types (AoS)](#structured-types-aos)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)

---

## Overview

CNDA's C++ core is a **header-only** library providing the `ContiguousND<T>` template class for cache-friendly, row-major N-dimensional arrays with:

- **Zero dependencies** - Only requires C++11 standard library
- **O(1) indexing** - Efficient `operator()` with compile-time optimization
- **Type safety** - Compile-time POD checks for structs
- **Move semantics** - No accidental copies
- **Bounds checking** - Optional `at()` or compile-time flag

### Quick Example

```cpp
#include "cnda/contiguous_nd.hpp"
#include <iostream>

int main() {
    // Create a 3×4 float array
    cnda::ContiguousND<float> arr({3, 4});
    
    // Set values
    arr(1, 2) = 42.5f;
    
    // Get values
    std::cout << "arr(1, 2) = " << arr(1, 2) << "\n";
    
    // Metadata
    std::cout << "Shape: (" << arr.shape()[0] << ", " 
              << arr.shape()[1] << ")\n";
    std::cout << "Size: " << arr.size() << "\n";
    
    return 0;
}
```

---

## Installation

### Header-Only Integration

The C++ core consists of just two headers with no external dependencies:

```
include/cnda/
├── contiguous_nd.hpp    # Main container template
└── aos_types.hpp        # Example structured types (optional)
```

#### Option 1: Copy Headers

```bash
# Copy headers to your project
cp -r include/cnda /path/to/your/project/include/
```

**Usage in your code:**
```cpp
#include "cnda/contiguous_nd.hpp"

int main() {
    cnda::ContiguousND<float> arr({10, 20});
    return 0;
}
```

**Compile:**
```bash
# Linux/macOS
g++ -std=c++11 -I/path/to/include main.cpp -o main

# Windows (MSVC)
cl /std:c++11 /EHsc /I\path\to\include main.cpp
```

#### Option 2: CMake Integration

```cmake
# Your CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(MyProject)

# Add CNDA as subdirectory
add_subdirectory(external/CNDA)

# Your executable
add_executable(my_program main.cpp)
target_link_libraries(my_program PRIVATE cnda_headers)
```

#### Option 3: CMake FetchContent

```cmake
include(FetchContent)

FetchContent_Declare(
  cnda
  GIT_REPOSITORY https://github.com/Linkenreefefedas/CNDA.git
  GIT_TAG        v0.1.0
)
FetchContent_MakeAvailable(cnda)

target_link_libraries(my_program PRIVATE cnda_headers)
```

### Compile-Time Options

#### Enable Bounds Checking

```bash
# Add -DCNDA_BOUNDS_CHECK flag
g++ -std=c++11 -DCNDA_BOUNDS_CHECK -I./include main.cpp -o main
```

**CMake:**
```cmake
target_compile_definitions(my_program PRIVATE CNDA_BOUNDS_CHECK)
```

**Effect:**
- `operator()` performs bounds checks (otherwise no checks for performance)
- `at()` always performs bounds checks regardless of this flag

---

## ContiguousND Template Class

### Class Template

```cpp
namespace cnda {

template <class T>
class ContiguousND {
  // Compile-time checks
  static_assert(std::is_standard_layout<T>::value,
                "ContiguousND requires T to be standard-layout type");
  static_assert(std::is_trivially_copyable<T>::value,
                "ContiguousND requires T to be trivially copyable");

public:
  // Constructors, accessors, indexing...
};

} // namespace cnda
```

### Supported Types

CNDA requires **POD (Plain Old Data)** types:

| Category | Types | Requirements |
|----------|-------|--------------|
| **Fundamental** | `float`, `double`, `int32_t`, `int64_t`, etc. | Always supported |
| **Structs** | Custom POD structs | Must be standard-layout + trivially-copyable |
| **Non-POD** | Classes with constructors, `std::string`, `std::vector` | Not supported |

**Why POD-only?**
- Safe for `memcpy` and contiguous storage
- Predictable binary layout for NumPy interop
- No hidden allocations or destructor side effects

### Constructors

#### Self-Owned Array

```cpp
explicit ContiguousND(std::vector<std::size_t> shape)
```

Creates a new array with the given shape. Memory is allocated and owned by the `ContiguousND` object.

**Parameters:**
- `shape` : `std::vector<std::size_t>` - Dimensions of the array

**Example:**
```cpp
#include "cnda/contiguous_nd.hpp"

// 1D array
cnda::ContiguousND<double> vec({100});

// 2D array
cnda::ContiguousND<float> matrix({10, 20});

// 3D array
cnda::ContiguousND<int32_t> cube({5, 10, 15});

// 4D array
cnda::ContiguousND<int64_t> hyper({2, 3, 4, 5});
```

#### External Memory View

```cpp
ContiguousND(std::vector<std::size_t> shape,
             T* external_data,
             std::shared_ptr<void> external_owner)
```

Creates a **view** over external memory. The `shared_ptr` keeps the data owner alive.

**Parameters:**
- `shape` : Dimensions
- `external_data` : Pointer to existing contiguous memory
- `external_owner` : Shared pointer to owner (keeps memory alive)

**Example:**
```cpp
// Existing buffer (e.g., from NumPy, another library, etc.)
float* buffer = get_external_buffer();

// Wrap as CNDA view
auto owner = std::make_shared<BufferOwner>(buffer);
cnda::ContiguousND<float> view({10, 20}, buffer, owner);

// view shares memory with buffer
view(0, 0) = 42.0f;
assert(buffer[0] == 42.0f);  // Same memory
```

### Move Semantics

`ContiguousND` is **move-only** to prevent accidental expensive copies:

```cpp
// Move construction/assignment (cheap - O(1))
cnda::ContiguousND<float> arr1({10, 20});
cnda::ContiguousND<float> arr2(std::move(arr1));

// Copy construction/assignment (deleted)
// cnda::ContiguousND<float> arr3(arr2);  // Compile error
```

See [Ownership and Lifetime](#ownership-and-lifetime) for detailed semantics.

### Metadata Accessors

#### `shape()`

```cpp
const std::vector<std::size_t>& shape() const noexcept
```

Returns the dimensions of the array.

**Example:**
```cpp
cnda::ContiguousND<float> arr({3, 4, 5});
const auto& s = arr.shape();
std::cout << "Shape: (" << s[0] << ", " << s[1] << ", " << s[2] << ")\n";
// Output: Shape: (3, 4, 5)
```

#### `strides()`

```cpp
const std::vector<std::size_t>& strides() const noexcept
```

Returns the stride (in **elements**, not bytes) for each dimension.

**Row-major stride formula:**
```
strides[i] = product of shape[i+1] through shape[n-1]
```

**Example:**
```cpp
cnda::ContiguousND<float> arr({3, 4, 5});
const auto& st = arr.strides();
std::cout << "Strides: (" << st[0] << ", " << st[1] << ", " << st[2] << ")\n";
// Output: Strides: (20, 5, 1)
```

See [Memory Layout](#memory-layout-and-guarantees) for stride computation details.

#### `ndim()`

```cpp
std::size_t ndim() const noexcept
```

Returns the number of dimensions.

**Example:**
```cpp
cnda::ContiguousND<float> arr({3, 4, 5});
std::cout << "Dimensions: " << arr.ndim() << "\n";  // Output: 3
```

#### `size()`

```cpp
std::size_t size() const noexcept
```

Returns the total number of elements (product of shape).

**Example:**
```cpp
cnda::ContiguousND<float> arr({3, 4, 5});
std::cout << "Total elements: " << arr.size() << "\n";  // Output: 60
```

#### `data()`

```cpp
T*       data()       noexcept
const T* data() const noexcept
```

Returns a pointer to the underlying contiguous buffer.

**Example:**
```cpp
cnda::ContiguousND<float> arr({10, 20});

// Mutable access
float* ptr = arr.data();
ptr[0] = 1.0f;

// Const access
const cnda::ContiguousND<float>& carr = arr;
const float* cptr = carr.data();
std::cout << cptr[0] << "\n";  // 1.0
```

#### `is_view()`

```cpp
bool is_view() const noexcept
```

Returns `true` if the array is a view over external memory, `false` if self-owned.

**Example:**
```cpp
// Self-owned
cnda::ContiguousND<float> arr({10, 20});
assert(!arr.is_view());

// View
auto owner = std::make_shared<float[]>(200);
cnda::ContiguousND<float> view({10, 20}, owner.get(), owner);
assert(view.is_view());
```

### Indexing

#### `operator()` - Fast Unchecked Access

```cpp
T& operator()(Index... indices)
const T& operator()(Index... indices) const
```

Provides **O(1) element access** with optimized code generation.

**Bounds checking:**
- **Without `-DCNDA_BOUNDS_CHECK`**: No checks (maximum performance)
- **With `-DCNDA_BOUNDS_CHECK`**: Runtime checks, throws `std::out_of_range`

**Optimized specializations:**
- 1D, 2D, 3D, 4D: Direct offset calculation (no loop)
- 5D+: General variadic version with loop

**Examples:**

**1D access:**
```cpp
cnda::ContiguousND<double> vec({100});
vec(50) = 3.14;
double val = vec(50);
```

**2D access:**
```cpp
cnda::ContiguousND<float> mat({10, 20});
mat(5, 10) = 42.5f;
float val = mat(5, 10);
```

**3D access:**
```cpp
cnda::ContiguousND<int32_t> cube({5, 10, 15});
cube(2, 5, 10) = 123;
int32_t val = cube(2, 5, 10);
```

**4D access:**
```cpp
cnda::ContiguousND<int64_t> hyper({2, 3, 4, 5});
hyper(1, 2, 3, 4) = 9999;
int64_t val = hyper(1, 2, 3, 4);
```

#### `at()` - Checked Access

```cpp
T& at(Index... indices)
const T& at(Index... indices) const
```

Provides **bounds-checked** access that **always** throws `std::out_of_range` on invalid index, regardless of compile flags.

**When to use:**
- Validating user input
- Debug builds
- When correctness > performance

**Examples:**

```cpp
cnda::ContiguousND<float> arr({3, 4});

try {
    arr.at(1, 2) = 42.0f;  // Valid
    float val = arr.at(1, 2);
    
    arr.at(3, 0) = 1.0f;   // Throws: index 3 >= shape[0]=3
} catch (const std::out_of_range& e) {
    std::cerr << "Error: " << e.what() << "\n";
}
```

#### Performance Comparison

```cpp
#include "cnda/contiguous_nd.hpp"
#include <chrono>

int main() {
    cnda::ContiguousND<float> arr({1000, 1000});
    
    // operator() without bounds check (~10 ns per access)
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 1000; ++i) {
        for (size_t j = 0; j < 1000; ++j) {
            arr(i, j) = i + j;
        }
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    // ~10 ms for 1M accesses
    
    // at() with bounds check (~15 ns per access)
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 1000; ++i) {
        for (size_t j = 0; j < 1000; ++j) {
            arr.at(i, j) = i + j;
        }
    }
    elapsed = std::chrono::high_resolution_clock::now() - start;
    // ~15 ms for 1M accesses (50% overhead)
    
    return 0;
}
```

---

## Memory Layout and Guarantees

### Row-Major Contiguous Layout

CNDA guarantees **row-major (C-style)** contiguous layout:

```
Shape: (3, 4)

Memory layout (row-major):
+-------------------------------------------------------------------------+
| [0,0] [0,1] [0,2] [0,3] [1,0] [1,1] [1,2] [1,3] [2,0] [2,1] [2,2] [2,3] |
+-------------------------------------------------------------------------+
  <----   Row 0   ------> <----   Row 1   ------> <----   Row 2   ------>

Index:  0     1     2     3     4     5     6     7     8     9    10    11
```

**Why row-major?**
- Matches C/C++ native array layout
- Compatible with NumPy's default
- Cache-friendly for typical nested loop access patterns
- Natural for most scientific computing workflows

### Stride Computation

For shape `(d0, d1, d2, ..., dn)`, strides are:

```cpp
strides[n-1] = 1
strides[n-2] = shape[n-1]
strides[n-3] = shape[n-1] * shape[n-2]
...
strides[0] = shape[1] * shape[2] * ... * shape[n-1]
```

**Example:**
```cpp
Shape:   (3, 4, 5)
Strides: (20, 5, 1)

Explanation:
strides[2] = 1                    // innermost dimension
strides[1] = shape[2] = 5
strides[0] = shape[1] * shape[2] = 4 * 5 = 20
```

**Offset formula:**
```cpp
// For arr(i, j, k):
offset = i * strides[0] + j * strides[1] + k * strides[2]
       = i * 20         + j * 5          + k * 1
```

### Memory Alignment

**Alignment guarantee:**
- Elements are aligned according to `alignof(T)`
- For fundamental types: natural alignment (4 bytes for `float`, 8 bytes for `double`)
- For structs: compiler-defined alignment (use `alignas` if needed)

**Example:**
```cpp
struct alignas(16) Vec4 {
    float x, y, z, w;
};

cnda::ContiguousND<Vec4> arr({100});
// Each Vec4 is 16-byte aligned
```

### Cache-Friendly Access Patterns

Row-major layout is optimized for nested loops that iterate outer dimension first:

```cpp
// GOOD: Iterate in memory order
for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
        arr(i, j) = compute(i, j);  // Sequential memory access
    }
}
```

See [Performance Optimization](#performance-optimization) for detailed access patterns and benchmarks.

---

## Ownership and Lifetime

### Self-Owned Arrays

When constructed with just a shape, `ContiguousND` **owns** its memory:

```cpp
cnda::ContiguousND<float> arr({10, 20});
// arr owns a std::vector<float> with 200 elements

// Memory is automatically freed when arr goes out of scope
```

**Characteristics:**
- Automatic memory management (RAII)
- No dangling pointers
- Move-only (no accidental copies)

### Views Over External Memory

When constructed with external data, `ContiguousND` is a **view**:

```cpp
auto buffer = std::make_shared<std::vector<float>>(200);
cnda::ContiguousND<float> view(
    {10, 20},
    buffer->data(),
    buffer  // shared_ptr keeps vector alive
);

// view doesn't own memory - buffer does
// view and buffer share lifetime via shared_ptr
```

**Lifetime guarantee:**
- View keeps `external_owner` alive via `shared_ptr`
- Memory won't be freed as long as view exists
- Multiple views can share the same owner

### Copy, Move, and Assignment

#### Move Semantics (Cheap)

```cpp
cnda::ContiguousND<float> arr1({10, 20});
arr1(0, 0) = 42.0f;

// Move construction - transfers ownership
cnda::ContiguousND<float> arr2(std::move(arr1));
// arr1 is now in moved-from state (don't use)
// arr2 owns the buffer

// Move assignment - transfers ownership
cnda::ContiguousND<float> arr3({5, 5});
arr3 = std::move(arr2);
// arr2 is now in moved-from state
// arr3 owns the buffer
```

**Cost:** O(1) - just pointer swap

#### Copy Semantics (Deleted)

```cpp
cnda::ContiguousND<float> arr1({10, 20});

// Copy construction is deleted
// cnda::ContiguousND<float> arr2(arr1);  // Compile error

// Copy assignment is deleted
// cnda::ContiguousND<float> arr3({5, 5});
// arr3 = arr1;  // Compile error
```

**Why deleted?**
- Prevents accidental O(n) copies
- Forces explicit intent when copying is needed

#### Explicit Copy (Manual)

If you need a deep copy, do it explicitly:

```cpp
cnda::ContiguousND<float> arr1({10, 20});
arr1(0, 0) = 42.0f;

// Explicit deep copy
cnda::ContiguousND<float> arr2({10, 20});
std::memcpy(arr2.data(), arr1.data(), arr1.size() * sizeof(float));

// Or use std::copy
std::copy(arr1.data(), arr1.data() + arr1.size(), arr2.data());
```

### Sharing via Raw Pointers

Multiple functions can operate on the same array via pointers:

```cpp
void process(float* data, size_t rows, size_t cols) {
    // Operate on data
}

void main_computation() {
    cnda::ContiguousND<float> arr({100, 200});
    
    // Pass raw pointer - no ownership transfer
    process(arr.data(), arr.shape()[0], arr.shape()[1]);
    
    // arr still owns the memory
}
```

**Warning:** Ensure the `ContiguousND` outlives any raw pointer usage!

---

## Structured Types (AoS)

CNDA supports **Array-of-Structs (AoS)** layouts for storing multiple related values per grid point.

### Requirements for Struct Types

Structs must be **POD (Plain Old Data)**:

1. **Standard layout**
   - No virtual functions
   - All members have same access control (all public or all private)
   - No base classes (or only standard-layout base classes)

2. **Trivially copyable**
   - No user-defined copy/move constructors
   - No user-defined copy/move assignment
   - No user-defined destructor

**Checked at compile time:**
```cpp
template <class T>
class ContiguousND {
  static_assert(std::is_standard_layout<T>::value,
                "ContiguousND requires T to be standard-layout type");
  static_assert(std::is_trivially_copyable<T>::value,
                "ContiguousND requires T to be trivially copyable");
};
```

### Example: Fluid Simulation Cell

```cpp
#include "cnda/contiguous_nd.hpp"
#include <cstdint>

// Define a fluid cell struct (POD)
struct Cell2D {
    float       u;     // velocity x
    float       v;     // velocity y
    float       pressure;
    std::int32_t flag; // boundary/fluid flag
};

// Verify POD compatibility (optional but recommended)
static_assert(std::is_standard_layout<Cell2D>::value, "Cell2D must be standard layout");
static_assert(std::is_trivially_copyable<Cell2D>::value, "Cell2D must be trivially copyable");

int main() {
    // Create 512×512 grid of fluid cells
    cnda::ContiguousND<Cell2D> grid({512, 512});
    
    // Initialize boundary cells
    for (size_t i = 0; i < 512; ++i) {
        grid(0, i).flag = -1;    // Bottom wall
        grid(511, i).flag = -1;  // Top wall
        grid(i, 0).flag = -1;    // Left wall
        grid(i, 511).flag = -1;  // Right wall
    }
    
    // Initialize fluid region
    for (size_t i = 1; i < 511; ++i) {
        for (size_t j = 1; j < 511; ++j) {
            grid(i, j).u = 1.0f;        // Uniform flow
            grid(i, j).v = 0.0f;
            grid(i, j).pressure = 1.0f; // Atmospheric
            grid(i, j).flag = 1;        // Fluid
        }
    }
    
    // Simulation step: update velocities
    for (size_t i = 1; i < 511; ++i) {
        for (size_t j = 1; j < 511; ++j) {
            if (grid(i, j).flag == 1) {  // Only fluid cells
                // Simple advection
                grid(i, j).u += 0.01f * (grid(i+1, j).u - grid(i-1, j).u);
                grid(i, j).v += 0.01f * (grid(i, j+1).v - grid(i, j-1).v);
            }
        }
    }
    
    return 0;
}
```

### Binary Compatibility with NumPy

For Python interop, C++ struct layout **must match** NumPy structured dtype:

**C++ Side:**
```cpp
struct Cell2D {
    float       u;     // 4 bytes, offset 0
    float       v;     // 4 bytes, offset 4
    std::int32_t flag; // 4 bytes, offset 8
};
// sizeof(Cell2D) = 12 bytes (no padding)
```

**Python Side (NumPy):**
```python
import numpy as np

cell_dtype = np.dtype([
    ('u', '<f4'),      # little-endian float32, offset 0
    ('v', '<f4'),      # little-endian float32, offset 4
    ('flag', '<i4')    # little-endian int32, offset 8
], align=True)
# itemsize = 12 bytes, matches C++ sizeof(Cell2D)
```

### Handling Padding and Alignment

**Problem:** Compiler may add padding for alignment:

```cpp
struct BadStruct {
    float x;       // 4 bytes, offset 0
    int8_t flag;   // 1 byte,  offset 4
    float y;       // 4 bytes, offset 8 (with padding) or 5 (tightly packed)?
};
// Compiler adds 3 bytes padding after flag
// sizeof(BadStruct) = 12 bytes (on most compilers)
```

**Solution 1: Reorder fields to avoid padding**
```cpp
struct GoodStruct {
    float x;       // 4 bytes, offset 0
    float y;       // 4 bytes, offset 4
    int8_t flag;   // 1 byte,  offset 8
    // 3 bytes padding at end (if needed for alignment)
};
// sizeof(GoodStruct) = 12 bytes
```

**Solution 2: Use explicit padding**
```cpp
struct ExplicitStruct {
    float x;       // 4 bytes, offset 0
    int8_t flag;   // 1 byte,  offset 4
    int8_t pad[3]; // 3 bytes, offset 5-7 (explicit padding)
    float y;       // 4 bytes, offset 8
};
// sizeof(ExplicitStruct) = 12 bytes
```

**Avoid `#pragma pack`:** Packing disables alignment and causes performance penalties.

### Verifying Struct Layout

```cpp
#include <iostream>
#include <cstddef>  // for offsetof

struct Cell2D {
    float       u;
    float       v;
    std::int32_t flag;
};

int main() {
    std::cout << "sizeof(Cell2D): " << sizeof(Cell2D) << " bytes\n";
    std::cout << "offsetof(Cell2D, u): " << offsetof(Cell2D, u) << "\n";
    std::cout << "offsetof(Cell2D, v): " << offsetof(Cell2D, v) << "\n";
    std::cout << "offsetof(Cell2D, flag): " << offsetof(Cell2D, flag) << "\n";
    
    // Expected output:
    // sizeof(Cell2D): 12 bytes
    // offsetof(Cell2D, u): 0
    // offsetof(Cell2D, v): 4
    // offsetof(Cell2D, flag): 8
    
    return 0;
}
```

**Match with Python:**
```python
import numpy as np

dtype = np.dtype([('u', '<f4'), ('v', '<f4'), ('flag', '<i4')], align=True)
print(f"NumPy itemsize: {dtype.itemsize} bytes")
for name in dtype.names:
    print(f"  {name}: offset {dtype.fields[name][1]}")

# Expected output:
# NumPy itemsize: 12 bytes
#   u: offset 0
#   v: offset 4
#   flag: offset 8
```

### Common AoS Patterns

#### Pattern 1: Particle System

```cpp
struct Particle {
    double x, y, z;        // position
    double vx, vy, vz;     // velocity
    double mass;
};

cnda::ContiguousND<Particle> particles({10000});

// Initialize
for (size_t i = 0; i < 10000; ++i) {
    particles(i).mass = 1.0;
    particles(i).x = random_double();
    particles(i).y = random_double();
    particles(i).z = random_double();
}

// Update positions
double dt = 0.01;
for (size_t i = 0; i < 10000; ++i) {
    particles(i).x += particles(i).vx * dt;
    particles(i).y += particles(i).vy * dt;
    particles(i).z += particles(i).vz * dt;
}
```

#### Pattern 2: Material Properties Grid

```cpp
struct MaterialPoint {
    float density;
    float temperature;
    float pressure;
    int32_t material_id;
};

cnda::ContiguousND<MaterialPoint> grid({100, 100, 100});

// Set properties
for (size_t i = 0; i < 100; ++i) {
    for (size_t j = 0; j < 100; ++j) {
        for (size_t k = 0; k < 100; ++k) {
            grid(i, j, k).density = 1.0f;
            grid(i, j, k).temperature = 300.0f;
            grid(i, j, k).pressure = 101325.0f;
            grid(i, j, k).material_id = 0;
        }
    }
}
```

---

## Performance Optimization

### 1. Use Raw Pointers for Tight Loops

```cpp
cnda::ContiguousND<float> arr({1000, 1000});

// Slower: operator() has function call overhead
for (size_t i = 0; i < 1000; ++i) {
    for (size_t j = 0; j < 1000; ++j) {
        arr(i, j) = i + j;
    }
}

// Faster: raw pointer eliminates overhead
float* data = arr.data();
for (size_t i = 0; i < 1000; ++i) {
    for (size_t j = 0; j < 1000; ++j) {
        data[i * 1000 + j] = i + j;
    }
}
```

**Speedup:** ~10-20% for simple operations

### 2. Cache-Friendly Access Patterns

```cpp
// GOOD: Row-major access (cache-friendly)
for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
        arr(i, j) = compute(i, j);
    }
}

// BAD: Column-major access (cache-unfriendly)
for (size_t j = 0; j < cols; ++j) {
    for (size_t i = 0; i < rows; ++i) {
        arr(i, j) = compute(i, j);  // Jumps around memory!
    }
}
```

### 3. Compiler Optimization Flags

```bash
# Enable optimizations
g++ -std=c++11 -O3 -march=native -I./include main.cpp -o main

# -O3: aggressive optimization
# -march=native: use CPU-specific instructions (SIMD, etc.)
```

### 4. Disable Bounds Checking in Release

```bash
# Debug build (with bounds checking)
g++ -std=c++11 -g -DCNDA_BOUNDS_CHECK -I./include main.cpp -o main_debug

# Release build (no bounds checking)
g++ -std=c++11 -O3 -I./include main.cpp -o main_release
```

### 5. Preallocate Arrays

```cpp
// BAD: Reallocate in loop
for (int iter = 0; iter < 100; ++iter) {
    cnda::ContiguousND<float> temp({1000, 1000});  // Allocates every iteration!
    compute(temp);
}

// GOOD: Allocate once
cnda::ContiguousND<float> temp({1000, 1000});
for (int iter = 0; iter < 100; ++iter) {
    compute(temp);  // Reuse allocation
}
```

### 6. Use std::fill and std::copy

```cpp
cnda::ContiguousND<float> arr({1000, 1000});

// Slower: Manual loop
for (size_t i = 0; i < arr.size(); ++i) {
    arr.data()[i] = 0.0f;
}

// Faster: std::fill (optimized by compiler)
std::fill(arr.data(), arr.data() + arr.size(), 0.0f);

// Or use memset for zero
std::memset(arr.data(), 0, arr.size() * sizeof(float));
```

---

## Best Practices

### 1. Prefer `operator()` Over `at()` in Hot Paths

```cpp
// Performance-critical code
for (size_t i = 0; i < 1000; ++i) {
    for (size_t j = 0; j < 1000; ++j) {
        arr(i, j) = compute(i, j);  // No bounds check overhead
    }
}

// Validation code
try {
    user_value = arr.at(user_i, user_j);  // Bounds checked
} catch (const std::out_of_range& e) {
    handle_error(e);
}
```

### 2. Use Move Semantics

```cpp
cnda::ContiguousND<float> create_array() {
    cnda::ContiguousND<float> result({1000, 1000});
    // ... fill result ...
    return result;  // Move, not copy
}

auto arr = create_array();  // Cheap move
```

### 3. Document Ownership in APIs

```cpp
// GOOD: Clear ownership
class Simulator {
    cnda::ContiguousND<float> grid_;  // Owns grid

public:
    // Non-owning view
    const float* get_data() const { return grid_.data(); }
    
    // Transfer ownership
    cnda::ContiguousND<float> take_grid() { return std::move(grid_); }
};
```

### 4. Validate POD Types

```cpp
template <typename T>
void validate_aos_type() {
    static_assert(std::is_standard_layout<T>::value, 
                  "T must be standard layout for AoS");
    static_assert(std::is_trivially_copyable<T>::value, 
                  "T must be trivially copyable for AoS");
}

// Usage
validate_aos_type<Cell2D>();
```

### 5. Use const-Correctness

```cpp
void read_only_function(const cnda::ContiguousND<float>& arr) {
    // Can read but not modify
    float val = arr(0, 0);  // OK
    // arr(0, 0) = 1.0f;    // Compile error
}

void modify_function(cnda::ContiguousND<float>& arr) {
    // Can modify
    arr(0, 0) = 1.0f;  // OK
}
```

---

## See Also

- **[QUICKSTART.md](QUICKSTART.md)** - Quick introduction
- **[PYTHON_USER_GUIDE.md](PYTHON_USER_GUIDE.md)** - Python API and NumPy interop
- **[INSTALLATION.md](INSTALLATION.md)** - Installation guide

---

**Version**: 0.1.0 | **Last Updated**: December 2024
