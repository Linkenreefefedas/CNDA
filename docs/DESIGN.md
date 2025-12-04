# CNDA Internal Design Documentation

**Version:** 0.1.0  
**Last Updated:** December 2, 2025  
**Audience:** Code reviewers, professors, contributors, and developers interested in understanding CNDA's internal architecture

---

## Table of Contents

1. [Overview](#1-overview)
2. [Memory Layout Design](#2-memory-layout-design)
3. [Zero-Copy Semantics](#3-zero-copy-semantics)
4. [Error & Safety Model](#4-error--safety-model)
5. [Type System Architecture](#5-type-system-architecture)
6. [Performance Considerations](#6-performance-considerations)

---

## 1. Overview

CNDA (Contiguous N-Dimensional Array) is designed as a minimal-overhead library for managing multi-dimensional arrays with explicit memory layout guarantees and zero-copy interoperability between C++ and Python/NumPy. This document explains the internal design decisions that make CNDA both safe and performant.

### 1.1 Design Philosophy

The design follows three core principles:

1. **Explicitness over implicitness**: All copy/zero-copy behavior is explicit, preventing silent performance issues
2. **Safety without overhead**: Memory safety checks are configurable (compile-time and runtime)
3. **Cache-friendly by default**: Row-major contiguous layout optimizes for modern CPU cache hierarchies

### 1.2 Key Design Goals

- **Predictable memory layout**: All arrays use contiguous row-major (C-order) storage
- **Zero runtime dependencies**: Core C++ library is header-only with no external dependencies
- **Type safety**: Compile-time checks for POD types; runtime validation for structured types
- **Interop correctness**: Strict validation of dtype, strides, and alignment for NumPy compatibility

---

## 2. Memory Layout Design

### 2.1 Row-Major Definition

CNDA uses **row-major (C-order)** layout exclusively. For a 3D array with shape `[N₀, N₁, N₂]`, elements are stored such that the rightmost index varies fastest in memory:

```
Memory order: [0,0,0], [0,0,1], [0,0,2], ..., [0,0,N₂-1], [0,1,0], [0,1,1], ...
```

**Why row-major?** NumPy compatibility, C/C++ convention, and cache efficiency (see Section 2.5 for detailed performance analysis).

### 2.2 Stride Calculation

Strides represent the **number of elements** (not bytes) to skip when incrementing along each dimension.

#### Stride Formula

For row-major layout with shape `[s₀, s₁, ..., sₙ₋₁]`:

```cpp
stride[n-1] = 1                              // Rightmost dimension
stride[i]   = stride[i+1] × shape[i+1]       // For i from n-2 down to 0
```

**Example:** Shape `[3, 4, 5]`
```
stride[2] = 1         // Last dimension
stride[1] = 1 × 5 = 5
stride[0] = 5 × 4 = 20
```

#### Implementation

```cpp
void compute_metadata() noexcept {
    m_ndim = m_shape.size();
    m_size = 1;
    for (std::size_t d : m_shape) {
        m_size *= d;
    }

    // Row-major, ELEMENT-BASED strides
    m_strides.assign(m_ndim, 0);
    if (m_ndim > 0) {
        m_strides[m_ndim - 1] = 1;
        for (std::size_t k = m_ndim; k-- > 1; ) {
            m_strides[k - 1] = m_strides[k] * m_shape[k];
        }
    }
}
```

**Key decision:** Strides stored in **elements**, not bytes, to simplify indexing and match NumPy's stride semantics (after converting from byte-strides).

### 2.3 Index Computation

#### General N-D Formula

For indices `[i₀, i₁, ..., iₙ₋₁]`, the flat offset is:

```
offset = i₀ × stride[0] + i₁ × stride[1] + ... + iₙ₋₁ × stride[n-1]
```

#### Optimized Specializations

For common dimensionalities, CNDA uses direct computation to avoid stride array lookups:

**2D Specialization:**
```cpp
// For 2D: offset = i * shape[1] + j
return m_data[i * m_shape[1] + j];
```

**3D Specialization:**
```cpp
// For 3D: offset = i * (shape[1] * shape[2]) + j * shape[2] + k
const std::size_t dim2 = m_shape[2];
return m_data[i * (m_shape[1] * dim2) + j * dim2 + k];
```

**Performance Impact:** These specializations eliminate:
- Array indexing overhead (1-2 memory accesses)
- Multiplication by stride[1]=1 for the last dimension
- Loop overhead in the general case

Benchmarks show **5-15% improvement** for 2D/3D access patterns (see `docs/BENCHMARKS.md`).

### 2.4 AoS vs SoA: Design Decisions

#### Array-of-Structures (AoS)

**Current implementation** (v0.1) focuses on AoS:

```cpp
struct Vec3f {
    float x, y, z;  // 12 bytes contiguous
};

ContiguousND<Vec3f> positions({1000});  // 1000 Vec3f structs, each 12 bytes
```

**Memory layout:**
```
[x₀, y₀, z₀][x₁, y₁, z₁][x₂, y₂, z₂]...
```

**Advantages:**
- **Locality for element operations**: Reading/writing all fields of one element is cache-friendly
- **Natural C++ representation**: Matches struct layout in memory
- **NumPy structured dtype support**: Direct mapping to NumPy's structured arrays
- **Simple indexing**: Single pointer arithmetic operation

**Use cases:**
- Particle systems (position + velocity + mass per particle)
- Simulation cells (multiple properties per grid point)
- Geometric primitives (vertices with position + normal + UV)

#### Structure-of-Arrays (SoA)

**Future work** (v0.2+) will add SoA support:

```cpp
struct Vec3f_SoA {
    ContiguousND<float> x;  // All x components
    ContiguousND<float> y;  // All y components  
    ContiguousND<float> z;  // All z components
};
```

**Memory layout:**
```
[x₀, x₁, x₂, ..., xₙ][y₀, y₁, y₂, ..., yₙ][z₀, z₁, z₂, ..., zₙ]
```

**Advantages:**
- **Vectorization-friendly**: SIMD operations can process arrays of single components
- **Selective access**: Load only needed fields (e.g., only x-coordinates)
- **Better cache utilization** for component-wise operations

**Implementation strategy:**
1. Add `SoAView<T>` wrapper class
2. Provide conversion utilities: `aos_to_soa()`, `soa_to_aos()`
3. Support NumPy record array interop with split fields

### 2.5 Why Contiguous ND-Array Matters for Cache Efficiency

#### Cache Line Fundamentals

Modern CPUs use cache hierarchies:
- **L1 cache**: ~32KB, ~4 cycles latency
- **L2 cache**: ~256KB, ~12 cycles latency
- **L3 cache**: ~8MB, ~40 cycles latency
- **DRAM**: ~16GB, ~200 cycles latency

Cache line size: typically **64 bytes** (16 floats or 8 doubles)

#### Sequential Access Patterns

**Row-major sequential iteration** (cache-friendly):
```cpp
for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
        arr(i, j) = compute();  // Accesses consecutive memory
    }
}
```

**Benefit:** Each cache line fetch (64 bytes) provides data for 16 consecutive floats. Effective bandwidth utilization approaches **90-95%**.

**Column-major iteration** (cache-unfriendly):
```cpp
for (size_t j = 0; j < M; ++j) {
    for (size_t i = 0; i < N; ++i) {
        arr(i, j) = compute();  // Jumps by stride[0] elements
    }
}
```

**Cost:** Each access may miss cache (stride[0] × sizeof(T) likely exceeds cache line). Effective bandwidth drops to **10-20%**.

#### Benchmark Evidence

From `bench_core.cpp` results:

| Operation | Sequential (row-major) | Random Access | Column-Major |
|-----------|----------------------|---------------|--------------|
| 1000×1000 float write | 8 ns/element | 42 ns/element | 38 ns/element |
| **Cache efficiency** | 95% | 20% | 22% |

**Conclusion:** Contiguous row-major layout is not just a convention—it's a **5-10× performance multiplier** for typical numerical workloads.

#### Why This Matters for Scientific Computing

Most scientific codes have:
1. **Spatial locality**: Neighboring grid points interact
2. **Sequential sweeps**: Iterating over entire domains
3. **Stencil operations**: Accessing small neighborhoods

Contiguous layout ensures these patterns hit cache efficiently, making CNDA a better foundation for performance-critical numerical software than non-contiguous or column-major alternatives.

---

## 3. Zero-Copy Semantics

### 3.1 Design Motivation

The core challenge in C++/Python interop is avoiding unnecessary copies while maintaining memory safety. CNDA solves this with **explicit zero-copy semantics** backed by strict validation.

### 3.2 Strict Dtype Validation

#### Fundamental Types

For primitive types (`float`, `double`, `int32`, `int64`), dtype validation is straightforward:

```cpp
template<typename T>
cnda::ContiguousND<T> from_numpy_impl(const py::array_t<T>& arr, bool copy) {
    // pybind11's py::array_t<T> already ensures dtype match
    // Additional checks: contiguity, strides
}
```

pybind11's type system guarantees `py::array_t<float>` matches `np.float32`.

#### Structured Types (AoS)

For composite types, validation is critical to prevent **silent data corruption**:

```cpp
template<typename T>
void validate_structured_dtype(const py::dtype& dtype, const std::string& type_name);
```

**Validation steps:**

1. **Is structured?** Check `dtype.kind() == 'V'` (void/structured)
2. **Field count match?** Compare `dtype.attr("names").size()` with expected
3. **Field names match?** Exact string comparison for each field
4. **Field types match?** Byte-level dtype comparison (endianness, size)

**Example:** `Cell2D` validation
```python
# Expected: [('u', '<f4'), ('v', '<f4'), ('flag', '<i4')]

# ✓ Valid
np.dtype([('u', np.float32), ('v', np.float32), ('flag', np.int32)])

# ✗ Invalid - field name mismatch
np.dtype([('u', np.float32), ('v', np.float32), ('state', np.int32)])
# TypeError: Cell2D field 2 must be 'flag', got 'state'

# ✗ Invalid - dtype mismatch  
np.dtype([('u', np.float32), ('v', np.float32), ('flag', np.int64)])
# TypeError: Cell2D.flag requires int32, got int64
```

**Why so strict?**

Lenient validation could allow:
```python
# NumPy array with fields ['u', 'v', 'pressure'] (12 bytes)
# C++ expects Cell2D with fields [u, v, flag]
# C++ would read 'pressure' as integer flag → garbage data
```

Strict validation prevents this class of bugs at the boundary.

### 3.3 Stride and Contiguity Validation

Even with correct dtype, layout must match:

```cpp
// Check C-contiguous flag
if (!(arr.flags() & py::array::c_style)) {
    throw ValueError("from_numpy(copy=False) requires C-contiguous array");
}

// Additional stride validation
for (ssize_t i = 0; i < arr.ndim(); ++i) {
    auto stride_bytes = arr.strides(i);
    auto elem_size = sizeof(T);
    
    // Stride must be multiple of element size
    if (stride_bytes % elem_size != 0) {
        throw ValueError("Incompatible strides");
    }
    
    // Stride must match row-major layout
    if (stride_bytes / elem_size != expected_strides[i]) {
        throw ValueError("Non-standard strides");
    }
}
```

**Why check strides explicitly?**

Some NumPy operations produce arrays that report `c_contiguous=True` but have unusual strides (e.g., after transpose-then-copy). Explicit validation catches these edge cases.

### 3.4 Python Capsule Deleter Pattern

The **capsule deleter** is the mechanism that enables safe zero-copy from C++ to Python.

#### Problem Statement

When returning a C++ array to Python as a NumPy view:
```python
arr = cnda.ContiguousND_f32([100, 100])
np_view = arr.to_numpy(copy=False)
del arr  # Warning: C++ object destroyed, but np_view still exists!
```

Without proper lifetime management, `np_view` becomes a **dangling pointer**.

#### Solution: PyCapsule Ownership

```cpp
py::array_t<T> ContiguousND<T>::to_numpy(bool copy) {
    if (copy) {
        // Deep copy: NumPy owns the memory
        return py::array_t<T>(shape, data);  // pybind11 copies
    } else {
        // Zero-copy: Keep C++ object alive via capsule
        auto* py_obj_ptr = new py::object(self_obj);  // Store Python object handle
        
        py::capsule capsule(py_obj_ptr, [](void* p) {
            // Deleter called when NumPy array is destroyed
            delete static_cast<py::object*>(p);
        });
        
        return py::array_t<T>(
            shape_ssize,
            strides_bytes,
            data_ptr,
            capsule  // NumPy holds reference to capsule
        );
    }
}
```

#### Lifetime Flow Diagram

```
C++ ContiguousND object (owned by Python)
    │
    ├─> Internal buffer (std::vector<T> or external pointer)
    │
    └─> Python object handle (py::object)
            │
            └─> Stored in PyCapsule
                    │
                    └─> Referenced by NumPy array

When NumPy array is destroyed:
    1. NumPy releases capsule reference
    2. PyCapsule destructor calls deleter lambda
    3. Deleter destroys py::object handle
    4. Python refcount decrements
    5. If refcount == 0, C++ destructor runs
    6. Internal buffer deallocated
```

**Key insight:** The capsule holds a Python object handle, not the C++ object directly. This works through Python's reference counting system rather than C++ RAII.

#### Alternative Approaches Considered

**Shared pointer approach** (rejected): `shared_ptr` refcount is C++-only and doesn't integrate with Python GC, causing potential leaks. Our capsule-based solution integrates with Python's refcounting for transparent lifetime management.

### 3.5 C++ Lifetime → Python Lifetime Flow

#### Scenario 1: NumPy → C++ (Zero-Copy)

```python
x = np.array([[1, 2], [3, 4]], dtype=np.float32)
arr = cnda.from_numpy(x, copy=False)
```

**Lifetime:**
1. `from_numpy` validates dtype/strides
2. Creates `ContiguousND<float>` with external pointer `x.data`
3. Stores `py::array` handle in `m_external_owner` (shared_ptr)
4. While `arr` exists, Python reference to `x` is held
5. When `arr` is destroyed, `m_external_owner` destructor releases `x` reference

**Result:** `x` cannot be garbage collected until `arr` is destroyed.

#### Scenario 2: C++ → Python (Zero-Copy)

```python
arr = cnda.ContiguousND_f32([100, 100])
y = arr.to_numpy(copy=False)
del arr  # Safe!
```

**Lifetime:**
1. `to_numpy(copy=False)` creates NumPy array with capsule
2. Capsule stores `py::object(arr)` handle
3. `del arr` decrements Python refcount, but capsule still holds reference
4. `arr` C++ object remains alive
5. When `y` is destroyed, capsule deleter releases handle
6. Python refcount → 0, C++ destructor runs

**Result:** Memory remains valid as long as any NumPy view exists.

#### Scenario 3: Error Case (Copy Required)

Non-contiguous arrays (e.g., Fortran-order) raise `ValueError` with clear guidance. **Why fail rather than silent copy?** Performance transparency—a silent copy could introduce 10-100× slowdowns without user awareness.

---

## 4. Error & Safety Model

### 4.1 Exception Hierarchy

CNDA uses standard C++ exceptions with clear semantic mapping to Python:

| C++ Exception | Python Exception | Use Case |
|---------------|-----------------|----------|
| `std::out_of_range` | `IndexError` | Out-of-bounds array access |
| `std::invalid_argument` | `ValueError` | Shape/layout/stride mismatch |
| `std::runtime_error` | `RuntimeError` | Lifetime/ownership issues |
| `pybind11::type_error` | `TypeError` | Dtype mismatch |

#### Implementation

```cpp
// In module.cpp: Register exception translators
py::register_exception_translator([](std::exception_ptr p) {
    try {
        if (p) std::rethrow_exception(p);
    } catch (const std::out_of_range &e) {
        PyErr_SetString(PyExc_IndexError, e.what());
    }
});
```

**Rationale:** Python users expect `IndexError` for out-of-bounds, not generic `RuntimeError`. This mapping provides Pythonic error behavior.

### 4.2 When to Throw `std::invalid_argument`

```cpp
// Empty shape
ContiguousND({});  // ✗ throws std::invalid_argument

// Rank mismatch
arr.at({i, j});  // for 3D array → std::out_of_range

// Non-contiguous from_numpy
from_numpy(fortran_array, copy=False);  // ✗ ValueError in Python
```

**Python mapping:**
```python
try:
    arr = cnda.ContiguousND_f32([])
except ValueError as e:
    print(e)  # "Shape cannot be empty"
```

### 4.3 Bounds Check Modes

CNDA provides **three levels** of bounds checking:

#### Level 1: No Bounds Check (Default)

```cpp
T& operator()(size_t i, size_t j) {
    return m_data[i * m_shape[1] + j];  // No validation
}
```

**Use case:** Performance-critical inner loops after validation

**Risk:** Undefined behavior on out-of-bounds access

#### Level 2: Compile-Time Bounds Check

```bash
cmake -DCNDA_BOUNDS_CHECK=ON -B build
```

```cpp
#ifdef CNDA_BOUNDS_CHECK
    if (i >= m_shape[0] || j >= m_shape[1]) {
        throw std::out_of_range("operator(): index out of bounds");
    }
#endif
return m_data[i * m_shape[1] + j];
```

**Use case:** Development, debugging, testing

**Overhead:** ~15-25% slowdown (from benchmarks)

#### Level 3: Runtime Bounds Check (`at()`)

```cpp
arr.at(i, j);  // Always checks bounds
```

**Use case:** Sporadic access in non-critical paths

**Overhead:** ~20-30% slowdown compared to unchecked `operator()`

#### Recommendation Matrix

| Context | Recommended Mode | Rationale |
|---------|-----------------|-----------|
| Development | Compile-time (`-DCNDA_BOUNDS_CHECK=ON`) | Catch bugs early |
| Testing | Compile-time | Validate correctness |
| Production (validated code) | No bounds check | Maximum performance |
| Production (user input) | Runtime `at()` | Safety first |
| Python bindings | Runtime (always) | Python users expect safe behavior |

### 4.4 Python Bindings Safety

**All Python indexing uses bounds-checked paths** (runtime validation in `compute_offset`). Python users expect safety—unlike C++ where developers opt into checks, Python should never silently corrupt memory.

See Section 4.3 for the three bounds checking levels and their performance trade-offs.

### 4.5 Lifetime Issue Prevention

**Summary:** CNDA prevents dangling references through three mechanisms:

1. **Move semantics**: C++ objects transfer ownership to Python via `py::cast(std::move(arr))`
2. **Capsule deleter**: For zero-copy views (see Section 3.4 for details)
3. **External owner tracking**: `shared_ptr<void>` holds references to source arrays

See Section 3.4-3.5 for detailed lifetime flow diagrams and implementation patterns.

---

## 5. Type System Architecture

### 5.1 POD Requirements

```cpp
template <class T>
class ContiguousND {
    static_assert(std::is_standard_layout<T>::value,
                  "ContiguousND requires T to be standard-layout type");
    static_assert(std::is_trivially_copyable<T>::value,
                  "ContiguousND requires T to be trivially copyable");
    // ...
};
```

**Why these requirements?**

1. **`std::is_standard_layout`**: Guarantees predictable memory layout for AoS types
2. **`std::is_trivially_copyable`**: Allows `memcpy`, essential for NumPy interop

**Rejected alternative:** Using non-POD types with constructors/destructors would require:
- Element-wise construction on allocation
- Element-wise destruction on deallocation
- No `memcpy` → 10-100× slower copy operations

### 5.2 AoS Type Registry System

To support user-defined AoS types without template bloat, CNDA uses a runtime registry:

```cpp
class AoSTypeRegistry {
public:
    static void register_type(const std::string& type_name, 
                             const std::vector<FieldSpec>& fields);
    static std::vector<FieldSpec> get_fields(const std::string& type_name);
};
```

**Registration example:**

```cpp
AoSTypeRegistry::register_type("Vec2f", {
    FieldSpec("x", py::dtype::of<float>()),
    FieldSpec("y", py::dtype::of<float>())
});
```

**Validation usage:**

```cpp
template<>
void validate_structured_dtype<Vec2f>(const py::dtype& dtype, const std::string& type_name) {
    auto fields = AoSTypeRegistry::get_fields("Vec2f");
    validate_aos_dtype(dtype, type_name, fields);  // Generic validation
}
```

**Benefit:** Adding new AoS types requires:
1. Define C++ struct
2. Register with `AoSTypeRegistry::register_type()`
3. Instantiate `bind_contiguous_nd<NewType>()`

No template specialization needed for validation logic.

### 5.3 Field Access API Design

For AoS types, direct struct member access is cumbersome in Python:

```python
# ✗ Not supported: arr[i, j].x (returns Python object, not mutable reference)
```

**Solution:** Provide accessor methods:

```cpp
// C++
template<>
void bind_aos_fields<Vec2f>(py::class_<ContiguousND<Vec2f>>& cls) {
    cls.def("get_x", [](const ContiguousND<Vec2f>& self, py::tuple indices) {
        size_t offset = compute_offset(self, tuple_to_indices(indices));
        return self.data()[offset].x;
    });
    cls.def("set_x", [](ContiguousND<Vec2f>& self, py::tuple indices, float val) {
        size_t offset = compute_offset(self, tuple_to_indices(indices));
        self.data()[offset].x = val;
    });
}
```

**Python usage:**

```python
arr = cnda.ContiguousND_Vec2f([100, 100])
arr.set_x((10, 20), 3.14)
x = arr.get_x((10, 20))
```

**Future enhancement (v0.2):** Return mutable references to enable:
```python
arr[i, j].x = 3.14  # Requires pybind11 custom type caster
```

---

## 6. Performance Considerations

### 6.1 Optimization Strategies

#### Inline Indexing Functions

```cpp
template <typename Index1, typename Index2>
inline T& operator()(Index1 i0, Index2 i1) {
    // Direct computation, no function call overhead
    return m_data[static_cast<std::size_t>(i0) * m_shape[1] + 
                  static_cast<std::size_t>(i1)];
}
```

**Impact:** Inlining eliminates function call overhead (~2-5 cycles) and enables further compiler optimizations.

#### Loop Unrolling for Small Dimensions

For 2D/3D, the compiler can unroll stride computation:

```cpp
// General case (loop):
for (size_t i = 0; i < ndim; ++i) {
    offset += indices[i] * strides[i];
}

// 2D specialization (unrolled):
offset = i * stride[0] + j * stride[1];
```

**Measured speedup:** 5-15% for 2D/3D access

#### Stride Storage Trade-off

**Current design:** Store strides in `std::vector`

**Alternative:** Compute strides on-the-fly

```cpp
// On-the-fly (not implemented)
size_t stride(size_t dim) const {
    size_t s = 1;
    for (size_t k = dim + 1; k < m_ndim; ++k) {
        s *= m_shape[k];
    }
    return s;
}
```

**Analysis:**

| Approach | Memory | Access Cost | Better For |
|----------|--------|-------------|-----------|
| Stored strides | +O(ndim) | O(1) | Repeated access |
| Computed strides | O(1) | O(ndim) | Infrequent access |

**Decision:** Store strides. The memory cost is negligible (8 bytes × ndim), and access patterns dominate (millions of accesses vs. dozens of stride computations).

### 6.2 Copy Elision and Move Semantics

```cpp
ContiguousND(ContiguousND&& other) noexcept;           // Move constructor
ContiguousND& operator=(ContiguousND&& other) noexcept; // Move assignment

// Copy operations deleted
ContiguousND(const ContiguousND&) = delete;
ContiguousND& operator=(const ContiguousND&) = delete;
```

**Rationale:**

1. **Prevent accidental copies:** Large arrays should never be copied implicitly
2. **Force explicit copy:** Use `copy=True` in interop functions
3. **Enable RVO:** Return by value invokes move, not copy

**Example:**

```cpp
ContiguousND<float> make_array() {
    ContiguousND<float> arr({1000, 1000});
    // ... fill arr ...
    return arr;  // Move, not copy (RVO)
}
```

### 6.3 Alignment Considerations

Modern CPUs benefit from aligned memory access:

- **16-byte alignment:** SSE/NEON
- **32-byte alignment:** AVX
- **64-byte alignment:** AVX-512, cache line

**Current implementation:** Uses `std::vector<T>`, which guarantees `alignof(T)` alignment.

**Future work (v0.2):** Add aligned allocation:

```cpp
template <size_t Alignment>
class AlignedContiguousND : public ContiguousND<T> {
    // Use custom allocator with aligned_alloc
};
```

**Use case:** Explicit SIMD vectorization, cache-line-aligned rows for HPC codes.

### 6.4 Benchmarking Insights

Key findings from `benchmarks/bench_core.cpp`:

1. **Sequential access**: CNDA matches raw pointer performance (within 2%)
2. **Random access**: 50% of sequential (expected, cache-limited)
3. **Bounds checking overhead**: 15-25% (compile-time), 20-30% (runtime `at()`)
4. **Memory bandwidth**: Saturates at ~40 GB/s (sequential), ~8 GB/s (random) on test hardware

**Conclusion:** Zero-abstraction overhead for cache-friendly patterns. Overhead only appears when mixing with cache-unfriendly algorithms.

---

## 7. Future Design Directions

### 7.1 SoA Support (v0.2)

```cpp
// Proposed API
template <typename T>
class SoAView {
    std::vector<ContiguousND<field_type>> fields;
};

SoAView<Vec3f> positions = aos_to_soa(aos_array);
```

### 7.2 Strided Views (v0.3)

```cpp
// Enable slicing
auto slice = arr.view({slice(0, 100), slice(10, 20)});
```

**Challenge:** Maintain zero-copy while allowing non-contiguous strides.

### 7.3 Multi-Threading Safety (v0.4)

Current: Single-threaded semantics.

Proposed:
- `read_view()`: Immutable multi-threaded read
- `write_lock()`: Exclusive write access
- `partition()`: Split array for parallel processing

### 7.4 GPU Interop (v0.5)

```cpp
// Zero-copy with CUDA/HIP
auto cuda_view = arr.to_cuda_device(device_id, copy=false);
```

**Requirement:** Pinned host memory for zero-copy transfers.

---

## 8. References

### Internal Documentation
- `docs/PYTHON_USER_GUIDE.md`: Python API and usage patterns
- `docs/BENCHMARKS.md`: Performance validation and optimization evidence
- `docs/CPP_USER_GUIDE.md`: C++ API reference

### External Resources
- [NumPy Array Interface](https://numpy.org/doc/stable/reference/arrays.interface.html)
- [pybind11 NumPy Support](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html)
- [PyCapsule Documentation](https://docs.python.org/3/c-api/capsule.html)
- [C++ Standard Layout](https://en.cppreference.com/w/cpp/language/data_members#Standard_layout)

### Academic References
- Drepper, Ulrich. "What Every Programmer Should Know About Memory." 2007.
  - Cache hierarchies, stride effects, bandwidth analysis
- Williams, Samuel et al. "Roofline: An Insightful Visual Performance Model." 2009.
  - Performance modeling for memory-bound codes

---

## Appendix A: Complete Error Code Reference

### C++ Exception → Python Mapping

```cpp
// std::out_of_range → IndexError
arr.at(100, 200);  // IndexError: at(): index out of bounds

// std::invalid_argument → ValueError  
ContiguousND<float>({});  // ValueError: Shape cannot be empty

// pybind11::type_error → TypeError
from_numpy(int64_array, "Vec2f");  // TypeError: Vec2f requires structured dtype

// std::runtime_error → RuntimeError
// (Rare: internal errors, e.g., failed capsule creation)
```

### Error Message Design Philosophy

1. **What went wrong**: Clear statement of the error
2. **Why it's wrong**: Explanation of the constraint violated
3. **How to fix**: Suggested resolution

**Example:**
```
ValueError: from_numpy(copy=False) requires C-contiguous (row-major) array.
The input array has Fortran-order (column-major) layout.
Use copy=True to force a copy, or ensure the input array is C-contiguous with np.ascontiguousarray().
```

---

## Appendix B: Memory Layout Examples

### Example 1: 2D Array

```python
arr = ContiguousND_f32([3, 4])
```

**Memory:**
```
Offset: 0  1  2  3  4  5  6  7  8  9  10 11
Data:  [00][01][02][03][10][11][12][13][20][21][22][23]
Index: (0,0)(0,1)(0,2)(0,3)(1,0)(1,1)(1,2)(1,3)(2,0)(2,1)(2,2)(2,3)
```

**Strides:** `[4, 1]`  
**Index formula:** `offset = i * 4 + j`

### Example 2: 3D Array

```python
arr = ContiguousND_f32([2, 3, 4])
```

**Memory:**
```
Offset: 0-3      4-7      8-11     12-15    16-19    20-23
Layer:  (0,0,*)  (0,1,*)  (0,2,*)  (1,0,*)  (1,1,*)  (1,2,*)
```

**Strides:** `[12, 4, 1]`  
**Index formula:** `offset = i * 12 + j * 4 + k`

### Example 3: AoS (Vec3f)

```python
arr = ContiguousND_Vec3f([2, 2])
```

**Memory (each cell is 12 bytes):**
```
Offset: 0          12         24         36
Data:   [x,y,z]00  [x,y,z]01  [x,y,z]10  [x,y,z]11
```

**Strides (in elements):** `[2, 1]`  
**Index formula:** `offset = i * 2 + j` (each element is a Vec3f struct)

---

**End of DESIGN.md**
