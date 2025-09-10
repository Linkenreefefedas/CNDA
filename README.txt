Contiguous N-Dimensional Array
==============================

A compact C++11/Python library for cache-friendly N-D arrays with struct support and **zero-copy** NumPy interoperability.

Basic Information
-----------------
- GitHub Repository: https://github.com/Linkenreefefedas/Contiguous_N-Dimensional_Array.git
- About: A lightweight C++11/Python library that provides contiguous multi-dimensional arrays with clean indexing, zero-copy NumPy interoperability, and support for both fundamental and composite (struct) types.

Problem to Solve
----------------
In scientific and numerical software, multi-dimensional arrays are fundamental data structures. 
However, existing approaches in C++ and Python interoperation expose several critical issues:

1. **Complex indexing in C++**  
  Multi-dimensional arrays are often represented as raw pointers in contiguous memory. 
  Manual offset arithmetic makes the code cryptic, error-prone, and difficult to maintain.

2. **Performance and memory overhead**  
  Sharing data between C++ and Python usually requires copying buffers. 
  For large-scale simulations, redundant copies result in wasted memory and significant performance degradation.

3. **Lack of composite type support**  
  Many numerical problems store multiple physical variables per grid point (e.g., density, velocity, pressure). 
  Supporting both Array of Structs (AoS) and Struct of Arrays (SoA, columnar) layouts is essential but rarely addressed in a lightweight library.

4. **Unclear API design**  
  Users expect clean and intuitive APIs similar to NumPy’s syntax (e.g., `x[2,3,4]` in Python), 
  but C++ implementations often expose cumbersome pointer arithmetic instead of high-level abstractions.

Prospective Users
-----------------
Users who need a lightweight and efficient way to manage multi-dimensional arrays across C++ and Python, with minimal memory overhead.

System Architecture
-------------------
The system consists of two main layers:

1. **Core (C++11)**
   - `ContiguousND<T>` is a templated class that manages a contiguous memory buffer.
   - Tracks `shape` and `strides` for constant-time index calculation.
   - Provides clean element access via `operator()` instead of manual pointer arithmetic.
   - Supports both fundamental types (float, int) and simple structs (AoS/SoA demos).

2. **Interop (pybind11)**
   - Provides functions `from_numpy()` and `to_numpy()`.
   - Enables zero-copy data sharing between NumPy arrays and `ContiguousND` when layout and type are compatible.
   - Falls back to explicit copies only when required.

**Inputs**
- From Python: an existing `numpy.ndarray` or a requested shape.
- From C++: a shape vector (e.g., `{nx, ny, nz}`).

**Outputs**
- C++: element references or raw pointers through the API.
- Python: NumPy views of the same buffer (no copy if safe).

**Workflow**
1. User creates a new array in C++ or passes an existing NumPy array.
2. The core stores/aliases the buffer contiguously and exposes safe indexing.
3. Operations are performed through C++ methods or Python wrappers.
4. Results can be returned as C++ values or NumPy arrays.

**Constraints (v0.1)**
- Row-major contiguous layout only.
- POD element types (float, double, int32, int64).
- Single-threaded semantics, with clear ownership rules for buffers.
- No slicing/broadcasting in v0.1 (reserved for later versions).

API Description
---------------

- **C++11 core**: a templated container ``cnda::ContiguousND<T>`` for contiguous N-D arrays
  with explicit ``shape`` / ``strides`` and O(1) offset computation.
- **Python binding (pybind11)**: a thin module ``cnda`` that provides
  ``from_numpy()`` (NumPy → C++ view) and ``to_numpy()`` (C++ → NumPy view),
  preferring **zero-copy** when dtype/layout are compatible.

C++ API (namespace ``cnda``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Primary container (header prototype)**

.. code-block:: cpp

  // contiguous_nd.hpp
  #pragma once
  #include <vector>
  #include <cstddef>

  namespace cnda {

  template<class T>
  class ContiguousND {
  public:
    // Construct an owning, row-major contiguous buffer of given shape.
    explicit ContiguousND(std::vector<std::size_t> shape);

    // Basic introspection.
    const std::vector<std::size_t>& shape()   const noexcept;
    const std::vector<std::size_t>& strides() const noexcept;
    std::size_t ndim()  const noexcept;
    std::size_t size()  const noexcept;

    // Raw access.
    T*       data()       noexcept;
    const T* data() const noexcept;

    // Indexing helpers (O(1) offset).
    std::size_t index(std::initializer_list<std::size_t> idx) const;
    T& operator()(std::size_t i);
    T& operator()(std::size_t i, std::size_t j);
    T& operator()(std::size_t i, std::size_t j, std::size_t k);
    // (Variadic overloads may be added later.)
  };

  } // namespace cnda

**Minimal usage (compiles as a prototype)**

.. code-block:: cpp

  #include "contiguous_nd.hpp"
  #include <iostream>
  using cnda::ContiguousND;

  int main() {
    ContiguousND<float> a({3, 4});   // 3x4 contiguous (row-major)
    a(1, 2) = 42.0f;
    std::cout << "a(1,2) = " << a(1,2) << "\n";
    std::cout << a.ndim() << "D, size=" << a.size() << "\n";
    return 0;
  }

Python API (module ``cnda``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Top-level functions & types**

- ``from_numpy(arr: numpy.ndarray) -> ContiguousND_f32/_f64/_i32/_i64``  
  Creates a **zero-copy view** if dtype and layout are compatible; otherwise raises or,
  in a future helper, allows an explicit copy.

- ``ContiguousND_*.to_numpy(copy: bool = False) -> numpy.ndarray``  
  Returns a **NumPy view** (no copy) by default when safe; with ``copy=True`` returns a new array.

**Typical script (round-trip, zero-copy)**

.. code-block:: python

  import numpy as np
  import cnda

  # NumPy → C++ view (no copy)
  x = np.arange(12, dtype=np.float32).reshape(3, 4)
  a = cnda.from_numpy(x)             # view into x's buffer

  # C++ → NumPy view (no copy)
  y = a.to_numpy()                   # shares memory with x
  y[1, 2] = 42
  assert x[1, 2] == 42
  assert y.ctypes.data == x.ctypes.data  # same buffer

**Allocate on C++ side and expose to NumPy**

.. code-block:: python

  import numpy as np
  import cnda

  b = cnda.ContiguousND_f32([2, 3])  # C++-owned contiguous buffer
  B = b.to_numpy()                    # NumPy view (no copy)
  B.fill(7.0)
  assert (B == 7.0).all()

Zero-copy and error semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- `from_numpy(arr, copy=False)` is **zero-copy** only if:
  (1) dtype matches the bound container type,
  (2) array is **C-contiguous (row-major)**,
  (3) lifetime is safe (binding keeps the producer alive).
  Otherwise:
  - if `copy=True`, make an explicit copy;
  - else raise `ValueError/TypeError` (Python) or throw `std::invalid_argument` (C++).
- `to_numpy(copy=False)` returns a **view** with a capsule deleter; set `copy=True` to force duplication.

Notes
~~~~~
- v0.1 scopes: row-major layout, POD element types (``float``, ``double``, ``int32``, ``int64``), single-threaded semantics.
- Future work: slicing/broadcasting, SoA (columnar) adapters, custom allocators, record-dtype/AoS helpers.

Engineering Infrastructure
--------------------------

Automatic build
~~~~~~~~~~~~~~~
Prereqs: CMake (>=3.18), C++11 compiler, Python 3.9+.

**C++ core**
::
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j
  ctest --test-dir build --output-on-failure

**Python binding (after pybind11 lands)**
::
  python -m venv .venv
  # Windows: .\.venv\Scripts\activate
  # Linux/macOS:
  source .venv/bin/activate
  pip install -U pip
  pip install -e .

Version control
~~~~~~~~~~~~~~~
- GitHub public repo; default branch: ``main`` (protected).
- Conventional commits (``feat:``, ``fix:``, ``test:``, ``docs:``, ``chore:``).
- Issues/Milestones aligned to the 8-week schedule.

Testing
~~~~~~~
- **C++**: Catch2/GoogleTest via CTest (shape/strides/index; negative cases).
- **Python**: pytest with NumPy as golden; zero-copy checks via ``ctypes.data``; dtype/contiguity validation.

Documentation
~~~~~~~~~~~~~
- ``README.rst`` = proposal + quickstart; updated via PRs.
- ``docs/`` for zero-copy policy, ownership rules, API examples.

Schedule
--------
8-week plan; Weeks 1–6 focus on core; Weeks 7–8 on integration/delivery. Dates are inclusive.

- Week 1: Set up the repository and CMake build, implement a minimal ``ContiguousND<float>`` with shape/strides/size/data, add initial C++ tests, and draft the README/proposal.
- Week 2: Add multiple scalar types (f32/f64/i32/i64), implement ``operator()`` for 1–3D access, introduce basic error handling, and extend C++ test coverage.
- Week 3: Implement pybind11 bindings for ``from_numpy`` and ``to_numpy``, validate zero-copy interop with NumPy, add a pytest suite, and configure CI.
- Week 4: Strengthen zero-copy safety with clear ownership and lifetime rules, implement an explicit copy path, enable debug-only bounds checks, and expand Python failure-path tests.
- Week 5: Demonstrate arrays-of-structs (AoS) with POD types, add structured-grid usage examples, run micro-benchmarks, and refine the public API.
- Week 6: Prototype a struct-of-arrays (SoA) adapter as optional work, and improve documentation and examples.
- Week 7: Freeze the v0.1 API, finalize edge-case and property-based tests, validate interop on Linux and Windows, and draft slides and demo scripts.
- Week 8: Perform final validation and documentation polish, tag and release v0.1.0, and deliver the presentation and repository submission.

References
----------
- https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
- https://numpy.org/doc/stable/reference/arrays.interface.html