Contiguous N-Dimensional Array
==============================

A compact C++11/Python library for cache-friendly N-D arrays with struct support
and **zero-copy** NumPy interoperability.

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

**Goal:**  
To design and implement a contiguous N-dimensional array library that:  

-  Provides O(1) index calculation with explicit `shape` and `strides`.  
-  Enables zero-copy interoperability between C++ and NumPy through the buffer protocol.  
-  Supports both fundamental scalar types and composite (struct) types, with a roadmap for SoA/columnar layouts.  
-  Offers a clean, maintainable API for both C++ and Python users.

Prospective Users
-----------------
The library is intended for both educational and practical use in the field of numerical and scientific computing.

1. **Researchers and engineers in HPC and simulation**  
   They require high-performance data structures to handle large grids or meshes. 
   This library enables them to prototype algorithms in Python while relying on C++ for efficient memory management and computation, without incurring unnecessary data copies.

2. **Students and educators**  
   In courses on numerical methods, scientific computing, or software engineering, 
   this project provides a clear and maintainable reference implementation of multi-dimensional arrays. 
   It helps learners understand the trade-offs of memory layouts (AoS vs SoA) and the importance of cache-aware design.

3. **Python developers bridging with C++**  
   Developers who already use NumPy can seamlessly integrate this library to extend functionality or optimize performance-critical sections in C++ while preserving a NumPy-like API style.

System Architecture
-------------------
This section describes how the system ingests data, represents and transforms it in memory, exposes interfaces to users, and returns results. It also states the constraints and the modular decomposition.

High-level workflow
~~~~~~~~~~~~~~~~~~~
The following depicts the typical execution paths from both C++ and Python:

.. code-block:: text

   [User Code]                                     [Library Internals]
   -----------                                     -------------------
   (Python) np.ndarray --from_numpy()------------> interop/pybind: validate layout
                                                   : (zero-copy view if compatible)
                                                   core/ContiguousND<T>: adopt external buffer
                                                   :
                                                   user ops via Python wrapper (indexing, access)
                                                   :
   to_numpy(copy=False) <-------------------------- return NumPy view (no copy if safe)

   (C++)   shape vector --ContiguousND<T>(shape)-> core: allocate contiguous buffer (RAII)
                                                   :
                                                   user ops via C++ API (operator(), data())
                                                   :
   expose as NumPy (optional) --to_numpy()-------> interop/pybind: create NumPy view

Inputs and outputs
~~~~~~~~~~~~~~~~~~
- **Inputs**
  - From **Python**: existing ``numpy.ndarray`` (dtype, shape, strides) or requested shape for new arrays.
  - From **C++**: a shape vector (e.g., ``{nx, ny, nz}``) and (optionally) initial values.
- **Outputs**
  - C++ references or values via ``operator()`` / ``data()``.
  - **Python NumPy views** (no copies when possible) via ``to_numpy(copy=False)``.
  - Explicit copies only when requested or when layout is incompatible with zero-copy rules.

Data model and memory layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Contiguous storage** in row-major order (C-style) by default.
- **Shape/strides** are tracked explicitly; offset computation is **O(1)**.
- **Element type** ``T`` is templated (``float``, ``double``, ``int32_t``, ``int64_t``; extensible to POD structs).
- **RAII ownership**: the core class either *owns* an internal buffer (allocated by the library) or *aliases* an external buffer (e.g., NumPy) without taking ownership, governed by clear lifetime rules.
- **Alignment**: default allocator alignment the platform provides; specialized allocators may be plugged in later.

Interfaces
~~~~~~~~~~
- **C++ (namespace ``cnda``)**:
  - ``template<class T> class ContiguousND`` with:
    - ``ContiguousND(std::vector<size_t> shape)`` (owning allocation)
    - ``shape()``, ``strides()``, ``ndim()``, ``size()``, ``data()``, ``const data()``
    - ``size_t index(std::initializer_list<size_t>)`` (offset calc)
    - ``T& operator()(size_t i, size_t j, ...)`` (element access; debug bounds optionally)
- **Python (module ``cnda`` via pybind11)**:
  - ``from_numpy(np.ndarray) -> ContiguousND_*`` (view if compatible)
  - ``obj.to_numpy(copy: bool = False) -> np.ndarray`` (view by default if safe)
  - Thin, typed wrappers (e.g., ``ContiguousND_f32``) expose properties and indexing helpers.

Module decomposition
~~~~~~~~~~~~~~~~~~~~
- **core/** (C++11)
  - ``contiguous_nd.hpp/.cpp``: the ``ContiguousND<T>`` container, shape/stride handling, offset logic, ownership policy.
  - ``detail/``: small utilities (bounds-check helpers for Debug builds, dtype traits, error categories).
- **interop/** (pybind11)
  - ``bindings.cpp``:
    - Validates NumPy dtype/contiguity/strides.
    - Constructs zero-copy views when layouts match.
    - Creates NumPy arrays that alias internal buffers (with a custom capsule deleter when owning).
- **python/** (optional helpers)
  - ``__init__.py``: import shims, convenience utilities (e.g., dtype mapping).
- **cli/** (optional)
  - Inspect/convert arrays (print shape/strides, verify aliasing).
- **tests/**
  - **C++** unit tests for index math, shape/stride invariants, AoS samples.
  - **Python** tests with NumPy as golden (values, shapes, strides, buffer aliasing).
- **docs/**
  - Proposal, design notes, API examples.

Zero-copy policy and safety
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Zero-copy eligibility** requires: compatible dtype, contiguous (or accepted stride pattern), and lifetime safety.
- **Adopting external buffers** (from NumPy): library stores a *non-owning* pointer with a reference/capsule to keep the Python object alive while views exist.
- **Exposing internal buffers** (to NumPy): NumPy receives a view that references the library-owned memory; a capsule deleter is attached to prevent premature free and ensure a single point of truth.
- **Explicit copies** happen when: dtype/stride is incompatible, or the user sets ``copy=True``.

Error handling and validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Precondition checks:
  - Shape non-empty, product fits in ``size_t`` and addressable memory.
  - Dtype compatibility and stride sanity for zero-copy interop.
- Error reporting:
  - C++: exceptions with specific categories (invalid argument, allocation failure, interop mismatch).
  - Python: mapped to ``ValueError``/``TypeError`` with precise diagnostics.
- Debug builds:
  - Optional bounds checks in ``operator()``.
  - Assertions for invariant preservation (shape, strides, size).

Performance considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~
- **O(1)** offset arithmetic (no virtual calls, constexpr-friendly helpers where applicable).
- Cache-aware contiguous layout; examples and micro-benchmarks included.
- Avoid temporaries and needless copies in hot paths (move semantics where beneficial).

Constraints and assumptions (v0.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Row-major only (column-major may be added later).
- Single-threaded semantics; users may externally synchronize for multi-threading.
- POD/standard-layout types for ``T``; non-POD support deferred.
- No slicing/broadcasting in v0.1 (reserved for subsequent versions).
- Platform: modern compilers supporting C++11; Python 3.9+.

Extensibility roadmap (post v0.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Slicing/broadcasting views; subarray and reshaping utilities.
- Columnar (SoA) adapter layer and traits.
- Custom allocators (aligned, pinned) and small-vector optimization for shapes.
- Multi-thread safety guards and shared-ownership views.

API Description
---------------
This section shows how to program against the library from **C++** and **Python**, including
core types, function signatures, zero-copy interop, and example scripts.

C++ API (namespace ``cnda``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Primary container
^^^^^^^^^^^^^^^^^
.. code-block:: cpp

  namespace cnda {

  template <class T>
  class ContiguousND {
  public:
    // --- ctors & ownership ---
    explicit ContiguousND(std::vector<size_t> shape);          // owning buffer
    ContiguousND(T* data,
                 std::vector<size_t> shape,
                 std::vector<size_t> strides,
                 bool owning);                                  // advanced (alias or take ownership)

    // --- shape & layout ---
    const std::vector<size_t>& shape()   const noexcept;
    const std::vector<size_t>& strides() const noexcept;
    size_t ndim()  const noexcept;     // rank
    size_t size()  const noexcept;     // total elements

    // --- data access ---
    T*       data()       noexcept;
    const T* data() const noexcept;

    // --- indexing ---
    size_t index(std::initializer_list<size_t> idx) const;      // O(1) offset
    T& operator()(size_t i);                                    // 1-D
    T& operator()(size_t i, size_t j);                          // 2-D
    T& operator()(size_t i, size_t j, size_t k);                // 3-D
    // (variadic helper overloads may be provided)

    // --- misc ---
    void fill(const T& value);                                  // utility
  };

  } // namespace cnda

Basic usage
^^^^^^^^^^^
.. code-block:: cpp

  #include "contiguous_nd.hpp"
  #include <iostream>
  using cnda::ContiguousND;

  int main() {
    ContiguousND<float> a({3,4});   // 3x4, row-major contiguous
    a(1,2) = 42.0f;
    std::cout << a(1,2) << "\n";    // prints 42
    std::cout << a.ndim() << "D size=" << a.size() << "\n";
    return 0;
  }

Array-of-Structs (AoS) demo
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cpp

  struct Cell {
    double rho, u, p;  // density, velocity, pressure
  };

  cnda::ContiguousND<Cell> grid({128, 128});
  grid(10, 20) = Cell{1.0, 0.3, 101325.0};
  double pressure = grid(10, 20).p;

Error handling (C++)
^^^^^^^^^^^^^^^^^^^^
- Invalid shapes/overflow → throw ``std::invalid_argument``.
- Mismatched strides/layout in advanced ctor → ``std::invalid_argument``.
- Debug builds may enable bounds checks in ``operator()`` (configurable).

Python API (module ``cnda`` via pybind11)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
High-level functions & types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- ``from_numpy(arr: numpy.ndarray) -> ContiguousND_f32/_f64/_i32/_i64``  
  Creates a **zero-copy view** if dtype/contiguity are compatible; otherwise raises, or copies when explicitly requested in a separate helper (e.g., ``from_numpy_copy``).
- ``ContiguousND_*.to_numpy(copy: bool = False) -> numpy.ndarray``  
  Returns a NumPy **view** by default (no copy) when safe; with ``copy=True`` returns a new array.

Typical script
^^^^^^^^^^^^^^
.. code-block:: python

  import numpy as np
  import cnda

  # 1) Start from NumPy → C++ view (no copy)
  x = np.arange(12, dtype=np.float32).reshape(3, 4)
  a = cnda.from_numpy(x)          # view into x's buffer
  y = a.to_numpy()                # returns a view back
  assert y.ctypes.data == x.ctypes.data

  # 2) Modify through the C++-backed view
  y[1, 2] = 42
  assert x[1, 2] == 42            # reflected in original array

  # 3) Allocate on C++ side (exposed to Python)
  b = cnda.ContiguousND_f32([2, 3])   # constructor bound in pybind11
  z = b.to_numpy()                     # NumPy view, shape (2, 3)
  z.fill(7.0)
  assert (z == 7.0).all()

Working with dtypes and shapes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

  # dtype-gated constructors (examples):
  cnda.ContiguousND_f32([64, 64])
  cnda.ContiguousND_f64([32, 32, 16])

  # Interop checks:
  # - from_numpy expects contiguous C-order and matching dtype.
  # - to_numpy returns a view; set copy=True to force duplication.

Array-of-Structs in Python (demo concept)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For educational purposes, an AoS example can be exposed via a typed wrapper or
record-dtype mapping (advanced; subject to v0.1 scope). A conceptual sketch:

.. code-block:: python

  # Pseudocode / optional extension:
  Cell = np.dtype([('rho', 'f8'), ('u', 'f8'), ('p', 'f8')])
  arr  = np.zeros((128, 128), dtype=Cell)     # NumPy record array
  # If bindings support record dtypes, cnda.from_numpy(arr) could alias it.

Zero-copy rules (Python)
^^^^^^^^^^^^^^^^^^^^^^^^
- **from_numpy**: zero-copy only if:
  - dtype matches the bound container type,
  - array is C-contiguous (or an accepted stride pattern),
  - lifetime is secured (binding retains a capsule/ref).
- **to_numpy**: returns a view over library-owned memory with a capsule deleter; use ``copy=True`` to force duplication.

Exceptions (Python)
^^^^^^^^^^^^^^^^^^^
- Incompatible dtype/strides/contiguity → ``ValueError``.
- Shape/size overflow or invalid input → ``ValueError`` / ``TypeError``.

End-to-end example (script)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

  """
  Example: finite-difference style stencil using ContiguousND_f32 from Python.
  """
  import numpy as np, cnda

  nx, ny = 256, 256
  a = cnda.ContiguousND_f32([nx, ny])      # C++-owned buffer
  A = a.to_numpy()                          # NumPy view (no copy)

  # initialize
  X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing='ij')
  A[:] = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)

  # simple 5-point Laplacian (periodic)
  out = np.empty_like(A)
  out[1:-1,1:-1] = (
      -4.0*A[1:-1,1:-1]
      + A[2:,1:-1] + A[:-2,1:-1]
      + A[1:-1,2:] + A[1:-1,:-2]
  )

  # pass back into C++ (still a view)
  b = cnda.from_numpy(out)                  # zero-copy view over NumPy array
  B = b.to_numpy()                          # view again; same buffer pointer
  assert B.ctypes.data == out.ctypes.data

Notes on extensibility
~~~~~~~~~~~~~~~~~~~~~~
- Future versions may add:
  - slicing/broadcasting views,
  - SoA (columnar) adapters and traits,
  - custom allocators (alignment/pinning),
  - property-based tests for API contracts.

Engineering Infrastructure
--------------------------
This section outlines how the project will be engineered end-to-end: build and packaging,
version control practices, testing strategy, documentation plan, and (optionally) CI.

Automatic build system
~~~~~~~~~~~~~~~~~~~~~~
**C++ build (CMake).**
- Minimum: CMake >= 3.18; a C++11-capable compiler (GCC/Clang/MSVC).
- Out-of-source configure & build::

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j

- Run C++ tests (via CTest)::

    ctest --test-dir build --output-on-failure

**Python binding (pybind11) – editable install (later milestone).**
- Project will expose a Python module (``cnda``) via pybind11.
- After bindings land, provide a pyproject-based editable install::

    python -m venv .venv
    # Windows: .\.venv\Scripts\activate
    # Linux/macOS:
    source .venv/bin/activate
    pip install -U pip
    pip install -e .    # builds the extension in-place

- Wheels for Linux/Windows can be produced via ``cibuildwheel`` (post-v0.1).

Artifacts & layout (planned)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Top-level files**: ``CMakeLists.txt``, ``pyproject.toml`` (later), ``README.rst``, ``LICENSE``.
- **Sources**:
  - ``core/contiguous_nd.hpp`` (+ ``.cpp`` if needed)
  - ``interop/bindings.cpp`` (pybind11)
- **Tests**:
  - ``tests/cpp/*`` (Catch2/GoogleTest)
  - ``tests/py/*`` (pytest)

Version control
~~~~~~~~~~~~~~~
**Repository & branching.**
- Host: GitHub (public).
- Mainline: ``main`` (protected).
- Topic branches:
  - ``feature/core-contiguousnd``
  - ``feature/pybind-zero-copy``
  - ``test/pytest-golden-numpy``
  - ``docs/sphinx-site`` (optional)
- Workflow: feature branch → pull request → code review → squash-merge.

**Commit practices.**
- Conventional messages (e.g., ``feat:``, ``fix:``, ``docs:``, ``test:``, ``chore:``).
- Small, reviewable commits; one logical change per PR.

**Issue tracking & milestones.**
- GitHub Issues for tasks/bugs; Milestones aligned with the weekly schedule.
- Labels: ``core``, ``interop``, ``api``, ``perf``, ``tests``, ``docs``.

Testing framework
~~~~~~~~~~~~~~~~~
**C++ unit tests.**
- Framework: **Catch2** or **GoogleTest** (invoked via CTest).
- Coverage:
  - Shape/stride invariants and O(1) index math.
  - Element access correctness (including debug-bounds mode).
  - AoS demo types (POD/standard-layout structs).
- Negative tests:
  - Invalid shapes/overflow; layout/stride mismatches.

**Python tests.**
- Framework: **pytest**, with **NumPy** as the *golden* reference.
- Coverage:
  - ``from_numpy`` / ``to_numpy`` zero-copy paths (pointer equality via ``ctypes.data``).
  - Dtype/contiguity validation (raise on incompatible inputs).
  - Round-trip semantics (NumPy → C++ view → NumPy view).
- Optional property-based tests (``hypothesis``) for shapes and random indexing.

**Benchmarks (lightweight).**
- Micro-benchmarks to compare raw-pointer loops vs. ``ContiguousND`` offsetting.
- Track stable timings locally; CI perf gates are **not** required for v0.1.

Documentation
~~~~~~~~~~~~~
**In-repo docs.**
- ``README.rst``: serves as the proposal and quickstart.
- ``docs/``: design notes, API examples, zero-copy policy, lifetime rules.
- Example notebooks (Python) illustrating typical numerical workflows.

**Sphinx site (optional, post-v0.1).**
- Autodoc for C++/Python APIs (with ``breathe``/``doxygen`` integration if needed).
- Publish via GitHub Pages.

Continuous integration (optional but recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **GitHub Actions** matrix (Linux/Windows):
  1) Checkout, set up Python
  2) Configure & build with CMake (Release)
  3) Run C++ tests via CTest
  4) Install test deps (``pytest``, ``numpy``) and run Python tests
- Artifacts: store build logs and (optionally) wheels for inspection.

Schedule
--------
The project is planned for 8 weeks. The first 6 weeks focus on core development and planning,
while the last 2 weeks are dedicated to integration, validation, and presentation. The schedule
is an initial estimate and will be adjusted as development progresses.

Planning phase (6 weeks)
~~~~~~~~~~~~~~~~~~~~~~~~

Week 1:
- Set up repository structure and build system (CMake).
- Implement minimal ``ContiguousND<float>`` with shape, strides, size, and raw data access.
- Add initial C++ unit tests for shape and indexing.
- Draft README and proposal documents.

Week 2:
- Add support for multiple scalar types (float32, float64, int32, int64).
- Implement ``operator()`` for clean element access in 1D, 2D, and 3D.
- Introduce basic error handling (invalid shape, overflow).
- Extend C++ test coverage.

Week 3:
- Develop pybind11 bindings for ``from_numpy`` and ``to_numpy``.
- Validate zero-copy interop with NumPy for contiguous arrays.
- Add Python pytest suite with NumPy as golden reference.
- Configure CI pipeline (GitHub Actions).

Week 4:
- Strengthen zero-copy safety: lifetime management, ownership policies, capsule deleters.
- Implement explicit copy option in API for incompatible inputs.
- Add debug-only bounds checking in C++.
- Expand Python test cases to cover failure paths.

Week 5:
- Demonstrate support for arrays of structs (AoS) with POD types.
- Add examples for structured grid usage in C++ and Python.
- Run micro-benchmarks comparing pointer arithmetic vs. ``ContiguousND`` offsetting.
- Refine API consistency.

Week 6:
- Optional: prototype Struct-of-Arrays (SoA/columnar) adapter.
- Add CLI utility for inspecting array shape, strides, and memory address.
- Improve documentation with design notes and usage examples.

Integration & delivery (2 weeks)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Week 7:
- Freeze v0.1 API and finalize test coverage (edge cases, property-based tests).
- Validate interop across Linux and Windows.
- Draft presentation slides and demo scripts.

Week 8:
- Perform final validation and polish documentation.
- Tag and release version 0.1.0.
- Deliver final presentation and repository submission.

References
----------
- NumPy ``ndarray`` and Python buffer protocol (for zero-copy views).
- pybind11 official documentation and examples (C++/Python interop).
- Literature and engineering discussions on AoS vs. SoA (columnar) layouts and cache behavior in numeric kernels.
- General C++11 RAII and memory management practices for numerical software.