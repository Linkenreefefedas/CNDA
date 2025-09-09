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
- Provides O(1) index calculation with explicit `shape` and `strides`.  
- Enables zero-copy interoperability between C++ and NumPy through the buffer protocol.  
- Supports both fundamental scalar types and composite (struct) types, with a roadmap for SoA/columnar layouts.  
- Offers a clean, maintainable API for both C++ and Python users.

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
**C++ (namespace ``cnda``).**
- ``template<class T> class ContiguousND``:
  - ``explicit ContiguousND(std::vector<size_t> shape)`` — allocate contiguous buffer
  - ``const std::vector<size_t>& shape() const`` / ``strides() const`` / ``ndim() const`` / ``size() const``
  - ``T* data()`` / ``const T* data() const``
  - ``size_t index(std::initializer_list<size_t> idx) const`` — O(1) offset computation
  - ``T& operator()(size_t i, size_t j, ...)`` — element access (bounds optional in Release)

**Python (via pybind11).**
- ``from_numpy(np_array) -> ContiguousND_f32/_f64/_i32/_i64`` (creates a view when layout is compatible; otherwise explicit copy)
- ``arr.to_numpy(copy: bool = False) -> numpy.ndarray`` (returns a view by default if safe)
- Example::
  
    import numpy as np, cnda
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    a = cnda.from_numpy(x)      # zero-copy view into x's buffer
    y = a.to_numpy()            # returns a view (no copy)
    assert y.ctypes.data == x.ctypes.data

Engineering Infrastructure
--------------------------
**Automatic build system.**
- CMake (>= 3.18)::
  
    cmake -S . -B build
    cmake --build build -j

- (After Python binding is added) Editable install::
  
    python -m venv .venv
    # Windows: .\.venv\Scripts\activate
    # Linux/macOS:
    source .venv/bin/activate
    pip install -U pip
    # pip install -e .

**Version control.**
- Git + GitHub (issues, branches, pull requests).
- Proposal and roadmap maintained in-repo and evolved via PRs.
- Suggested branch scheme:
  - ``feature/core-contiguousnd``
  - ``feature/pybind-zero-copy``
  - ``test/pytest-golden-numpy``

**Testing framework.**
- **C++**: Catch2 or GoogleTest (via ``ctest``) for shape/strides/indexing correctness and AoS samples.
- **Python**: ``pytest`` with NumPy as the **golden** reference (values, shapes, strides, memory sharing).
- Ownership/lifetime stress tests to prevent dangling views or double-frees.

**Documentation.**
- ``README.rst`` (this proposal), plus ``docs/`` for design notes and API examples.
- Optional Sphinx site once APIs stabilize.

**Continuous integration (optional, recommended).**
- GitHub Actions (Linux/Windows) matrix: configure → build → run C++ tests → run ``pytest``.

Schedule
--------
*Assume an 8-week timeline beginning 09/09 (Asia/Taipei). The initial 6 weeks are the planning/implementation phase; weeks 7–8 focus on integration and delivery. Dates denote Monday–Sunday ranges.*

- **Planning phase (6 weeks from 09/09 to 10/20)**

  - **Week 1 (09/09–09/15):** Repository scaffold; minimal ``ContiguousND<float>`` (``shape/strides/index/data``); baseline C++ tests; draft docs.
  - **Week 2 (09/16–09/22):** pybind11 bindings (``from_numpy``/``to_numpy``); Python smoke tests (no-copy paths); CI green.
  - **Week 3 (09/23–09/29):** Zero-copy safety (ownership/lifetime rules); explicit copy switch; documentation refinements; micro-benchmarks vs. raw pointer.
  - **Week 4 (09/30–10/06):** Typed instantiations (f32/f64/i32/i64); debug-only bounds checks; robust error handling; example notebook.
  - **Week 5 (10/07–10/13):** AoS demo (array of structs) and notes on cache locality; public API polish.
  - **Week 6 (10/14–10/20):** (Optional) SoA adapter prototype; CLI inspector; packaging and layout cleanup.

- **Integration & delivery (2 weeks from 10/21 to 11/03)**

  - **Week 7 (10/21–10/27):** Freeze v0.1 API; improve tests (property-based/edge cases); slide draft for presentation.
  - **Week 8 (10/28–11/03):** Final validation; tag **v0.1.0**; presentation materials finalized.

References
----------
- NumPy ``ndarray`` and Python buffer protocol (for zero-copy views).
- pybind11 official documentation and examples (C++/Python interop).
- Literature and engineering discussions on AoS vs. SoA (columnar) layouts and cache behavior in numeric kernels.
- General C++11 RAII and memory management practices for numerical software.