Contiguous N-Dimensional Array
==============================

A compact C++11/Python library for cache-friendly N-D arrays with struct support
and **zero-copy** NumPy interoperability.

Basic Information
-----------------
- GitHub repository: https://github.com/Linkenreefefedas/Contiguous_N-Dimensional_Array.git
- About (one sentence): Contiguous N-D arrays with struct support and zero-copy NumPy interop in C++11/Python.
- Additional documents: All auxiliary documents (proposal, design notes, API examples) will be added under ``docs/``.

Problem to Solve
----------------
**Field/industry.** Numerical and scientific software (HPC, simulation pre/post-processing, mesh/geometry tools) frequently manipulates multi-dimensional arrays, often across a C++ compute core and a Python-facing interface.

**Background.** Raw pointers and manual offset arithmetic make N-D indexing cryptic and error-prone in C++. Crossing the C++/Python boundary commonly incurs unnecessary buffer copies, harming performance and memory footprint. Handling composite types—array of structs (AoS) vs. struct of arrays (SoA, columnar)—further complicates implementation and cache behavior.

**Goal.** Provide a lightweight N-D array core with:
- **Contiguous layout** managed by explicit ``shape`` and ``strides`` (O(1) offset computation).
- **Zero-copy** NumPy interoperability via the buffer protocol/pybind11.
- **Type coverage** for fundamental scalars and composite/struct types, with a path to columnar layouts.
- A clean, maintainable API usable from both C++ and Python.

**Methods / numerical angle.** While not a solver, the library underpins numerics by enforcing deterministic memory layout, predictable cache locality, and constant-time index computation—prerequisites for stable performance in numerical kernels.

Prospective Users
-----------------
- Researchers, students, and engineers building solvers, preprocessors/postprocessors, or teaching code who need a **readable**, **tested**, and **fast** array backbone.
- Python users who want to view/operate on C++ memory from NumPy **without copies** (and vice versa).
- Developers exploring AoS vs. SoA trade-offs and cache effects on numeric kernels.

System Architecture
-------------------
**Workflow overview.**
1. **Input**: Users provide shapes (and optionally strides) from C++ or pass a ``numpy.ndarray`` from Python.
2. **Core**: A templated C++ class manages contiguous storage, shape/stride bookkeeping, and O(1) index calculation.
3. **Interop**: pybind11 bindings create zero-copy views when layouts are compatible; controlled copies are explicit.
4. **Output**: Results are exposed as C++ references or Python ``numpy.ndarray`` views.

**Modules.**
- ``core`` (C++11): ``cnda::ContiguousND<T>`` with RAII buffer, ``shape()``, ``strides()``, ``ndim()``, ``size()``, ``data()``, and offset/element accessors.
- ``interop`` (pybind11): ``from_numpy`` / ``to_numpy`` (zero-copy where possible), typed Python wrappers (e.g., ``ContiguousND_f32``).
- ``cli`` (optional): Inspect/dump shape/strides, validate sharing (for demos and debugging).

**Constraints/assumptions (v0.1).**
- Focus on correctness, zero-copy safety, and clear API before advanced features.
- Slicing/broadcasting and full SoA adapters are **post-v0.1** goals.
- Single-threaded semantics initially; document ownership/lifetime rules for safe interop.

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