# Analysis: CNDA vs. SimpleArray

## Introduction

This analysis compares the design philosophies and capabilities of **CNDA** 
with the [**SimpleArray**](https://github.com/solvcon/modmesh/blob/master/tests/test_buffer.py) 
implementation. While both provide C++/Python multi-dimensional array containers 
with NumPy interoperability, there are meaningful differences in focus, supported 
features, and design principles.

## Interop Policy and Transparency

**CNDA** centers its design around explicit interoperability policies. The
functions `from_numpy(copy=...)` and `to_numpy(copy=...)` make copy vs.
zero-copy decisions visible at the call site. The library guarantees:

* Zero-copy is attempted only when safe (matching dtype, layout, and lifetime).
* Copy occurs when incompatibilities are detected, with explicit user control.
* Documentation clarifies the exact rules under which each path occurs.

In **SimpleArray**, NumPy interoperability is supported, but the design follows
a NumPy-like style. A user may construct a `SimpleArray` directly from a NumPy
array and call `.clone()` when a copy is needed. This is practical, but it does
not expose a policy boundary at the API level. The difference lies in emphasis:
CNDA prioritizes *predictability and explicitness*, while SimpleArray 
prioritizes *a minimal, NumPy-inspired interface*.

## Stride and Aliasing Documentation

Both libraries permit transpose and stride-based views. In practice, zero-copy
views are possible only when stride reinterpretation preserves a valid memory
layout. Otherwise, a copy must be performed. This behavioral rule is the same 
in both systems.

The **difference is in documentation and policy clarity**:

* CNDA explicitly documents which axis permutations are safe for zero-copy 
  aliasing.
* SimpleArray implements transpose and slicing correctly, but leaves aliasing
  behavior implicit, following NumPy conventions without dedicated documentation.

This distinction matters for developers who require guarantees about when data
will be shared versus copied. CNDA elevates these rules into its public contract.

## Composite Type (AoS) Support

**CNDA** explicitly supports composite (struct) types and Array-of-Structs (AoS)
layouts. Through NumPy structured dtypes, CNDA can map multiple values per grid
point into a single array entry. This enables storing heterogeneous but
logically grouped variables (e.g., velocity components, material parameters)
in a compact, layout-controlled way.

**SimpleArray**, by contrast, focuses on POD scalar types: integers, floats,
booleans, and complex numbers. It does not provide AoS abstractions or 
structured dtype round-tripping. For workflows where structured element types 
are essential, CNDA provides functionality not present in SimpleArray.

## Implementation Style

A further difference lies in implementation style:

* **CNDA Core** is designed to be **header-only**, consisting of C++11 templates
  with no external dependencies beyond the standard library. This pattern makes
  it easy to integrate into other C++ projects by simply including headers. 
* **CNDA Interop** (pybind11 bindings) is compiled as a Python extension. This
  compiled layer enforces the copy/zero-copy policy when bridging to NumPy 
  arrays.
* **SimpleArray**, in contrast, requires compilation of both its C++ core and 
  its pybind11 bindings.

This separation highlights CNDA’s emphasis on minimizing the footprint of its
core while still providing robust compiled interop for Python.

## Design Positioning and Scope

**CNDA** positions itself as a *core infrastructure* library. Its purpose is not
to compete with numerical or scientific libraries, but to guarantee:

* A reliable container for multi-dimensional data.
* Explicit and predictable interop between C++ and Python.
* Memory layout guarantees suitable for building higher-level algorithms.

As a result, CNDA deliberately avoids bundling extensive numerical operations. 
It focuses instead on providing a dependable base on which computation libraries 
can be built.

**SimpleArray**, in contrast, integrates a broad set of numerical features:

* Statistical reductions: sum, min, max, mean, median, variance.
* Elementwise arithmetic and broadcasting.
* Sorting and indexed selection (argsort, take\_along\_axis).
* Ghost cell support for scientific stencil computations.

These features make SimpleArray closer to a *NumPy-lite numerical toolkit*, 
while CNDA is a *foundation for interop and layout control*.

## Conclusion

In summary, CNDA and SimpleArray share common ground as multi-dimensional array
containers with NumPy interoperability. However, CNDA differentiates itself by:

1. **Explicit interop policy**: API-level control over copy vs. zero-copy.
2. **Documented stride/aliasing rules**: Clear guidance on when views share 
   memory.
3. **Composite type support**: Structured dtypes and AoS layouts.
4. **Header-only core**: Simplifies integration, with compiled interop isolated 
   to Python bindings.
5. **Focused scope**: A reliable data container and interop layer, not a full
   numerical library.

SimpleArray provides strong features, particularly for numerical computation, 
but CNDA emphasizes clarity, predictability, and extensibility ininterop and 
memory layout. These distinctions frame CNDA as infrastructure on which 
domain-specific or performance-critical computation systems can be built.
