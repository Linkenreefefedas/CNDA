# CNDA Examples

This directory contains concise, practical examples demonstrating key CNDA features.

## Quick Start

Run any example directly:

```bash
# Make sure CNDA is installed
pip install -e .

# Run examples
python examples/01_python_roundtrip.py
python examples/02_cpp_to_numpy.py
python examples/03_aos_struct.py
python examples/04_zero_copy_success_vs_failure.py
python examples/05_error_handling.py
```

## Examples Overview

### 1. Python Round-Trip (`01_python_roundtrip.py`)

**What it demonstrates:**
- Zero-copy memory sharing between NumPy and CNDA
- Round-trip conversion: NumPy → CNDA → NumPy
- Memory address verification

**When to use:**
- Most common use case
- When you need to pass NumPy arrays to C++ without copying
- Quick validation that zero-copy is working

**Key takeaway:** All three objects (NumPy, CNDA, NumPy) share the same memory buffer.

---

### 2. C++ Array → NumPy Export (`02_cpp_to_numpy.py`)

**What it demonstrates:**
- Creating arrays on the C++ side
- Exporting to NumPy for visualization
- Zero-copy vs explicit copy

**When to use:**
- C++ does heavy computation
- Python does visualization/analysis
- Common pattern in scientific workflows

**Key takeaway:** Allocate in C++, compute in C++, visualize in Python without copying.

---

### 3. Array-of-Structs (AoS) (`03_aos_struct.py`)

**What it demonstrates:**
- NumPy structured dtypes with multiple fields
- Binary compatibility with C++ structs
- Fluid simulation grid example

**When to use:**
- Storing multiple values per grid point
- Physics simulations (velocity + pressure + flags)
- Particle systems (position + velocity + mass)
- Material properties grids

**Key takeaway:** Structured dtypes enable efficient storage of heterogeneous data with C++ interop.

---

### 4. Zero-Copy Success vs Failure (`04_zero_copy_success_vs_failure.py`)

**What it demonstrates:**
- When zero-copy works (3 success cases)
- When zero-copy fails (4 failure cases)
- How to fix common issues
- Complete requirements checklist

**When to use:**
- Debugging zero-copy issues
- Understanding layout requirements
- Learning best practices

**Key takeaway:** Zero-copy requires: supported dtype + C-contiguous layout + proper lifetime management.

---

### 5. Error Handling (`05_error_handling.py`)

**What it demonstrates:**
- All 4 exception types (TypeError, ValueError, IndexError, RuntimeError)
- Automatic error recovery with wrapper functions
- Production-ready error handling patterns

**When to use:**
- Building robust production code
- Handling user input
- Implementing fault-tolerant pipelines

**Key takeaway:** Always validate input and handle errors gracefully for reliable code.

---

## Running Requirements

- Python 3.9+
- CNDA installed (`pip install -e .`)
- NumPy
- Matplotlib (for examples 2 and 3)

## Example Output

Each example prints clear output showing:
- Success indicators
- Failure indicators with explanations
- Summaries and takeaways

Some examples also save visualizations as PNG files.

## Next Steps

After running these examples, see:

- **[QUICKSTART.md](../docs/QUICKSTART.md)** - Quick introduction
- **[PYTHON_USER_GUIDE.md](../docs/PYTHON_USER_GUIDE.md)** - Complete Python API
- **[CPP_USER_GUIDE.md](../docs/CPP_USER_GUIDE.md)** - C++ API reference

## C++ Examples

For C++ examples, see:
- **[docs/CPP_USER_GUIDE.md](../docs/CPP_USER_GUIDE.md)** - C++ usage patterns

---

**Version**: 0.1.0 | **Last Updated**: December 2024
