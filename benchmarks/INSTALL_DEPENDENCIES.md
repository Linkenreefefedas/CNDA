# Installing Benchmark Dependencies

This document explains how to install optional dependencies for comprehensive benchmarking.

## Core Dependencies (Always Required)

These are automatically handled by CMake:
- **Catch2** - C++ testing and benchmarking framework (auto-fetched)
- **pybind11** - Python bindings (from conda environment)
- **NumPy** - Python array library (from conda environment)

## Optional Comparison Libraries

### 1. pytest-benchmark (for Python benchmarks)

**What is it?**
- Python benchmarking extension for pytest
- Required for `bench_numpy_interop.py` and `bench_vs_numpy.py`

**Installation:**
```bash
# Using pip
pip install pytest-benchmark

# Or using conda
conda install -c conda-forge pytest-benchmark
```

**Verification:**
```bash
pytest --version
python -c "import pytest_benchmark; print(pytest_benchmark.__version__)"
```

## Building with Optional Dependencies

### Building Benchmarks

To build the C++ benchmarks:

```bash
cd build
cmake .. -DCNDA_BUILD_BENCHMARKS=ON
cmake --build . --config Release
```

## Benchmark Execution

### With All Dependencies

```bash
cd benchmarks
.\run_all_benchmarks.ps1  # Windows
# OR
./run_all_benchmarks.sh   # Linux/Mac
```

### Individual Benchmarks

```bash
# Core CNDA benchmarks (no dependencies)
cd build/benchmarks/Release
.\bench_core.exe --benchmark-samples 50

# Comparison benchmarks
.\bench_comparison.exe --benchmark-samples 50

# Python benchmarks (requires pytest-benchmark)
cd ../../..
pytest benchmarks/bench_numpy_interop.py --benchmark-only
```

## Recommended Setup for Complete Benchmarking

```bash
# Create conda environment with all dependencies
conda create -n cnda_bench python=3.9
conda activate cnda_bench

# Install core dependencies
conda install -c conda-forge pybind11 numpy pytest

# Install Python benchmarking
pip install pytest-benchmark

# Build CNDA with benchmarks
cd CNDA
mkdir build && cd build
cmake .. -DCNDA_BUILD_BENCHMARKS=ON
cmake --build . --config Release

# Run all benchmarks
cd benchmarks
.\run_all_benchmarks.ps1  # Windows
```

## Troubleshooting

### pytest-benchmark not working

**Symptom:** Python benchmarks fail with "No module named 'pytest_benchmark'"

**Solutions:**
1. Install in active environment: `pip install pytest-benchmark`
2. Check pytest version: `pytest --version` (needs 3.0+)
3. Verify import: `python -c "import pytest_benchmark"`

## Performance Considerations

### Build Configuration

Always use Release mode for accurate benchmarks:

```bash
cmake --build . --config Release
```

### Sample Size

Adjust sample size based on operation cost:
- Fast operations (< 1ms): `--benchmark-samples 100`
- Medium operations (1-10ms): `--benchmark-samples 50`
- Slow operations (> 10ms): `--benchmark-samples 20`

### System Configuration

For reproducible benchmarks:
1. Close unnecessary applications
2. Disable CPU frequency scaling (if possible)
3. Run multiple times and average results
4. Report system specs (CPU, RAM, compiler version)

## Expected Results

### vs std::vector
- **CNDA should be**: 1.5-2x faster than `vector<vector<T>>`, similar to flat `vector<T>`

### vs NumPy
- **Zero-copy overhead**: < 1µs (CNDA's key strength)
- **Vectorized ops**: Equal (CNDA uses NumPy backend)
- **Python loops**: Both slow (use vectorized operations)

## Documentation

For more details, see:
- [BENCHMARKS.md](../docs/BENCHMARKS.md) - Comprehensive benchmark guide
- [COMPARISON_GUIDE.md](COMPARISON_GUIDE.md) - Comparison strategy and interpretation
- [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) - Quick execution reference
