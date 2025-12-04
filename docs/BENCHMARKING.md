# CNDA Benchmarking Guide

This guide covers everything you need to run comprehensive performance benchmarks for CNDA.

## Quick Start

```bash
# Install dependencies
pip install pytest-benchmark

# Build with benchmarks enabled
cd build
cmake .. -DCNDA_BUILD_BENCHMARKS=ON
cmake --build . --config Release

# Run all benchmarks
cd benchmarks
.\run_all_benchmarks.ps1  # Windows
./run_all_benchmarks.sh   # Linux/Mac
```

---



## Table of Contents

1. [Dependencies](#dependencies)
2. [Building Benchmarks](#building-benchmarks)
3. [Running Benchmarks](#running-benchmarks)
4. [Performance Considerations](#performance-considerations)
5. [Interpreting Results](#interpreting-results)
6. [Troubleshooting](#troubleshooting)

---

## Dependencies

### Core Dependencies (Always Required)

Automatically handled by CMake:
- **Catch2** (v3.0+) - C++ benchmarking framework (auto-fetched)
- **pybind11** (v2.6.0+) - Python bindings
- **NumPy** - Python array library

### Optional: pytest-benchmark (Python benchmarks)

```bash
pip install pytest-benchmark
# Verify: python -c "import pytest_benchmark; print('OK')"
```

### Recommended Complete Setup

```bash
# Create dedicated conda environment
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
```

---

## Building Benchmarks

### CMake Configuration

Enable benchmarks during configuration:

```bash
mkdir build && cd build
cmake .. -DCNDA_BUILD_BENCHMARKS=ON
```

Additional useful flags:
```bash
cmake .. \
  -DCNDA_BUILD_BENCHMARKS=ON \
  -DCNDA_BOUNDS_CHECK=OFF \      # Disable bounds checking for max performance
  -DCMAKE_BUILD_TYPE=Release     # Linux/Mac
```

### Compile

```bash
# Windows (Visual Studio)
cmake --build . --config Release

# Linux/Mac (Make/Ninja)
cmake --build .
```

### Verify Build

```bash
# Windows: ls build\benchmarks\Release\bench_*.exe
# Linux/Mac: ls build/benchmarks/bench_*
# Expected: bench_core, bench_comparison, bench_aos
```

---

## Running Benchmarks

### Option 1: Run All Benchmarks (Recommended)

**Windows:**
```bash
cd benchmarks
.\run_all_benchmarks.ps1
```

**Linux/Mac:**
```bash
cd benchmarks
./run_all_benchmarks.sh
```

This script runs:
1. All C++ benchmarks with optimized settings
2. Python benchmarks (if pytest-benchmark is available)
3. Generates summary report

### Option 2: Individual C++ Benchmarks

```bash
cd build/benchmarks/Release  # Windows
cd build/benchmarks          # Linux/Mac

# Core CNDA operations
.\bench_core.exe --benchmark-samples 50

# Comparison with std::vector
.\bench_comparison.exe --benchmark-samples 50

# Array-of-Structs performance
.\bench_aos.exe --benchmark-samples 50
```

**Catch2 Benchmark Options:**
- `--benchmark-samples N` - Number of measurement samples (default: 100)
- `--benchmark-resamples N` - Number of resamples for statistics (default: 100000)
- `--benchmark-confidence-interval 0.95` - Confidence level
- `--benchmark-warmup-time 100` - Warmup time in milliseconds
- `--benchmark-no-analysis` - Skip statistical analysis

### Option 3: Python Benchmarks

```bash
# From repository root
pytest benchmarks/bench_numpy_interop.py --benchmark-only

# With more samples
pytest benchmarks/bench_numpy_interop.py --benchmark-only --benchmark-min-rounds=100

# Save results
pytest benchmarks/bench_numpy_interop.py --benchmark-only --benchmark-save=results
```

**pytest-benchmark Options:**
- `--benchmark-only` - Skip tests, run only benchmarks
- `--benchmark-min-rounds=N` - Minimum measurement rounds
- `--benchmark-max-time=S` - Maximum time per benchmark (seconds)
- `--benchmark-save=NAME` - Save results to JSON
- `--benchmark-compare` - Compare with previous results

---

## Performance Considerations

### Build Configuration

**Critical: Always use Release mode** (Debug is 10-100× slower). See "Building Benchmarks" section for commands.

### Sample Size Guidelines

Adjust based on operation cost:

| Operation Duration | Recommended Samples | Example |
|-------------------|-------------------|---------|
| < 1 µs | 1000-5000 | Single element access |
| 1-100 µs | 100-1000 | Small array operations |
| 100 µs - 10 ms | 50-100 | Medium array operations |
| > 10 ms | 20-50 | Large array operations |

**Example:**
```bash
# Fast operations (element access)
.\bench_core.exe --benchmark-samples 1000

# Medium operations (10k element sum)
.\bench_core.exe --benchmark-samples 100

# Slow operations (1M element copy)
.\bench_core.exe --benchmark-samples 50
```

### System Configuration

For reproducible benchmarks:

#### 1. Close Unnecessary Applications
- Close browsers, IDEs, background apps
- Disable antivirus real-time scanning (temporarily)
- Stop indexing services

#### 2. CPU Frequency Scaling (Advanced)

**Windows:**
```powershell
# Set high-performance power plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

**Linux:**
```bash
# Disable CPU frequency scaling
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

#### 3. Process Priority (Advanced)

Run benchmarks with elevated priority:
```bash
# Windows PowerShell (as Administrator)
Start-Process .\bench_core.exe -Verb RunAs

# Linux
nice -n -20 ./bench_core
```

#### 4. Multiple Runs

Run benchmarks 3-5 times and average:
```bash
for i in 1..5 { .\bench_core.exe --benchmark-samples 100 }
```

---

## Interpreting Results

### C++ Benchmark Output (Catch2)

Example output:
```
benchmark name                       samples       iterations    estimated
                                     mean          low mean      high mean
                                     std dev       low std dev   high std dev
-------------------------------------------------------------------------------
ContiguousND: Element access         100           1234          12.34 ms
                                     98.7 ns       97.2 ns       101.3 ns
                                     8.2 ns        6.1 ns        12.4 ns
```

**Key metrics:**
- **mean**: Average execution time (primary metric)
- **low mean / high mean**: Confidence interval (95% by default)
- **std dev**: Standard deviation (variability)

**What to look for:**
- Low std dev (< 10% of mean) = consistent performance
- Warning: High std dev (> 20% of mean) = investigate noise sources
- Narrow confidence interval = reliable measurement

### Python Benchmark Output (pytest-benchmark)

Example output:
```
Name (time in us)                  Min       Max      Mean    StdDev    Median
--------------------------------------------------------------------------------
test_zero_copy[CNDA]              1.23      2.45     1.35     0.12      1.32
test_zero_copy[NumPy]            45.67     89.12    52.34     8.45     50.23
```

**Key metrics:**
- **Mean**: Average execution time
- **Median**: Middle value (less sensitive to outliers)
- **StdDev**: Variability

### Benchmark Results

The following results were measured on:
- **CPU**: Intel Core (3.6 GHz class)
- **OS**: Windows 11
- **Compiler**: MSVC 2022 (Release mode)
- **Build Flags**: `/O2` optimization, bounds checking OFF
- **Samples**: 30 runs per benchmark

#### 1. Core Operations Performance

| Operation | Array Size | Mean Time | Std Dev | Notes |
|-----------|-----------|-----------|---------|-------|
| **Construction** |
| `ContiguousND<float>` | 100×100 | 759 ns | 285 ns | Object creation overhead |
| `ContiguousND<double>` | 100×100 | 1.25 µs | 612 ns | Slightly slower (8-byte type) |
| `ContiguousND<float>` | 1000×1000 | 805 µs | 81 µs | 1M elements allocation |
| **Sequential Access** |
| Raw pointer write | 1000×1000 | 520 µs | 44 µs | Baseline performance |
| `operator()` write | 1000×1000 | 741 µs | 24 µs | 1.43× raw pointer |
| `operator()` read | 1000×1000 | 770 µs | 12 µs | Similar to write |
| **Random Access** |
| Random read | 10,000 ops | 7.91 µs | 1.94 µs | ~791 ns per access |
| **Bounds Checking** |
| `operator()` (no check) | 10,000 ops | 7.56 µs | 181 ns | Fast path |
| `at()` (with check) | 10,000 ops | 433 µs | 7.95 µs | 57× slower (check overhead) |
| **3D Arrays** |
| Write 100×100×100 | 1M elements | 996 µs | 42 µs | Good cache locality |
| Read 100×100×100 | 1M elements | 776 µs | 20 µs | Faster than write |
| **Memory Bandwidth** |
| Sequential write | 10M floats (40MB) | 8.34 ms | 978 µs | ~4.80 GB/s |
| Sequential read | 10M floats (40MB) | 7.72 ms | 119 µs | ~5.18 GB/s |
| Copy | 10M floats (40MB) | 5.99 ms | 231 µs | ~6.68 GB/s |

**Key Observations:**
- Construction overhead is minimal (~760 ns for 100×100 arrays)
- `operator()` adds ~43% overhead vs raw pointers (acceptable for safety)
- Warning: Bounds checking (`at()`) is 57× slower - use only when necessary
- Memory bandwidth: 4.8-6.7 GB/s (good cache utilization)

#### 2. Comparison: CNDA vs std::vector

| Layout | Access Pattern | Mean Time | Speedup | Notes |
|--------|---------------|-----------|---------|-------|
| **2D Access (1000×1000)** |
| CNDA `operator(i,j)` | Sequential | 745 µs | 1.04× | Baseline |
| `std::vector<T>` manual indexing | Sequential | 774 µs | 1.00× | Comparable |
| Raw pointer | Sequential | 526 µs | 1.42× | Fastest (unsafe) |
| **3D Access (100×100×100)** |
| CNDA `arr(i,j,k)` | Sequential | 1.42 ms | 0.73× | Contiguous memory |
| Nested `vec[i][j][k]` | Sequential | 1.04 ms | 1.00× | Fragmented memory |

**Analysis:**
- CNDA is comparable to flat `std::vector` (within 4%)
- Warning: 3D nested vectors appear faster due to cache effects (small test size)
- CNDA provides type safety without major performance penalty

#### 3. Array-of-Structs (AoS) Performance

| Struct Type | Operation | Mean Time | Notes |
|-------------|-----------|-----------|-------|
| **Vec2f (8 bytes)** |
| Write all fields | 1000×1000 | 4.46 ms | Writing x, y |
| Read magnitude | 1000×1000 | 832 µs | `sqrt(x² + y²)` |
| Read x field only | 1000×1000 | 784 µs | Single field access |
| **Cell2D (12 bytes)** |
| Velocity update | 100×100 | 1.06 ms | Fluid advection step |
| **Particle (56 bytes)** |
| Position update | 100,000 | 467 µs | Euler integration |
| Gravity force | 100,000 | 441 µs | Force calculation |
| **Cache Efficiency** |
| Vec2f (8 bytes) | 1M sequential | 816 µs | Best cache usage |
| Vec3f (12 bytes) | 1M sequential | 869 µs | 1.06× slower |
| Particle (56 bytes) | 1M sequential | 2.64 ms | 3.24× slower (cache misses) |
| **AoS vs SoA** |
| AoS: both fields | 1000×1000 | 831 µs | Access x and y |
| AoS: single field (x) | 1000×1000 | 790 µs | Slightly better |
| SoA: single field (x) | 1000×1000 | 766 µs | 3.1% faster (best for selective access) |

**Key Insights:**
- Smaller structs (8-12 bytes) have excellent cache performance
- Warning: Larger structs (56+ bytes) show 3.2× slower sequential access
- SoA layout is 3.1% faster for single-field access
- AoS is better when accessing all fields together

#### 4. Expected Performance Summary

| Operation Category | Performance Range | Recommendation |
|-------------------|-------------------|----------------|
| **Element Access** | 100-800 ns | Use `operator()` for balance of safety/speed |
| **Array Construction** | 760 ns - 805 µs | Negligible for most use cases |
| **Sequential Iteration** | 520-770 µs per 1M ops | Excellent cache locality |
| **Random Access** | ~791 ns per op | Good for scattered access |
| **Bounds Checking** | 57× overhead | Disable in hot loops |
| **Memory Bandwidth** | 4.8-6.7 GB/s | Good DRAM utilization |
| **AoS Structs** | 8-12 bytes optimal | Keep structs small for cache efficiency |

#### 5. NumPy Interoperability Performance

The following Python benchmarks measure zero-copy overhead and interop costs:

**Test Environment:**
- Python: 3.9.21
- NumPy: 2.0.1
- pytest-benchmark: 5.1.0
- Measurement: 50+ rounds per test

| Operation | Array Size | Mean Time | Throughput | Notes |
|-----------|-----------|-----------|------------|-------|
| **Zero-Copy: NumPy → CNDA** |
| `from_numpy` (small) | 1 KB | 3.39 µs | 294,801 ops/s | Minimal validation overhead |
| `from_numpy` (large) | 100 MB | 14.0 ms | 71 ops/s | Validation scales with size |
| **Zero-Copy: CNDA → NumPy** |
| `to_numpy` (small) | 1 KB | 2.48 µs | 403,484 ops/s | Capsule creation |
| `to_numpy` (large) | 100 MB | 1.77 µs | 564,000 ops/s | Size-independent! |
| **Deep Copy Operations** |
| `from_numpy` (copy) | 1 MB | 150 µs | 6,667 ops/s | Actual memory copy |
| `from_numpy` (copy) | 100 MB | 22.8 ms | 44 ops/s | Memory bandwidth limited |
| `to_numpy` (copy) | 100 MB | 19.6 ms | 51 ops/s | Similar to from_numpy |
| **Round-Trip Conversion** |
| Zero-copy round-trip | 100×100 | 5.48 µs | 182,602 ops/s | NumPy→CNDA→NumPy |
| Deep-copy round-trip | 100×100 | 6.95 µs | 143,879 ops/s | 27% slower |
| **Array Creation** |
| NumPy `zeros()` | 1000×1000 | 27.3 µs | 36,600 ops/s | Baseline |
| CNDA constructor | 1000×1000 | 830 µs | 1,205 ops/s | 30× slower (use NumPy instead) |
| NumPy `copy()` | 1M elements | 813 µs | 1,230 ops/s | Memory bandwidth |
| **AoS Structured Types** |
| Vec2f: CNDA → NumPy | 1000×1000 | 1.77 µs | 566,250 ops/s | Zero-copy struct export |
| Vec2f: NumPy → CNDA | 1000×1000 | 1.70 ms | 587 ops/s | Struct layout validation |
| **Sequential Access** |
| Pure NumPy sum | 1M elements | 763 µs | 1,311 Kops/s | Baseline vectorized |
| CNDA→NumPy→sum | 1M elements | 935 µs | 1,069 Kops/s | 23% slower (extra view) |

**Critical Insights:**

- **`to_numpy` is size-independent**: ~1.8 µs overhead regardless of array size
- **`from_numpy` validation scales**: 3.4 µs (1 KB) → 14 ms (100 MB) due to layout checks
- **Deep copy is 100× slower**: 20-23 ms for 100 MB (bandwidth limited)
- **CNDA construction is slow**: 30× slower than NumPy—use NumPy→CNDA pipeline
- **AoS struct export is fast**: 1.77 µs zero-copy; import is 1.70 ms (validation)

**Recommendations:**
1. **Always use zero-copy** when possible (2.5-3.4 µs overhead for small arrays)
2. **`to_numpy` is extremely fast** - constant ~1.77 µs regardless of size
3. **`from_numpy` validation scales with size** - ~14 ms for 100 MB (still faster than copying)
4. **Avoid CNDA constructors** - use NumPy→CNDA pipeline instead (30× faster)
5. **Struct export is cheap** (~1.77 µs) - perfect for passing to visualization
6. **Deep copy only when necessary** - ~100× slower than zero-copy

### Performance Comparison Table

| Operation | ContiguousND | std::vector (flat) | NumPy | Winner |
|-----------|-------------|-------------------|-------|--------|
| 2D element access | 745 µs | 774 µs | N/A | CNDA (3.9% faster) |
| 3D element access | 1.42 ms | 1.04 ms (nested) | N/A | Comparable |
| Array creation | 830 µs | ~100 µs | 27 µs | NumPy (30× faster) |
| Zero-copy (to_numpy) | 1.7-1.8 µs | N/A | N/A | CNDA unique feature |
| Zero-copy (from_numpy small) | 2.5-3.4 µs | N/A | N/A | CNDA unique feature |
| Deep copy (100MB) | 20-23 ms | N/A | N/A | Bandwidth limited |
| Memory overhead | 32 bytes | 24 bytes | 96 bytes | std::vector (flat) |
| Type safety | Strong | Manual | Strong | Tie (CNDA/NumPy) |
| Zero-copy interop | Yes | No | Yes (with CNDA) | CNDA |
| Cache efficiency | Excellent | Excellent | Excellent | Tie |

---

## Troubleshooting

### pytest-benchmark not found

**Symptom:** 
```
Python benchmarks fail with "No module named 'pytest_benchmark'"
```

**Solutions:**
1. Install in active environment:
   ```bash
   pip install pytest-benchmark
   ```

2. Check pytest version (needs 3.0+):
   ```bash
   pytest --version
   ```

3. Verify import:
   ```bash
   python -c "import pytest_benchmark; print('OK')"
   ```

### Benchmark executables not found

**Symptom:** 
```
bench_core.exe: No such file or directory
```

**Solutions:**
1. Verify CMake built benchmarks:
   ```bash
   cmake .. -DCNDA_BUILD_BENCHMARKS=ON
   ```

2. Check correct directory:
   ```bash
   # Windows
   cd build\benchmarks\Release
   
   # Linux/Mac
   cd build/benchmarks
   ```

3. Rebuild in Release mode:
   ```bash
   cmake --build . --config Release
   ```

### High variability in results

**Symptom:** 
```
std dev > 20% of mean
```

**Solutions:**
1. Close background applications
2. Increase sample count: `--benchmark-samples 200`
3. Run benchmarks multiple times and average
4. Check CPU thermal throttling (reduce load, improve cooling)
5. Disable turbo boost for consistency

### Unexpectedly slow performance

**Symptom:** 
```
CNDA slower than expected
```

**Checklist:**
- Built in Release mode? (`cmake --build . --config Release`)
- Bounds checking disabled? (compile with `-DCNDA_BOUNDS_CHECK=OFF`)
- Compiler optimizations enabled? (`-O3` or `/O2`)
- Using correct executable? (Release, not Debug)
- System not under load? (check Task Manager / htop)

---

## Reporting Benchmark Results

When sharing benchmark results, include:

### System Information
```bash
# CPU
wmic cpu get name  # Windows
lscpu | grep "Model name"  # Linux

# RAM
wmic memorychip get capacity  # Windows
free -h  # Linux

# Compiler
cl /?  # MSVC
g++ --version  # GCC
clang++ --version  # Clang
```

### Build Configuration
- CMake version
- Compiler and version
- Build type (Release/Debug)
- CMake flags used

### Benchmark Configuration
- Number of samples
- Warmup time
- Array sizes tested
- Data types tested

---

## Additional Resources

- **[INSTALLATION.md](INSTALLATION.md)** - Installing CNDA
- **[QUICKSTART.md](QUICKSTART.md)** - Getting started guide
- **[CPP_USER_GUIDE.md](CPP_USER_GUIDE.md)** - C++ API reference
- **[PYTHON_USER_GUIDE.md](PYTHON_USER_GUIDE.md)** - Python API reference

---

**Version:** 0.1.0 | **Last Updated:** December 2024
