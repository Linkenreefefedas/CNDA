# CNDA Installation Guide

Quick installation guide for CNDA on Windows, Linux, and macOS.

## Table of Contents

- [Requirements](#requirements)
- [Quick Install](#quick-install)
- [C++ Header-Only](#c-header-only)
- [Python Bindings](#python-bindings)
- [Build from Source](#build-from-source)
- [Verification](#verification)

---

## Requirements

| Component | Version |
|-----------|---------|
| **CMake** | 3.18+ |
| **C++ Compiler** | C++11 compatible |
| **Python** | 3.9+ (for Python bindings) |
| **pybind11** | 2.6.0+ (for Python bindings) |
| **NumPy** | Any recent version |

### Compilers

**Windows:** Visual Studio 2015+ (with C++ workload) or MinGW-w64  
**Linux:** GCC 4.8.1+ or Clang 3.4+  
**macOS:** Xcode Command Line Tools

**Quick setup:**
```bash
# Linux (Ubuntu/Debian)
sudo apt install build-essential cmake

# Linux (Fedora/RHEL)
sudo dnf install gcc-c++ cmake

# macOS
xcode-select --install
brew install cmake
```

---

## Quick Install

### Python Bindings (Recommended)

```bash
# Create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .\.venv\Scripts\Activate.ps1  # Windows

# Install
pip install -U pip
pip install .

# Verify
python -c "import cnda; print(cnda.__version__)"
```

---

## C++ Header-Only Library

The CNDA core is a **header-only** library, meaning you don't need to compile anything to use it in your C++ projects.

### Installation Steps

#### Option 1: Manual Header Copy

1. **Download or clone the repository:**
   ```bash
   git clone https://github.com/Linkenreefefedas/CNDA.git
   cd CNDA
   ```

2. **Copy headers to your project:**
   ```bash
   # Copy the entire include/cnda directory to your project
   cp -r include/cnda /path/to/your/project/include/
   ```

3. **Include in your C++ code:**
   ```cpp
   #include "cnda/contiguous_nd.hpp"
   
   int main() {
       cnda::ContiguousND<float> arr({3, 4});
       arr(1, 2) = 42.0f;
       return 0;
   }
   ```

4. **Compile your project:**
   ```bash
   # Add include path to your compiler
   g++ -std=c++11 -I/path/to/include your_code.cpp -o your_program
   ```

#### Option 2: CMake Integration

Add CNDA as a subdirectory in your CMakeLists.txt:

```cmake
# Your CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(YourProject)

# Add CNDA
add_subdirectory(external/CNDA)

# Your executable
add_executable(your_program main.cpp)
target_link_libraries(your_program PRIVATE cnda_headers)
```

Or use CMake's FetchContent:

```cmake
include(FetchContent)

FetchContent_Declare(
  cnda
  GIT_REPOSITORY https://github.com/Linkenreefefedas/CNDA.git
  GIT_TAG        v0.1.0
)
FetchContent_MakeAvailable(cnda)
---

## C++ Header-Only

CNDA core is **header-only** with zero external dependencies.

### Manual Installation

---

## Python Bindings

### Standard Installation

```bash
git clone https://github.com/Linkenreefefedas/CNDA.git
cd CNDA
pip install .
```

### Development Mode

```bash
pip install -e .  # Editable install

# Rebuild after C++ changes
pip install -e . --force-reinstall --no-deps
```

### Custom Build Options

```bash
# Enable bounds checking
CMAKE_ARGS="-DCNDA_BOUNDS_CHECK=ON" pip install .

# Debug build
DEBUG=1 pip install .

# Verbose output
pip install . -v
```ke --build . --config Release -j

# Run tests
ctest -C Release --output-on-failure
```

### Python Bindings Only

```bash
# Install pybind11 first
pip install "pybind11[global]"

# Configure with Python bindings
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release

# Add to Python path for testing
# Windows:
$env:PYTHONPATH="$env:PYTHONPATH;$(Get-Location)\python"
# Linux/macOS:
export PYTHONPATH="$PYTHONPATH:$(pwd)/python"

# Test
python -c "import cnda; print(cnda.__version__)"
```

### With Benchmarks

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
cmake --build . --config Release -j

# Run benchmarks
cd benchmarks
.\run_all_benchmarks.ps1  # Windows
# or
./run_all_benchmarks.sh   # Linux/macOS
```

---

## Verification

After installation, verify that CNDA is working correctly:

### Python Bindings

```python
import cnda
import numpy as np

# Check version
print(f"CNDA version: {cnda.__version__}")

# Test basic functionality
arr = cnda.ContiguousND_f32([3, 4])
---

## Build from Source

### C++ Tests

```bash
git clone https://github.com/Linkenreefefedas/CNDA.git
cd CNDA
mkdir build && cd build

# Linux/macOS
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
ctest --output-on-failure

# Windows
cmake .. -A x64
cmake --build . --config Release -j
ctest -C Release --output-on-failure
```

### With Benchmarks

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
cmake --build . --config Release -j

# Run
cd benchmarks
./run_all_benchmarks.sh  # Linux/macOS
.\run_all_benchmarks.ps1 # Windows
```ke: command not found
```

**Solution:**

- **Windows**: Install CMake from https://cmake.org/download/ or install Visual Studio
- **Linux**: `sudo apt install cmake` or `sudo dnf install cmake`
- **macOS**: `brew install cmake`

---

## Verification

### Python Test

```python
import cnda
import numpy as np

# Version check
print(f"CNDA: {cnda.__version__}")

# Basic test
arr = cnda.ContiguousND_f32([3, 4])
arr[1, 2] = 42.0
print(f"Shape: {arr.shape}, Value: {arr[1, 2]}")

# Zero-copy test
x = np.arange(12, dtype=np.float32).reshape(3, 4)
cnda_arr = cnda.from_numpy(x, copy=False)
y = cnda_arr.to_numpy(copy=False)
print(f"Zero-copy: {y.ctypes.data == x.ctypes.data}")
```

### C++ Test

```cpp
#include "cnda/contiguous_nd.hpp"
#include <iostream>

int main() {
    cnda::ContiguousND<float> arr({3, 4});
    arr(1, 2) = 42.0f;
    std::cout << "Value: " << arr(1, 2) << "\n";
    return 0;
}
```

```bash
# Compile
g++ -std=c++11 -I./include test.cpp -o test

# Run
./test
```

### Run Tests

```bash
# Python
pip install pytest
pytest tests/python/ -v

# C++
cd build
ctest --output-on-failure
```---

## Next Steps

- [QUICKSTART.md](QUICKSTART.md) - 5-minute tutorial
- [PYTHON_USER_GUIDE.md](PYTHON_USER_GUIDE.md) - Python API
- [CPP_USER_GUIDE.md](CPP_USER_GUIDE.md) - C++ API

---

**Version:** 0.1.0 | **Release:** Dec 2024 | [GitHub](https://github.com/Linkenreefefedas/CNDA)