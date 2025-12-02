# PowerShell script to run all CNDA benchmarks
# Usage: .\run_all_benchmarks.ps1

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "CNDA Performance Benchmarks" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$BUILD_DIR = Split-Path -Parent $PSScriptRoot | Join-Path -ChildPath "build"
$BENCHMARKS_DIR = Join-Path $BUILD_DIR "benchmarks\Release"

# C++ Benchmarks
Write-Host "Running C++ Core Benchmarks..." -ForegroundColor Yellow
& "$BENCHMARKS_DIR\bench_core.exe" --benchmark-samples 100
Write-Host ""

Write-Host "Running C++ AoS Benchmarks..." -ForegroundColor Yellow
& "$BENCHMARKS_DIR\bench_aos.exe" --benchmark-samples 100
Write-Host ""

Write-Host "Running C++ Comparison Benchmarks..." -ForegroundColor Yellow
& "$BENCHMARKS_DIR\bench_comparison.exe" --benchmark-samples 50
Write-Host ""

# Python Benchmarks
Write-Host "Running Python NumPy Interop Benchmarks..." -ForegroundColor Yellow
$env:PYTHONPATH = Join-Path $BUILD_DIR "python\Release"
python -m pytest "$PSScriptRoot\bench_numpy_interop.py" `
    -v --benchmark-only --benchmark-group-by=group
Write-Host ""

Write-Host "======================================" -ForegroundColor Green
Write-Host "Benchmarks Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
