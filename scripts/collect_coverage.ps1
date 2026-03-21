#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run pytest and report combined Python + C++ coverage.

.DESCRIPTION
    Runs the test suite with Python coverage (pytest-cov) and LLVM source-based
    C++ coverage instrumentation, then reports uncovered lines for both layers.

    The package must have been installed with coverage instrumentation before
    running this script (use -Rebuild to do that automatically):

        pip install --no-build-isolation -v . `
            --config-settings cmake.args="-DPLSCAN_COVERAGE=ON;-DCMAKE_BUILD_TYPE=Debug;-DCMAKE_CXX_COMPILER=clang-cl"

.PARAMETER Rebuild
    Reinstall the package with -DPLSCAN_COVERAGE=ON before running tests.

.PARAMETER HtmlReport
    Generate an HTML C++ coverage report in coverage_html/.
#>
param(
    [switch]$Rebuild,
    [switch]$HtmlReport
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# --- Prerequisites ---
python -c "import pytest_cov" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "pytest-cov not found. Install with: pip install pytest-cov"
    exit 1
}
if (-not (Get-Command llvm-profdata -ErrorAction SilentlyContinue)) {
    Write-Error "llvm-profdata not found. Ensure LLVM is installed and on PATH."
    exit 1
}

# --- Optionally rebuild ---
if ($Rebuild) {
    Write-Host "`nRebuilding with coverage instrumentation..." -ForegroundColor Cyan
    pip install --no-build-isolation -v . `
        --config-settings cmake.args="-DPLSCAN_COVERAGE=ON;-DCMAKE_BUILD_TYPE=Debug;-DCMAKE_CXX_COMPILER=clang-cl"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# --- Run pytest (Python + C++ instrumentation) ---
Write-Host "`n=== Running pytest ===" -ForegroundColor Cyan
$env:LLVM_PROFILE_FILE = "$PWD\coverage.profraw"
pytest . -q --cov=fast_plscan --cov-report=term-missing
$pytestExit = $LASTEXITCODE
Remove-Item Env:\LLVM_PROFILE_FILE

# --- Merge C++ profile ---
if (-not (Test-Path coverage.profraw)) {
    Write-Warning "No coverage.profraw was written. Ensure the package was built with -DPLSCAN_COVERAGE=ON."
    exit 1
}
Write-Host "`nMerging C++ profile data..." -ForegroundColor DarkGray
llvm-profdata merge -sparse coverage.profraw -o coverage.profdata
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$pyd = python -c "import fast_plscan._api as m; print(m.__file__)"
$sources = "src/fast_plscan/_api"

# --- C++ summary ---
Write-Host "`n=== C++ Coverage Summary ===" -ForegroundColor Cyan
llvm-cov report $pyd -instr-profile coverage.profdata -sources $sources

# --- C++ uncovered lines ---
Write-Host "`n=== C++ Uncovered Lines ===" -ForegroundColor Cyan
$currentFile = $null
llvm-cov show $pyd -instr-profile coverage.profdata -sources $sources -format=text |
    ForEach-Object {
        # File header lines end with ".cpp:" or ".h:"
        if ($_ -match '\.(?:cpp|h):$') {
            $currentFile = $_
        } elseif ($_ -match '^\s+\d+\|\s+0\|') {
            if ($null -ne $currentFile) {
                Write-Host $currentFile -ForegroundColor Yellow
                $currentFile = $null
            }
            Write-Host $_
        }
    }

# --- Optional HTML report ---
if ($HtmlReport) {
    Write-Host "`nGenerating C++ HTML report..." -ForegroundColor DarkGray
    New-Item -ItemType Directory -Force coverage_html | Out-Null
    llvm-cov show $pyd -instr-profile coverage.profdata -sources $sources `
        -format=html -output-dir=coverage_html
    Write-Host "C++ HTML report: coverage_html\index.html" -ForegroundColor Green
}

# --- Clean up coverage data files ---
Write-Host "`nCleaning up coverage data files..." -ForegroundColor DarkGray
Remove-Item -ErrorAction SilentlyContinue default.profraw, coverage.profraw, coverage.profdata, .coverage

exit $pytestExit
