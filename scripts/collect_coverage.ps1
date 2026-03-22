#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run pytest and report combined Python + C++ coverage.

.DESCRIPTION
    Runs the test suite with Python coverage (pytest-cov) and LLVM source-based
    C++ coverage instrumentation, then reports uncovered lines for both layers.

    The package must have been installed with coverage instrumentation before
    running this script (use -Rebuild to do that automatically):

        uv pip install -ve . `
            --config-settings cmake.args="-DPLSCAN_COVERAGE=ON;-DCMAKE_BUILD_TYPE=Debug;-T ClangCL"

.PARAMETER Rebuild
    Reinstall the package with -DPLSCAN_COVERAGE=ON using uv pip install.

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
if (-not (Get-Command llvm-profdata -ErrorAction SilentlyContinue)) {
    Write-Error "llvm-profdata not found. Ensure LLVM is installed and on PATH."
    exit 1
}

# --- Optionally rebuild ---
if ($Rebuild) {
    Write-Host "`nRebuilding with coverage instrumentation..." -ForegroundColor Cyan
    uv pip install -ve . --config-settings cmake.args="-DPLSCAN_COVERAGE=ON;-DCMAKE_BUILD_TYPE=Debug;-T ClangCL"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# --- Run pytest (Python + C++ instrumentation) ---
Write-Host "`n=== Running pytest ===" -ForegroundColor Cyan
# %m is substituted with a per-binary hash so the .pyd DLL writes its own profraw file.
$env:LLVM_PROFILE_FILE = "$PWD\coverage-%m.profraw"
$ErrorActionPreference = "Continue"
pytest . --cov=fast_plscan --cov-report=term-missing
$pytestExit = $LASTEXITCODE
$ErrorActionPreference = "Stop"
Remove-Item Env:\LLVM_PROFILE_FILE

# --- Merge C++ profile ---
$profrawFiles = Get-Item coverage-*.profraw -ErrorAction SilentlyContinue
if (-not $profrawFiles) {
    Write-Warning "No coverage-*.profraw files were written. Ensure the package was built with -DPLSCAN_COVERAGE=ON."
    exit 1
}
Write-Host "`nMerging C++ profile data..." -ForegroundColor DarkGray
llvm-profdata merge -sparse $profrawFiles -o coverage.profdata
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

# --- Combined coverage summary ---
Write-Host "`n=== Combined Coverage ===" -ForegroundColor Cyan

# Python: parse TOTAL line from coverage report
$pyReport = python -m coverage report 2>$null
$pyTotal  = $pyReport | Where-Object { $_ -match '^\s*TOTAL\s' }
if ($pyTotal -match '(\d+)\s+(\d+)\s+(\d+)%') {
    $pyStmts = [int]$Matches[1]
    $pyMiss  = [int]$Matches[2]
    $pyHit   = $pyStmts - $pyMiss
    $pyPct   = [int]$Matches[3]
} else {
    $pyStmts = 0; $pyHit = 0; $pyPct = 0
}

# C++ via llvm-cov export
$cppJson  = llvm-cov export $pyd -instr-profile coverage.profdata -sources $sources `
    -format=text -summary-only | ConvertFrom-Json
$cppTotal = $cppJson.data[0].totals.lines.count
$cppHit   = $cppJson.data[0].totals.lines.covered
$cppPct   = [math]::Round(100.0 * $cppHit / $cppTotal, 1)

$combinedTotal = $pyStmts + $cppTotal
$combinedHit   = $pyHit + $cppHit
$combinedPct   = [math]::Round(100.0 * $combinedHit / $combinedTotal, 1)
Write-Host ("Python {0}% ({1}/{2} stmts)  |  C++ {3}% ({4}/{5} lines)  |  Combined {6}%" -f `
    $pyPct, $pyHit, $pyStmts, $cppPct, $cppHit, $cppTotal, $combinedPct) -ForegroundColor Green

# --- Clean up coverage data files ---
Write-Host "`nCleaning up coverage data files..." -ForegroundColor DarkGray
Remove-Item -ErrorAction SilentlyContinue *.profraw, *.profdata, .coverage

exit $pytestExit
