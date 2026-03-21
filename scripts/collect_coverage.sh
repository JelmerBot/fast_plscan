#!/usr/bin/env bash
# Run pytest and report combined Python + C++ coverage.
#
# Usage:
#   ./scripts/collect_coverage.sh [--rebuild] [--html-report]
#
# Options:
#   --rebuild      Reinstall the package with -DPLSCAN_COVERAGE=ON before running tests.
#   --html-report  Generate an HTML C++ coverage report in coverage_html/.
#
# The package must have been installed with coverage instrumentation before
# running this script (use --rebuild to do that automatically):
#
#   pip install --no-build-isolation -v . \
#       --config-settings cmake.args="-DPLSCAN_COVERAGE=ON;-DCMAKE_BUILD_TYPE=Debug"

set -euo pipefail

REBUILD=0
HTML_REPORT=0

for arg in "$@"; do
    case $arg in
        --rebuild)     REBUILD=1 ;;
        --html-report) HTML_REPORT=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# --- Prerequisites ---
python -c "import pytest_cov" 2>/dev/null || {
    echo "ERROR: pytest-cov not found. Install with: pip install pytest-cov" >&2
    exit 1
}
command -v gcovr &>/dev/null || {
    echo "ERROR: gcovr not found. Install with: pip install gcovr" >&2
    exit 1
}

# --- Optionally rebuild ---
if [[ $REBUILD -eq 1 ]]; then
    echo ""
    echo "=== Rebuilding with coverage instrumentation ==="
    pip install --no-build-isolation -v . \
        --config-settings cmake.args="-DPLSCAN_COVERAGE=ON;-DCMAKE_BUILD_TYPE=Debug"
fi

# --- Run pytest (Python + C++ instrumentation) ---
echo ""
echo "=== Running pytest ==="
SCIPY_ARRAY_API=1 pytest . -q --cov=fast_plscan --cov-report=term-missing
PYTEST_EXIT=$?

# --- C++ summary via gcovr ---
echo ""
echo "=== C++ Coverage Summary ==="
gcovr --root . --filter src/fast_plscan/_api --print-summary

# --- C++ uncovered lines ---
echo ""
echo "=== C++ Uncovered Lines ==="
gcovr --root . --filter src/fast_plscan/_api --txt - 2>/dev/null | \
    awk '
        /\.cpp$|\.h$/ { file = $0; printed = 0; next }
        /\*\*\*\*\*/ {
            if (!printed && file != "") { print file; printed = 1 }
            print
        }
    '

# --- Optional HTML report ---
if [[ $HTML_REPORT -eq 1 ]]; then
    echo ""
    echo "Generating C++ HTML report..."
    mkdir -p coverage_html
    gcovr --root . --filter src/fast_plscan/_api --html-details coverage_html/index.html
    echo "C++ HTML report: coverage_html/index.html"
fi

# --- Clean up coverage data files ---
echo ""
echo "Cleaning up coverage data files..."
find . -name "*.gcda" -delete 2>/dev/null || true
rm -f .coverage

exit $PYTEST_EXIT
