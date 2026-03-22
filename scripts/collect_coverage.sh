#!/usr/bin/env bash
# Run pytest and report combined Python + C++ coverage.
#
# Usage:
#   bash scripts/collect_coverage.sh [--rebuild] [--html-report]
#
# Options:
#   --rebuild      Reinstall the package with -DPLSCAN_COVERAGE=ON using uv sync.
#   --html-report  Generate an HTML C++ coverage report in coverage_html/.
#
# The package must have been installed with coverage instrumentation before
# running this script (use --rebuild to do that automatically):
#
#   uv pip install -ve . \
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

# --- Optionally rebuild ---
if [[ $REBUILD -eq 1 ]]; then
    echo ""
    echo "=== Rebuilding with coverage instrumentation ==="
    uv pip install -ve . --config-settings cmake.args="-DPLSCAN_COVERAGE=ON;-DCMAKE_BUILD_TYPE=Debug"
fi

# --- Run pytest (Python + C++ instrumentation) ---
echo ""
echo "=== Running pytest ==="
PYTEST_EXIT=0
pytest . --cov=fast_plscan --cov-report=term-missing || PYTEST_EXIT=$?

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

# --- Combined coverage summary ---
echo ""
echo "=== Combined Coverage ==="

# Python: parse TOTAL line from coverage report
PY_REPORT=$(python -m coverage report 2>/dev/null || true)
PY_TOTAL_LINE=$(echo "$PY_REPORT" | grep -E '^\s*TOTAL\s' || true)
if [[ -n "$PY_TOTAL_LINE" ]]; then
    read -r _ PY_STMTS PY_MISS PY_PCT_STR <<< "$PY_TOTAL_LINE"
    PY_PCT=${PY_PCT_STR//%/}
    PY_HIT=$((PY_STMTS - PY_MISS))
else
    PY_STMTS=0; PY_HIT=0; PY_PCT=0
fi

# C++: parse from gcovr JSON summary
CPP_JSON=$(gcovr --root . --filter src/fast_plscan/_api --json-summary 2>/dev/null)
CPP_TOTAL=$(echo "$CPP_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['line_total'])")
CPP_HIT=$(echo "$CPP_JSON"   | python -c "import sys,json; d=json.load(sys.stdin); print(d['line_covered'])")
CPP_PCT=$(echo "$CPP_JSON"   | python -c "import sys,json; d=json.load(sys.stdin); print(str(round(d['line_percent'],1))+'%')")

COMB_TOTAL=$((PY_STMTS + CPP_TOTAL))
COMB_HIT=$((PY_HIT + CPP_HIT))
COMB_PCT=$(python -c "print(str(round(100.0 * $COMB_HIT / $COMB_TOTAL, 1))+'%')")

echo "Python ${PY_PCT}% (${PY_HIT}/${PY_STMTS} stmts)  |  C++ ${CPP_PCT} (${CPP_HIT}/${CPP_TOTAL} lines)  |  Combined ${COMB_PCT}"

# --- Clean up coverage data files ---
echo ""
echo "Cleaning up coverage data files..."
find . -name "*.gcda" -delete 2>/dev/null || true
rm -f .coverage

exit $PYTEST_EXIT
