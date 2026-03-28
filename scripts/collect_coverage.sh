#!/usr/bin/env bash
# Run pytest and report combined Python + C++ coverage.
#
# Usage:
#   bash scripts/collect_coverage.sh

set -euo pipefail

# --- Creating coverage build ---
echo ""
echo "=== Building with coverage instrumentation ==="
uv pip install -ve . --config-settings cmake.args="-DPLSCAN_COVERAGE=ON;-DCMAKE_BUILD_TYPE=Debug"

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
