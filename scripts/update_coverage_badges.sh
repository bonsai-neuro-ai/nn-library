#!/usr/bin/env bash
# Run the test suite under coverage and check docstring coverage, regenerating the
# badge SVGs in badges/ that are embedded in README.md.
#
# Runs against a fresh clone of the repo's committed HEAD in a temp directory, so
# results aren't sensitive to uncommitted local edits. Only the resulting badge SVGs
# are copied back into this checkout.
#
# Run this manually (e.g. on the lab server, where CUDA is available) after committing
# changes you want reflected in the README badges, then commit the updated SVGs.
#
# Usage: DATA_ROOT=/data/datasets scripts/update_coverage_badges.sh
set -uo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

echo "==> Cloning from latest GitHub main branch into $tmpdir..."
git clone https://github.com/bonsai-neuro-ai/nn-library "$tmpdir" || exit 1
cd "$tmpdir" || exit 1
git fetch origin coverage && git checkout origin/coverage || exit 1

mkdir -p badges
status=0

echo "==> Running test suite under coverage..."
PYTHONPATH=src uv run --extra dev coverage run -m unittest discover tests
test_status=$?
[ "$test_status" -ne 0 ] && status=$test_status

# Generate the coverage report/badge from whatever coverage data was collected,
# even if some tests failed above, so the badge always reflects the latest run.
uv run --extra dev coverage report
uv run --extra dev coverage-badge -o badges/coverage.svg -f

echo "==> Checking docstring coverage..."
uv run --extra dev interrogate -v src/nn_lib --generate-badge badges --badge-style flat
interrogate_status=$?
[ "$interrogate_status" -ne 0 ] && status=$interrogate_status

mkdir -p "$repo_root/badges"
cp badges/coverage.svg badges/interrogate_badge.svg "$repo_root/badges/"

echo "==> Done. Updated badges/coverage.svg and badges/interrogate_badge.svg"
exit "$status"
