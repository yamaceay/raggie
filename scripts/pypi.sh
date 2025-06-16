#!/bin/bash
# This script deploys the package to the actual PyPI.
# Usage: ./scripts/pypi.sh <version>

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <version>"
  exit 1
fi
VERSION=$1

# Ensure required tools are installed
if ! uv pip list | grep -q "^build " && ! uv pip list | grep -q "^twine "; then
    uv pip install build twine
fi

# Clean previous builds
rm -rf dist

# Build the package
uv run python -m build

# Install the built package for verification
uv pip install dist/raggie-${VERSION}-py3-none-any.whl

# Upload to PyPI
uv run python -m twine upload --repository pypi dist/*
