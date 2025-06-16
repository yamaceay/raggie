# auth token 
# Usage: ./scripts/testpypi.sh <version>
#!/bin/bash
set -e
if [ -z "$1" ]; then
  echo "Usage: $0 <version>"
  exit 1
fi
VERSION=$1

if ! uv pip list | grep -q "^build " && ! uv pip list | grep -q "^twine "; then
    uv pip install build twine
fi

rm -rf dist
uv run python -m build
uv pip install dist/raggie-${VERSION}-py3-none-any.whl
uv run python -m twine upload --repository testpypi dist/*