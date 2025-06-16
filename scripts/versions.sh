#!/bin/bash

# Function to increment semantic version
increment_semver() {
    local version=$1
    local mode=$2

    IFS='.' read -r MAJOR MINOR PATCH <<< "$version"

    case "$mode" in
        patch)
            PATCH=$((PATCH + 1))
            ;;
        minor)
            MINOR=$((MINOR + 1))
            PATCH=0
            ;;
        major)
            MAJOR=$((MAJOR + 1))
            MINOR=0
            PATCH=0
            ;;
        *)
            echo "Invalid mode '$mode'. Use 'patch', 'minor', or 'major'."
            exit 1
            ;;
    esac

    echo "${MAJOR}.${MINOR}.${PATCH}"
}

# Update version in pyproject.toml
update_pyproject_version() {
    local file=$1
    local mode=$2

    local current_version=$(grep -E '^version *= *"' "$file" | head -1 | sed -E 's/version *= *"([^"]+)"/\1/')
    local new_version=$(increment_semver "$current_version" "$mode")

    sed -i.bak -E "s/(^version *= *\").*(\")/\1${new_version}\2/" "$file"
    echo "📦 pyproject.toml version updated: $current_version → $new_version"
}

# Update version in setup.py
update_setup_py_version() {
    local file=$1
    local mode=$2

    local current_version=$(grep -E 'version *= *"' "$file" | head -1 | sed -E 's/.*version *= *"([^"]+)".*/\1/')
    local new_version=$(increment_semver "$current_version" "$mode")

    sed -i.bak -E "s/(version *= *\").*(\")/\1${new_version}\2/" "$file"
    echo "🐍 setup.py version updated: $current_version → $new_version"
}

# Main
MODE="$1"
TARGET="$2"

if [[ -z "$MODE" || -z "$TARGET" ]]; then
    echo "Usage: ./scripts/semver.sh patch|minor|major pyproject.toml|setup.py"
    exit 1
fi

if [[ "$TARGET" == "pyproject.toml" ]]; then
    update_pyproject_version "$TARGET" "$MODE"
elif [[ "$TARGET" == "setup.py" ]]; then
    update_setup_py_version "$TARGET" "$MODE"
else
    echo "Unsupported file: $TARGET"
    exit 1
fi
