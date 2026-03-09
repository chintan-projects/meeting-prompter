#!/bin/bash
# Build the Tauri app and install to /Applications.
#
# Uses symlinks to reference the existing dev source tree, venv, and context
# docs — no freezing or bundling. For personal use on a single machine.
#
# Usage:
#   bash scripts/package.sh          # Build + install
#   bash scripts/package.sh --skip-build  # Symlink only (re-run after code changes)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="Meeting Prompter"
APP_BUNDLE="$PROJECT_ROOT/app/src-tauri/target/release/bundle/macos/${APP_NAME}.app"
INSTALL_PATH="/Applications/${APP_NAME}.app"

echo "=== Meeting Prompter Packager ==="
echo "Project root: $PROJECT_ROOT"

# 1. Build Tauri release (unless --skip-build)
if [ "${1:-}" != "--skip-build" ]; then
    echo ""
    echo "Building Tauri release..."
    cd "$PROJECT_ROOT/app"
    npm run tauri build
    cd "$PROJECT_ROOT"
else
    echo ""
    echo "Skipping build (--skip-build)"
fi

# 2. Verify .app bundle exists
if [ ! -d "$APP_BUNDLE" ]; then
    echo "Error: .app bundle not found at $APP_BUNDLE"
    echo "Run without --skip-build first."
    exit 1
fi

RESOURCES="$APP_BUNDLE/Contents/Resources"

# 3. Create symlinks into Resources
echo ""
echo "Creating symlinks..."

# Source code
for dir in lib src scripts context; do
    target="$PROJECT_ROOT/$dir"
    link="$RESOURCES/$dir"
    if [ -e "$link" ]; then
        rm -f "$link"
    fi
    ln -sf "$target" "$link"
    echo "  $dir → $target"
done

# Virtual environment
if [ -e "$RESOURCES/venv" ]; then
    rm -f "$RESOURCES/venv"
fi
ln -sf "$PROJECT_ROOT/venv" "$RESOURCES/venv"
echo "  venv → $PROJECT_ROOT/venv"

# Config
if [ -e "$RESOURCES/config.yaml" ]; then
    rm -f "$RESOURCES/config.yaml"
fi
ln -sf "$PROJECT_ROOT/config.yaml" "$RESOURCES/config.yaml"
echo "  config.yaml → $PROJECT_ROOT/config.yaml"

# 4. Write MEETING_PROMPTER_ROOT into Info.plist environment
# The wrapper script handles this, but we also embed it for discoverability
PLIST="$APP_BUNDLE/Contents/Info.plist"
if [ -f "$PLIST" ]; then
    # Add LSEnvironment with MEETING_PROMPTER_ROOT if not already present
    if ! /usr/libexec/PlistBuddy -c "Print :LSEnvironment" "$PLIST" >/dev/null 2>&1; then
        /usr/libexec/PlistBuddy -c "Add :LSEnvironment dict" "$PLIST"
    fi
    /usr/libexec/PlistBuddy -c "Set :LSEnvironment:MEETING_PROMPTER_ROOT $PROJECT_ROOT" "$PLIST" 2>/dev/null || \
        /usr/libexec/PlistBuddy -c "Add :LSEnvironment:MEETING_PROMPTER_ROOT string $PROJECT_ROOT" "$PLIST"
    echo "  Info.plist: MEETING_PROMPTER_ROOT=$PROJECT_ROOT"
fi

# 5. Copy to /Applications
echo ""
echo "Installing to /Applications..."
if [ -d "$INSTALL_PATH" ]; then
    rm -rf "$INSTALL_PATH"
fi
cp -R "$APP_BUNDLE" "$INSTALL_PATH"

echo ""
echo "Installed: $INSTALL_PATH"
echo "Launch from Spotlight or: open '$INSTALL_PATH'"
