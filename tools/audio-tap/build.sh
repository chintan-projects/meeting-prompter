#!/bin/bash
# Build the audio-tap Swift CLI tool for macOS.
# Output: runners/macos-arm64/audio-tap
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/runners/macos-arm64"

cd "$SCRIPT_DIR"

echo "[audio-tap] Building..."
swift build -c release 2>&1 | grep -v "Build complete" || true

mkdir -p "$OUTPUT_DIR"
cp ".build/release/audio-tap" "$OUTPUT_DIR/audio-tap"

echo "[audio-tap] Built: $OUTPUT_DIR/audio-tap"
