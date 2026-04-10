#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDIO_DIR="$ROOT/foreblocks/studio"
STUDIO_DIST="$STUDIO_DIR/dist"
TARGET_DIR="$ROOT/web/studio"

if [[ ! -d "$STUDIO_DIR" ]]; then
  echo "❌ Studio directory not found: $STUDIO_DIR"
  exit 1
fi

if [[ ! -f "$STUDIO_DIR/package.json" ]]; then
  echo "❌ Studio package.json not found in $STUDIO_DIR"
  exit 1
fi

command -v npm >/dev/null 2>&1 || {
  echo "❌ npm is required to build the Studio frontend"
  exit 1
}

echo "🧱 Building foreBlocks Studio frontend..."
rm -rf "$STUDIO_DIST"
cd "$STUDIO_DIR"
if [[ -f package-lock.json ]]; then
  npm ci --no-audit --no-fund
else
  npm install --no-audit --no-fund
fi
npm run build -- --base /studio/

if [[ ! -f "$STUDIO_DIST/index.html" ]]; then
  echo "❌ Studio build failed: $STUDIO_DIST/index.html not found"
  exit 1
fi

echo "📦 Publishing Studio build to web/studio..."
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"
cp -a "$STUDIO_DIST/." "$TARGET_DIR/"

echo "✅ Studio assets compiled and copied to web/studio"
echo "Access the app at /studio/ on the remote server"
