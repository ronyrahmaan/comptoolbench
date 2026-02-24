#!/usr/bin/env bash
# Bundle CompToolBench paper for ArXiv submission.
# Creates arxiv_submission.tar.gz ready to upload.
#
# Usage:
#   bash scripts/bundle_arxiv.sh
#
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE_DIR="$(mktemp -d)/comptoolbench_arxiv"
OUTPUT="$PROJ_ROOT/arxiv_submission.tar.gz"

echo "Bundling ArXiv submission..."

mkdir -p "$BUNDLE_DIR/figures"
mkdir -p "$BUNDLE_DIR/tables"

# Core LaTeX files
cp "$PROJ_ROOT/paper/main.tex" "$BUNDLE_DIR/"
cp "$PROJ_ROOT/paper/references.bib" "$BUNDLE_DIR/"
cp "$PROJ_ROOT/paper/neurips_2025.sty" "$BUNDLE_DIR/"

# Figures (PDF only — ArXiv prefers vector)
cp "$PROJ_ROOT/paper/figures/"*.pdf "$BUNDLE_DIR/figures/"

# Tables
cp "$PROJ_ROOT/paper/tables/leaderboard.tex" "$BUNDLE_DIR/tables/"

# Create the tarball
cd "$(dirname "$BUNDLE_DIR")"
tar -czf "$OUTPUT" "$(basename "$BUNDLE_DIR")"

echo ""
echo "Done! ArXiv submission bundle:"
echo "  $OUTPUT"
echo ""
echo "Contents:"
tar -tzf "$OUTPUT"
echo ""
SIZE=$(du -h "$OUTPUT" | cut -f1)
echo "Size: $SIZE (ArXiv limit: 50MB)"
