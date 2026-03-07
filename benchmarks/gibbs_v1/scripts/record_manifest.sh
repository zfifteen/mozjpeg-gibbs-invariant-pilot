#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
BASE_DIR="$ROOT_DIR/.worktrees/mozjpeg_baseline"
BIN_DIR="$ROOT_DIR/benchmarks/gibbs_v1/bin"
MANIFEST_PATH="$ROOT_DIR/benchmarks/gibbs_v1/manifests/build_manifest.json"

mkdir -p "$(dirname "$MANIFEST_PATH")"

PATCHED_COMMIT="$(cd "$ROOT_DIR" && git rev-parse HEAD)"
PATCHED_BRANCH="$(cd "$ROOT_DIR" && git rev-parse --abbrev-ref HEAD)"
PATCHED_DIRTY=false
if ! (cd "$ROOT_DIR" && git diff --quiet --ignore-submodules HEAD --); then
  PATCHED_DIRTY=true
elif [ -n "$(cd "$ROOT_DIR" && git ls-files --others --exclude-standard | head -n 1)" ]; then
  PATCHED_DIRTY=true
fi
BASELINE_COMMIT=""
BASELINE_DIRTY=false
if [ -e "$BASE_DIR/.git" ]; then
  BASELINE_COMMIT="$(cd "$BASE_DIR" && git rev-parse HEAD)"
fi

CMAKE_VER="$(cmake --version | head -n 1 | sed 's/"/\\"/g')"
CC_VER="$(cc --version | head -n 1 | sed 's/"/\\"/g')"
UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"
UNAME_R="$(uname -r)"

A_SHA=""
B_SHA=""
if [ -f "$BIN_DIR/cjpeg_A_baseline" ]; then
  A_SHA="$(shasum -a 256 "$BIN_DIR/cjpeg_A_baseline" | awk '{print $1}')"
fi
if [ -f "$BIN_DIR/cjpeg_B1_logonly" ]; then
  B_SHA="$(shasum -a 256 "$BIN_DIR/cjpeg_B1_logonly" | awk '{print $1}')"
fi

cat > "$MANIFEST_PATH" <<JSON
{
  "schema": "mozjpeg-gibbs-benchmark-manifest-v1",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "project_root": "$ROOT_DIR",
  "baseline_ref": "origin/master",
  "baseline_commit": "$BASELINE_COMMIT",
  "baseline_dirty": $BASELINE_DIRTY,
  "patched_branch": "$PATCHED_BRANCH",
  "patched_commit": "$PATCHED_COMMIT",
  "patched_dirty": $PATCHED_DIRTY,
  "system": {
    "os": "$UNAME_S",
    "kernel": "$UNAME_R",
    "arch": "$UNAME_M"
  },
  "toolchain": {
    "cmake": "$CMAKE_VER",
    "cc": "$CC_VER"
  },
  "binaries": {
    "A": {
      "path": "benchmarks/gibbs_v1/bin/cjpeg_A_baseline",
      "sha256": "$A_SHA"
    },
    "B1": {
      "path": "benchmarks/gibbs_v1/bin/cjpeg_B1_logonly",
      "sha256": "$B_SHA"
    }
  }
}
JSON

echo "Wrote $MANIFEST_PATH"
