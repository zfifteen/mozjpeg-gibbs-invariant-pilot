#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
WORKTREE_DIR="$ROOT_DIR/.worktrees/mozjpeg_baseline"
BIN_DIR="$ROOT_DIR/benchmarks/gibbs_v1/bin"
BASE_REF="${BASE_REF:-origin/master}"
JOBS="${JOBS:-8}"

mkdir -p "$BIN_DIR" "$ROOT_DIR/.worktrees"

cd "$ROOT_DIR"
git fetch origin

if [ ! -e "$WORKTREE_DIR/.git" ]; then
  git worktree add --detach "$WORKTREE_DIR" "$BASE_REF"
else
  (
    cd "$WORKTREE_DIR"
    git fetch origin
    git checkout --detach "$BASE_REF"
  )
fi

cmake -S "$WORKTREE_DIR" -B "$WORKTREE_DIR/build-release" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build "$WORKTREE_DIR/build-release" -j"$JOBS"
cp "$WORKTREE_DIR/build-release/cjpeg" "$BIN_DIR/cjpeg_A_baseline"

cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build-patched" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT_DIR/build-patched" -j"$JOBS"
cp "$ROOT_DIR/build-patched/cjpeg" "$BIN_DIR/cjpeg_B1_logonly"

shasum -a 256 "$BIN_DIR/cjpeg_A_baseline" "$BIN_DIR/cjpeg_B1_logonly" > "$BIN_DIR/SHA256SUMS.txt"

echo "Frozen binaries:"
ls -lh "$BIN_DIR/cjpeg_A_baseline" "$BIN_DIR/cjpeg_B1_logonly" "$BIN_DIR/SHA256SUMS.txt"
