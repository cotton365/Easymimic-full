#!/usr/bin/env bash
# Fetch the upstream JAKA Zu3 MuJoCo model that assets/jaka_zu3/*.xml reference.
#
# Why it is not vendored here: the mesh files come from a JAKA source copy with
# no clear redistribution licence (no LICENSE file; upstream README.md:43 says TODO), and the
# checkout is ~150 MB (75 MB of it are two Objaverse props this project never
# uses, ~35 MB are gripper STLs we do not load yet). Everything we wrote is
# licence-clean; the meshes are not ours to ship.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/third_party/jaka-zu3-mujoco"
COMMIT=c592951
URL=https://github.com/yy2006-maker/jaka-zu3-mujoco

if [ -d "$DEST/.git" ]; then
  echo "already cloned: $DEST"
else
  mkdir -p "$(dirname "$DEST")"
  git clone "$URL" "$DEST"
fi

# An existing clone made before $COMMIT existed cannot check it out until it
# fetches: try, then fetch once and retry.
if ! git -C "$DEST" checkout --quiet "$COMMIT" 2>/dev/null; then
  echo "commit $COMMIT missing locally, fetching origin..."
  git -C "$DEST" fetch --quiet origin
  git -C "$DEST" checkout --quiet "$COMMIT"
fi

# The assets were authored and verified against $COMMIT only: fail loudly instead of
# quietly building a different model.
head_sha="$(git -C "$DEST" rev-parse HEAD)"
case "$head_sha" in
  "$COMMIT"*) ;;
  *) echo "unexpected HEAD $head_sha (want $COMMIT...)" >&2; exit 1 ;;
esac

test -f "$DEST/jaka_zu3_mujoco.xml" \
  || { echo "missing $DEST/jaka_zu3_mujoco.xml" >&2; exit 1; }
test -f "$DEST/assets/jaka_zu3_meshes/Link_1.STL" \
  || { echo "missing meshes under $DEST/assets/jaka_zu3_meshes (incomplete clone?)" >&2; exit 1; }

echo "third_party ready at $(git -C "$DEST" rev-parse --short HEAD)"
