#!/usr/bin/env bash
# Fetch the upstream JAKA Zu3 MuJoCo model that assets/jaka_zu3/*.xml reference.
#
# Why it is not vendored here: the mesh files come from a JAKA source copy with
# no clear redistribution licence (upstream package.xml says TODO), and the
# checkout is ~100 MB (75 MB of it are two Objaverse props this project never
# uses). Everything we wrote is licence-clean; the meshes are not ours to ship.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/third_party/jaka-zu3-mujoco"
COMMIT=7c40fec
URL=https://github.com/yy2006-maker/jaka-zu3-mujoco

if [ -d "$DEST/.git" ]; then
  echo "already cloned: $DEST"
else
  mkdir -p "$(dirname "$DEST")"
  git clone "$URL" "$DEST"
fi

git -C "$DEST" checkout --quiet "$COMMIT"
test -f "$DEST/jaka_zu3_mujoco.xml" || { echo "missing jaka_zu3_mujoco.xml in $DEST" >&2; exit 1; }
echo "third_party ready at $(git -C "$DEST" rev-parse --short HEAD)"
