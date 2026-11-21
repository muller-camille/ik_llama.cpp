#!/bin/bash
# Replaces examples/server/public_llamacpp with a build of the web UI of llama.cpp ("Llama Chat"),
# the one llama-server serves with --webui llamacpp.
#
#   scripts/update-webui-llamacpp.sh            the current master of ggml-org/llama.cpp
#   scripts/update-webui-llamacpp.sh <commit>   that commit, tag or branch
#
# Needs git, Node 20 or newer, and the network (GitHub and the npm registry). The sources of the UI
# are not kept in this repository: they are fetched, built and thrown away. Afterwards, build
# llama-server and try the UI before committing: it talks to this server, not to the one it was
# written for.
set -euo pipefail

REF=${1:-master}
REPO=${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp}
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
DEST=$ROOT/examples/server/public_llamacpp
[ -f "$DEST/loading.html" ] || { echo "not found: $DEST/loading.html"; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "== fetching $REF of $REPO"
git init -q "$WORK/src"
git -C "$WORK/src" fetch -q --depth 1 "$REPO" "$REF"
git -C "$WORK/src" checkout -q FETCH_HEAD
COMMIT=$(git -C "$WORK/src" rev-parse HEAD)
DATE=$(git -C "$WORK/src" log -1 --format=%ad --date=short)
[ -d "$WORK/src/ui/apps/chat" ] || { echo "no ui/apps/chat at $COMMIT: is this commit older than Llama Chat?"; exit 1; }

echo "== building the UI at $COMMIT ($DATE)"
cd "$WORK/src/ui/apps/chat"
# the lock file is written by npm 11; an older npm refuses it
npx -y npm@11 ci --no-audit --no-fund
LLAMA_BUILD_NUMBER="b${COMMIT:0:8}" npx -y npm@11 run build
[ -f dist/index.html ] || { echo "the build left no dist/index.html"; exit 1; }

echo "== replacing $DEST"
find "$DEST" -mindepth 1 ! -name loading.html -delete
cp -r dist/. "$DEST/"
cat > "$DEST/UPSTREAM" <<END
The web UI of llama.cpp ("Llama Chat"), as built from ui/apps/chat of
$REPO

commit: $COMMIT
date:   $DATE

llama-server serves these files with --webui llamacpp. loading.html is the page
shown while the model loads and is not part of that build.

To replace them with a newer build: scripts/update-webui-llamacpp.sh [commit]
END
echo "== done: $(find "$DEST" -type f | wc -l) files. Build llama-server, try --webui llamacpp, then commit."
