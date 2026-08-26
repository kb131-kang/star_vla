#!/usr/bin/env bash
# Step 1: third_party 원본 리포 클론 (재실행 가능). 원본은 수정하지 않는다.
set -euo pipefail
WS="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$WS/third_party" "$WS/scripts" "$WS/configs" "$WS/logs" "$WS/reports" "$WS/patches" "$WS/docs"
cd "$WS/third_party"
[ -d starVLA/.git ] || git clone -b starVLA --depth 1 https://github.com/starVLA/starVLA.git starVLA
[ -d SimplerEnv/.git ] || git clone --depth 1 https://github.com/simpler-env/SimplerEnv.git SimplerEnv
git -C SimplerEnv submodule update --init --recursive
echo "starVLA:    $(git -C starVLA rev-parse --abbrev-ref HEAD) @ $(git -C starVLA rev-parse --short HEAD)"
echo "SimplerEnv: $(git -C SimplerEnv rev-parse --short HEAD)"
[ -L "$HOME/vla-workspace" ] || ln -sfn "$WS" "$HOME/vla-workspace"
