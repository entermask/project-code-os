#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/sglang-omni-worktree" >&2
  exit 2
fi

repo=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
patch="$script_dir/patches/0001-prefill-admission-coalescing.patch"
expected_head=df62e91a00d383e6f73ab9604386ffac6c520529
expected_1008a_diff=6f8504a3230652a8bc1f6b943fe878900ef99a84251f810fcb3b52bcf09cd5b3
expected_scheduler=fecf292edd21b72c5ad53649e42e5fd9fa935a465fe0b2bea4d3b9789ecf1695
expected_stages=20a0bbbc9d328f42262f5040d0c52509b2ceef25bffcdff7a4caa3c650da6e3f

actual_head=$(git -C "$repo" rev-parse HEAD)
if [[ "$actual_head" != "$expected_head" ]]; then
  echo "refusing to patch unexpected HEAD: $actual_head" >&2
  exit 1
fi

if ! git -C "$repo" diff --cached --quiet --exit-code; then
  echo "refusing to patch a worktree with staged changes" >&2
  exit 1
fi

if [[ -n "$(git -C "$repo" ls-files --others --exclude-standard)" ]]; then
  echo "refusing to patch a worktree with untracked files" >&2
  exit 1
fi

actual_1008a_diff=$(git -C "$repo" diff --no-ext-diff | sha256sum | awk '{print $1}')
if [[ "$actual_1008a_diff" != "$expected_1008a_diff" ]]; then
  echo "refusing to patch unexpected 1008a candidate: $actual_1008a_diff" >&2
  exit 1
fi

actual_scheduler=$(sha256sum "$repo/sglang_omni/scheduling/omni_scheduler.py" | awk '{print $1}')
actual_stages=$(sha256sum "$repo/sglang_omni/models/higgs_tts/stages.py" | awk '{print $1}')
if [[ "$actual_scheduler" != "$expected_scheduler" ]]; then
  echo "refusing unexpected scheduler preimage: $actual_scheduler" >&2
  exit 1
fi
if [[ "$actual_stages" != "$expected_stages" ]]; then
  echo "refusing unexpected stages preimage: $actual_stages" >&2
  exit 1
fi

git -C "$repo" apply --unidiff-zero --check "$patch"
git -C "$repo" apply --unidiff-zero "$patch"
git -C "$repo" diff --check
git -C "$repo" status --short
