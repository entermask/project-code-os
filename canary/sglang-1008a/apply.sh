#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/sglang-omni-worktree" >&2
  exit 2
fi

repo=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
patch_dir="$script_dir/patches"
expected_head=df62e91a00d383e6f73ab9604386ffac6c520529
expected_prod_diff=9a3ba2d6f6b8459e631b488b76eb5a9a96432ed32edc5dcab770789dd4ef6ad4

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

actual_prod_diff=$(git -C "$repo" diff --no-ext-diff | sha256sum | awk '{print $1}')
if [[ "$actual_prod_diff" != "$expected_prod_diff" ]]; then
  echo "refusing to patch unexpected production overlay: $actual_prod_diff" >&2
  exit 1
fi

for patch in \
  "$patch_dir/0001-test-align-production-fixtures.patch" \
  "$patch_dir/0002-preallocate-output-code-buffer.patch"
do
  git -C "$repo" apply --check "$patch"
  git -C "$repo" apply "$patch"
done

git -C "$repo" diff --check
git -C "$repo" status --short
