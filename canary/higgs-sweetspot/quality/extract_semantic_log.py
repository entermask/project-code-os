#!/usr/bin/env python3
"""Pair test-only semantic-code log records with a quality-gate summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PATTERN = re.compile(
    r"HIGGS_SEMANTIC_CODES "
    r"request_id=(?P<request_id>\S+) "
    r"frames=(?P<frames>\d+) "
    r"prefix_sha256=(?P<prefix_sha256>[0-9a-f]{64}) "
    r"full_sha256=(?P<full_sha256>[0-9a-f]{64}) "
    r"prefix=(?P<prefix>\[\[.*\]\])$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    expected = summary["results"]
    parsed: list[dict] = []
    for line in args.log.read_text(encoding="utf-8", errors="strict").splitlines():
        match = PATTERN.search(line)
        if not match:
            continue
        row = match.groupdict()
        row["frames"] = int(row["frames"])
        row["prefix"] = json.loads(row["prefix"])
        parsed.append(row)

    if len(parsed) < len(expected):
        raise RuntimeError(
            f"Only {len(parsed)} semantic records for {len(expected)} requests"
        )
    parsed = parsed[-len(expected) :]
    rows = []
    for quality, semantic in zip(expected, parsed, strict=True):
        expected_frames = max(
            0,
            int(quality["usage"]["completion_tokens"]) - 7,
        )
        if semantic["frames"] != expected_frames:
            raise RuntimeError(
                f"Frame/token mismatch for {quality['prompt_id']} "
                f"r{quality['repeat']:02d}: "
                f"{semantic['frames']} != {expected_frames}"
            )
        rows.append(
            {
                "prompt_id": quality["prompt_id"],
                "repeat": quality["repeat"],
                **semantic,
            }
        )

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["prompt_id"], []).append(row)
    repeat_exact = {
        prompt_id: len({row["full_sha256"] for row in group}) == 1
        for prompt_id, group in grouped.items()
    }
    output = {
        "schema_version": 1,
        "arm": summary["arm"],
        "mode": summary["mode"],
        "source_log": str(args.log),
        "source_log_sha256": sha256_file(args.log),
        "all_repeat_exact": all(repeat_exact.values()),
        "repeat_exact_by_prompt": repeat_exact,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(rows),
                "all_repeat_exact": output["all_repeat_exact"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
