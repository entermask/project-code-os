#!/usr/bin/env python3
"""Compare raw Higgs semantic-code records from two quality arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError(f"Unsupported semantic summary: {path}")
    return value


def keyed(summary: dict) -> dict[tuple[str, int], dict]:
    rows = {
        (str(row["prompt_id"]), int(row["repeat"])): row
        for row in summary["rows"]
    }
    if len(rows) != len(summary["rows"]):
        raise ValueError("Duplicate prompt/repeat semantic rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    baseline_rows = keyed(baseline)
    candidate_rows = keyed(candidate)
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("Semantic prompt/repeat sets differ")

    comparisons = []
    for key in sorted(baseline_rows):
        before = baseline_rows[key]
        after = candidate_rows[key]
        comparisons.append(
            {
                "prompt_id": key[0],
                "repeat": key[1],
                "prefix_exact": (
                    before["prefix_sha256"] == after["prefix_sha256"]
                ),
                "full_exact": before["full_sha256"] == after["full_sha256"],
                "baseline_frames": before["frames"],
                "candidate_frames": after["frames"],
            }
        )
    output = {
        "schema_version": 1,
        "baseline_arm": baseline["arm"],
        "candidate_arm": candidate["arm"],
        "pairs": len(comparisons),
        "prefix_exact_pairs": sum(
            row["prefix_exact"] for row in comparisons
        ),
        "full_exact_pairs": sum(row["full_exact"] for row in comparisons),
        "all_prefix_exact": all(
            row["prefix_exact"] for row in comparisons
        ),
        "all_full_exact": all(row["full_exact"] for row in comparisons),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
