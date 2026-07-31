#!/usr/bin/env python3
"""Compare two fixed Higgs quality-gate summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("schema_version") != 1:
        raise ValueError(f"Unsupported quality-gate schema: {path}")
    return row


def result_map(summary: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    mapped = {
        (str(row["prompt_id"]), int(row["repeat"])): row
        for row in summary["results"]
    }
    if len(mapped) != len(summary["results"]):
        raise ValueError("Duplicate prompt/repeat rows")
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    if baseline["mode"] != candidate["mode"]:
        raise ValueError("Quality-gate modes differ")
    if baseline["reference"]["sha256"] != candidate["reference"]["sha256"]:
        raise ValueError("Reference fingerprints differ")
    if baseline["settings"] != candidate["settings"]:
        raise ValueError("Quality-gate settings differ")
    if baseline["mode"] == "prefix" and (
        not baseline["all_repeat_exact"] or not candidate["all_repeat_exact"]
    ):
        raise ValueError("At least one deterministic arm was not repeat-exact")

    baseline_rows = result_map(baseline)
    candidate_rows = result_map(candidate)
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("Prompt/repeat sets differ")

    comparisons: list[dict[str, Any]] = []
    failures: list[str] = []
    for key in sorted(baseline_rows):
        base = baseline_rows[key]
        cand = candidate_rows[key]
        if base["prompt_sha256"] != cand["prompt_sha256"]:
            raise ValueError(f"Prompt fingerprint differs for {key}")
        duration_ratio = cand["wav"]["duration_s"] / base["wav"]["duration_s"]
        base_tokens = int(base["usage"]["completion_tokens"])
        cand_tokens = int(cand["usage"]["completion_tokens"])
        token_ratio = cand_tokens / base_tokens if base_tokens else None
        pcm_exact = cand["wav"]["pcm_sha256"] == base["wav"]["pcm_sha256"]
        row = {
            "prompt_id": key[0],
            "repeat": key[1],
            "pcm_exact": pcm_exact,
            "duration_ratio": duration_ratio,
            "completion_token_ratio": token_ratio,
        }
        comparisons.append(row)
        if baseline["mode"] == "prefix" and not pcm_exact:
            failures.append(f"{key}: prefix PCM differs")
        if not 0.95 <= duration_ratio <= 1.05:
            failures.append(f"{key}: duration ratio {duration_ratio:.6f}")
        if token_ratio is None or not 0.95 <= token_ratio <= 1.05:
            failures.append(f"{key}: completion-token ratio {token_ratio}")

    report = {
        "schema_version": 1,
        "baseline": baseline["arm"],
        "candidate": candidate["arm"],
        "mode": baseline["mode"],
        "pairs": len(comparisons),
        "passed": not failures,
        "failures": failures,
        "comparisons": comparisons,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
