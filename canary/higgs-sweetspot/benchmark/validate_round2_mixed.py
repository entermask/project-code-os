#!/usr/bin/env python3
"""Validate and summarize one synchronized long/short mixed wave."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_clean(result: dict[str, Any], jobs: int, chunks: int, lane: str) -> None:
    counts = result["counts"]
    assert counts["submitted"] == jobs, counts
    assert counts["accepted"] == jobs, counts
    assert counts["succeeded"] == jobs, counts
    assert counts["failed_or_timeout"] == 0, counts
    assert counts["chunks_completed"] == chunks, counts
    assert counts["chunks_failed"] == 0, counts
    assert counts["chunks_degraded"] == 0, counts
    assert result["cache"]["cache_misses"] == 0, result["cache"]
    assert all(job["lane"] == lane for job in result["jobs"]), lane


def require_paired_config(
    long_result: dict[str, Any], short_result: dict[str, Any]
) -> tuple[str, int | None]:
    long_config = long_result["config"]
    short_config = short_result["config"]
    long_marker = long_config["run_marker"]
    short_marker = short_config["run_marker"]
    assert isinstance(long_marker, str)
    assert re.fullmatch(r"[A-Za-z0-9_-]{8}", long_marker, flags=re.ASCII), (
        f"invalid long run marker: {long_marker!r}"
    )
    assert short_marker == long_marker, (
        f"mixed run marker mismatch: long={long_marker!r}, short={short_marker!r}"
    )

    long_top_k = long_config["sampling"]["top_k"]
    short_top_k = short_config["sampling"]["top_k"]
    if long_top_k is None:
        assert short_top_k is None, (
            f"mixed top_k mismatch: long={long_top_k!r}, short={short_top_k!r}"
        )
        return long_marker, None
    assert type(long_top_k) is int and long_top_k > 0, (
        f"invalid long top_k: {long_top_k!r}"
    )
    assert short_top_k == long_top_k, (
        f"mixed top_k mismatch: long={long_top_k!r}, short={short_top_k!r}"
    )
    return long_marker, long_top_k


def require_run_identity(
    long_result: dict[str, Any],
    short_result: dict[str, Any],
    run_marker: str,
) -> int:
    long_match = re.search(r"-mixed-long-r([0-9]+)$", long_result["label"])
    short_match = re.search(r"-mixed-short-r([0-9]+)$", short_result["label"])
    assert long_match is not None, f"invalid long label: {long_result['label']!r}"
    assert short_match is not None, f"invalid short label: {short_result['label']!r}"
    long_run = int(long_match.group(1))
    short_run = int(short_match.group(1))
    assert short_run == long_run, (
        f"mixed run index mismatch: long={long_run}, short={short_run}"
    )
    assert long_run <= 9_999_999
    expected_marker = f"R{long_run:07d}"
    assert run_marker == expected_marker, (
        f"run marker does not match run index: expected={expected_marker!r}, "
        f"actual={run_marker!r}"
    )
    return long_run


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: validate_round2_mixed.py LONG.json SHORT.json OUTPUT.json"
        )
    long_result = load(sys.argv[1])
    short_result = load(sys.argv[2])
    require_clean(long_result, jobs=10, chunks=100, lane="long")
    require_clean(short_result, jobs=4, chunks=16, lane="short")
    run_marker, top_k = require_paired_config(long_result, short_result)
    run_index = require_run_identity(long_result, short_result, run_marker)

    long_submit = long_result["timing"]["submission"]["first_submit_monotonic_ns"]
    short_submit = short_result["timing"]["submission"]["first_submit_monotonic_ns"]
    assert long_submit is not None and short_submit is not None
    skew_ms = (short_submit - long_submit) / 1_000_000
    # Process scheduling under an already-saturated mixed wave can add a few
    # dozen milliseconds around the intended 750 ms offset. Keep a bounded
    # -70/+80 ms gate: tight enough to preserve the workload shape, while avoiding
    # false rejection from host scheduling noise.
    assert 680 <= skew_ms <= 830, f"mixed submit skew out of gate: {skew_ms:.3f} ms"

    jobs = long_result["jobs"] + short_result["jobs"]
    created = [job["created_at"] for job in jobs]
    updated = [job["updated_at"] for job in jobs]
    assert all(value is not None for value in created + updated)
    server_window_s = max(updated) - min(created)
    completion_tokens = sum(job["completion_tokens"] for job in jobs)
    summary = {
        "long_label": long_result["label"],
        "short_label": short_result["label"],
        "run_index": run_index,
        "run_marker": run_marker,
        "top_k": top_k,
        "submit_skew_ms": skew_ms,
        "long_submit_fanout_ms": long_result["timing"]["submission"][
            "submit_fanout_ms"
        ],
        "short_submit_fanout_ms": short_result["timing"]["submission"][
            "submit_fanout_ms"
        ],
        "long": {
            "server_completion_tokens_per_s": long_result["rates"][
                "server_completion_tokens_per_s"
            ],
            "server_job": long_result["latency_s"]["server_job"],
            "server_queue": long_result["latency_s"]["server_queue"],
            "server_service": long_result["latency_s"]["server_service"],
        },
        "short": {
            "server_completion_tokens_per_s": short_result["rates"][
                "server_completion_tokens_per_s"
            ],
            "server_job": short_result["latency_s"]["server_job"],
            "server_queue": short_result["latency_s"]["server_queue"],
            "server_service": short_result["latency_s"]["server_service"],
        },
        "combined": {
            "completion_tokens": completion_tokens,
            "server_window_s": server_window_s,
            "server_completion_tokens_per_s": completion_tokens / server_window_s,
        },
    }
    Path(sys.argv[3]).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
