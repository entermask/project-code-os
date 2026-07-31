#!/usr/bin/env python3
"""Summarize repeated Higgs sweet-spot benchmark waves.

Labels must use ``ARM--WORKLOAD--RUN`` so the report can compare matching
workloads without silently pooling different traffic shapes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


METRICS: dict[str, Callable[[dict[str, Any]], float]] = {
    "completion_tokens_per_s": lambda row: float(
        row["rates"]["completion_tokens_per_s"]
    ),
    "audio_s_per_s": lambda row: float(row["rates"]["estimated_audio_s_per_s"]),
    "chunks_per_s": lambda row: float(row["rates"]["chunks_per_s"]),
    "total_wall_s": lambda row: float(row["timing"]["total_wall_s"]),
    "job_latency_p50_s": lambda row: float(row["latency_s"]["job"]["p50"]),
    "job_latency_p95_s": lambda row: float(row["latency_s"]["job"]["p95"]),
    "gpu_power_avg_w": lambda row: float(row["gpu"]["power_w"]["avg"]),
    "gpu_power_max_w": lambda row: float(row["gpu"]["power_w"]["max"]),
    "gpu_temperature_max_c": lambda row: float(row["gpu"]["temperature_c"]["max"]),
    "gpu_sm_clock_avg_mhz": lambda row: float(row["gpu"]["sm_clock_mhz"]["avg"]),
    "gpu_memory_used_max_mb": lambda row: float(row["gpu"]["memory_used_mb"]["max"]),
}

HIGHER_IS_BETTER = {
    "completion_tokens_per_s",
    "audio_s_per_s",
    "chunks_per_s",
}


def normalize_json_numbers(value: Any) -> Any:
    """Make numerically equal JSON values stable across argparse representations."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [normalize_json_numbers(item) for item in value]
    if isinstance(value, dict):
        return {
            key: normalize_json_numbers(item)
            for key, item in value.items()
        }
    return value


def workload_signature(row: dict[str, Any]) -> str:
    """Return the traffic-shape signature that must match across benchmark arms."""
    return json.dumps(
        normalize_json_numbers(
            {
                "config": row["config"],
                "ref_pool": row.get("ref_pool"),
                "text_chunks": row.get("text_chunks"),
            }
        ),
        sort_keys=True,
        separators=(",", ":"),
    )


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def bootstrap_ratio_ci(
    baseline: list[float],
    candidate: list[float],
    *,
    seed: int = 20260730,
    samples: int = 20_000,
) -> dict[str, float | int | None]:
    paired = min(len(baseline), len(candidate))
    if paired == 0:
        return {"paired_waves": 0, "ratio": None, "ci95_low": None, "ci95_high": None}
    ratios = [candidate[index] / baseline[index] for index in range(paired)]
    point = math.exp(statistics.fmean(math.log(value) for value in ratios))
    if paired == 1:
        return {
            "paired_waves": 1,
            "ratio": point,
            "ci95_low": None,
            "ci95_high": None,
        }

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        picked = [ratios[rng.randrange(paired)] for _ in range(paired)]
        draws.append(math.exp(statistics.fmean(math.log(value) for value in picked)))
    return {
        "paired_waves": paired,
        "ratio": point,
        "ci95_low": percentile(draws, 2.5),
        "ci95_high": percentile(draws, 97.5),
    }


def load_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not all(key in row for key in ("label", "counts", "rates", "gpu")):
            continue
        parts = str(row["label"]).split("--")
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"{path}: label must be ARM--WORKLOAD--RUN, got {row['label']!r}"
            )
        row["_path"] = str(path)
        row["_arm"], row["_workload"], row["_run"] = parts
        row["_signature"] = workload_signature(row)
        rows.append(row)
    if not rows:
        raise ValueError(f"No benchmark result JSON found under {root}")
    return rows


def validate_run(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    counts = row["counts"]
    jobs = int(row["config"]["jobs"])
    chunks = int(row["config"]["chunks_per_job"])
    if int(counts["succeeded"]) != jobs:
        errors.append(f"succeeded={counts['succeeded']} expected={jobs}")
    if int(counts["chunks_completed"]) != jobs * chunks:
        errors.append(
            f"chunks_completed={counts['chunks_completed']} expected={jobs * chunks}"
        )
    if int(counts["failed_or_timeout"]) or int(counts["rejected"]):
        errors.append(
            f"failed_or_timeout={counts['failed_or_timeout']} rejected={counts['rejected']}"
        )
    if int(row["gpu"].get("samples", 0)) == 0:
        errors.append("no valid GPU telemetry")
    if row["gpu"].get("sampler_errors"):
        errors.append(f"GPU sampler errors={len(row['gpu']['sampler_errors'])}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = load_runs(args.root)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    invalid: list[dict[str, Any]] = []
    for row in rows:
        problems = validate_run(row)
        if problems:
            invalid.append({"path": row["_path"], "errors": problems})
            continue
        grouped[row["_arm"]][row["_workload"]].append(row)

    if invalid:
        raise RuntimeError(
            "Refusing to summarize invalid benchmark waves:\n"
            + json.dumps(invalid, indent=2)
        )
    if args.baseline not in grouped:
        raise ValueError(f"Missing baseline arm: {args.baseline}")

    for arm, workloads in grouped.items():
        for workload, workload_rows in workloads.items():
            run_ids = [row["_run"] for row in workload_rows]
            if len(run_ids) != len(set(run_ids)):
                raise ValueError(f"Duplicate run IDs for {arm}/{workload}: {run_ids}")
            signatures = {row["_signature"] for row in workload_rows}
            if len(signatures) != 1:
                raise ValueError(
                    f"Traffic configuration changed within {arm}/{workload}"
                )

    summaries: dict[str, Any] = {}
    for arm, workloads in sorted(grouped.items()):
        summaries[arm] = {}
        for workload, workload_rows in sorted(workloads.items()):
            workload_rows.sort(key=lambda row: row["_run"])
            summaries[arm][workload] = {
                "runs": [row["_run"] for row in workload_rows],
                "metrics": {
                    metric: distribution([extractor(row) for row in workload_rows])
                    for metric, extractor in METRICS.items()
                },
            }

    comparisons: dict[str, Any] = {}
    baseline_workloads = grouped[args.baseline]
    for arm, workloads in sorted(grouped.items()):
        if arm == args.baseline:
            continue
        comparisons[arm] = {}
        for workload in sorted(set(baseline_workloads) & set(workloads)):
            baseline_by_run = {
                row["_run"]: row for row in baseline_workloads[workload]
            }
            candidate_by_run = {row["_run"]: row for row in workloads[workload]}
            if set(baseline_by_run) != set(candidate_by_run):
                raise ValueError(
                    f"Unpaired run IDs for {arm}/{workload}: "
                    f"baseline={sorted(baseline_by_run)} "
                    f"candidate={sorted(candidate_by_run)}"
                )
            baseline_signatures = {
                row["_signature"] for row in baseline_by_run.values()
            }
            candidate_signatures = {
                row["_signature"] for row in candidate_by_run.values()
            }
            if baseline_signatures != candidate_signatures:
                raise ValueError(
                    f"Traffic configuration mismatch for {arm}/{workload}"
                )
            paired_runs = sorted(baseline_by_run)
            baseline_rows = [baseline_by_run[run] for run in paired_runs]
            candidate_rows = [candidate_by_run[run] for run in paired_runs]
            comparisons[arm][workload] = {}
            for metric, extractor in METRICS.items():
                result = bootstrap_ratio_ci(
                    [extractor(row) for row in baseline_rows],
                    [extractor(row) for row in candidate_rows],
                )
                ratio = result["ratio"]
                result["delta_pct"] = (
                    ((float(ratio) - 1.0) * 100.0)
                    if ratio is not None and metric in HIGHER_IS_BETTER
                    else ((1.0 - float(ratio)) * 100.0)
                    if ratio is not None
                    else None
                )
                comparisons[arm][workload][metric] = result

    report = {
        "baseline": args.baseline,
        "valid_waves": len(rows),
        "arms": summaries,
        "comparisons": comparisons,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
