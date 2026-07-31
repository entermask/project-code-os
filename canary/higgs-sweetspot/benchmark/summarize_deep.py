#!/usr/bin/env python3
"""Build the reproducible summary for the deep Higgs sweet-spot round."""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable


METRICS: dict[str, tuple[str, Callable[[dict[str, Any]], float]]] = {
    "completion_tokens_per_s": (
        "higher_is_better",
        lambda row: float(row["rates"]["completion_tokens_per_s"]),
    ),
    "wall_s": (
        "lower_is_better",
        lambda row: float(row["timing"]["total_wall_s"]),
    ),
    "job_p50_s": (
        "lower_is_better",
        lambda row: float(row["latency_s"]["job"]["p50"]),
    ),
    "job_p95_s": (
        "lower_is_better",
        lambda row: float(row["latency_s"]["job"]["p95"]),
    ),
}


PAIRED_COHORTS = [
    (
        "syncfree_k20_long10",
        "syncfree/off-long10-r*.json",
        "syncfree/on-long10-r*.json",
    ),
    (
        "syncfree_k20_short10",
        "syncfree/off-short10-r*.json",
        "syncfree/on-short10-r*.json",
    ),
    (
        "syncfree_k16_deterministic_long10",
        "syncfree-k16-det/syncfree-off-k16-det-long10-r*.json",
        "syncfree-k16-det/syncfree-on-k16-det-long10-r*.json",
    ),
    (
        "syncfree_k16_deterministic_short10",
        "syncfree-k16-det/syncfree-off-k16-det-short10-r*.json",
        "syncfree-k16-det/syncfree-on-k16-det-short10-r*.json",
    ),
    ("g92_long10", "syncfree/on-long10-r*.json", "g92/long10-r*.json"),
    ("g92_short10", "syncfree/on-short10-r*.json", "g92/short10-r*.json"),
    (
        "fcfs_to_lpm_long10",
        "schedule/fcfs1-long10-r*.json",
        "schedule/lpm1-long10-r*.json",
    ),
    (
        "fcfs_to_lpm_short10",
        "schedule/fcfs1-short10-r*.json",
        "schedule/lpm1-short10-r*.json",
    ),
    (
        "torchcompile_long10",
        "schedule/fcfs1-long10-r*.json",
        "torchcompile/torchcompile-k20-long10-r*.json",
    ),
    (
        "torchcompile_short10",
        "schedule/fcfs1-short10-r*.json",
        "torchcompile/torchcompile-k20-short10-r*.json",
    ),
    (
        "k16_to_k20_fcfs_long10",
        "k-fcfs/k16-fcfs1-long10-r*.json",
        "k-fcfs/k20-fcfs1-confirm-long10-r*.json",
    ),
    (
        "k16_to_k20_fcfs_short10",
        "k-fcfs/k16-fcfs1-short10-r*.json",
        "k-fcfs/k20-fcfs1-confirm-short10-r*.json",
    ),
    (
        "k16_to_k20_fcfs_true_short4",
        "true-short/k16-fcfs1-true-short4-r*.json",
        "true-short/k20-fcfs1-true-short4-r*.json",
    ),
    (
        "pcm_all_mp3_loudnorm_long10",
        "pcm-all-mp3/pcm-all-off-mp3-long10-r*.json",
        "pcm-all-mp3/pcm-all-on-mp3-long10-r*.json",
    ),
    (
        "pcm_all_mp3_loudnorm_short10",
        "pcm-all-mp3/pcm-all-off-mp3-short10-r*.json",
        "pcm-all-mp3/pcm-all-on-mp3-short10-r*.json",
    ),
    (
        "pcm_all_mp3_loudnorm_true_short4",
        "pcm-all-mp3/pcm-all-off-mp3-true-short4-r*.json",
        "pcm-all-mp3/pcm-all-on-mp3-true-short4-r*.json",
    ),
]


ARM_COHORTS = [
    ("final_sustained_20x10", "final-sustained/*.json"),
    ("trtllm_long10", "trtllm/long10-r*.json"),
    ("trtllm_short10", "trtllm/short10-r*.json"),
]


def load_rows(root: Path, pattern: str) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(str(root / pattern)))
    if not paths:
        raise FileNotFoundError(f"No benchmark files matched {pattern!r}")
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]


def geometric_effect(ratios: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in ratios)) - 1.0


def bootstrap_ci(
    ratios: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    effects = []
    for _ in range(samples):
        resample = [ratios[rng.randrange(len(ratios))] for _ in ratios]
        effects.append(geometric_effect(resample))
    effects.sort()
    low_index = int(0.025 * samples)
    high_index = max(low_index, int(0.975 * samples) - 1)
    return effects[low_index], effects[high_index]


def summarize_pair(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(before) != len(after):
        raise ValueError("Paired cohorts have different lengths")
    metrics: dict[str, Any] = {}
    for metric_name, (direction, getter) in METRICS.items():
        before_values = [getter(row) for row in before]
        after_values = [getter(row) for row in after]
        ratios = [
            after_value / before_value
            for before_value, after_value in zip(before_values, after_values)
        ]
        effect = geometric_effect(ratios)
        low, high = bootstrap_ci(
            ratios,
            samples=bootstrap_samples,
            seed=seed,
        )
        metrics[metric_name] = {
            "direction": direction,
            "before_mean": statistics.fmean(before_values),
            "after_mean": statistics.fmean(after_values),
            "paired_geometric_effect_pct": effect * 100.0,
            "bootstrap_95_ci_pct": [low * 100.0, high * 100.0],
            "paired_effects_pct": [(ratio - 1.0) * 100.0 for ratio in ratios],
        }
    return {
        "waves": len(before),
        "before_failed_or_timeout": sum(
            int(row["counts"]["failed_or_timeout"]) for row in before
        ),
        "after_failed_or_timeout": sum(
            int(row["counts"]["failed_or_timeout"]) for row in after
        ),
        "metrics": metrics,
    }


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "waves": len(rows),
        "chunks_completed": sum(int(row["counts"]["chunks_completed"]) for row in rows),
        "failed_or_timeout": sum(
            int(row["counts"]["failed_or_timeout"]) for row in rows
        ),
        "completion_tokens": [int(row["usage"]["completion_tokens"]) for row in rows],
        "completion_tokens_per_s_mean": statistics.fmean(
            float(row["rates"]["completion_tokens_per_s"]) for row in rows
        ),
        "wall_s_mean": statistics.fmean(
            float(row["timing"]["total_wall_s"]) for row in rows
        ),
        "peak_memory_used_mb": max(
            float(row["gpu"]["memory_used_mb"]["max"]) for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20_260_730)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")

    paired = {}
    for index, (name, before_pattern, after_pattern) in enumerate(PAIRED_COHORTS):
        paired[name] = summarize_pair(
            load_rows(args.root, before_pattern),
            load_rows(args.root, after_pattern),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + index,
        )

    arms = {
        name: summarize_arm(load_rows(args.root, pattern))
        for name, pattern in ARM_COHORTS
    }
    summary = {
        "schema_version": 1,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "method": "paired percentile bootstrap of geometric mean ratios",
        },
        "paired_cohorts": paired,
        "arm_cohorts": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
