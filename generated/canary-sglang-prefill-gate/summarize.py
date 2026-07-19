#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PHASES = {
    "single": {
        "k0_start": "prefill-k0-single-eafd437",
        "k0_end": "prefill-k0-drift-single-eafd437",
        "k8": "prefill-k8-single-eafd437",
        "k12": "prefill-k12-single-d2249e4",
        "k16": "prefill-k16-single-eafd437",
        "k20": "prefill-k20-single-eafd437",
    },
    "dual": {
        "k0_start": "prefill-k0-dual-eafd437",
        "k0_end": "prefill-k0-drift-dual-eafd437",
        "k8": "prefill-k8-dual-eafd437",
        "k12": "prefill-k12-dual-d2249e4",
        "k16": "prefill-k16-dual-eafd437",
        "k20": "prefill-k20-dual-eafd437",
    },
}
METRICS = {
    "service_makespan_s": False,
    "service_chunks_per_s": True,
    "service_audio_s_per_s": True,
    "service_completion_tokens_per_s": True,
}


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def load_phase(name: str) -> dict[str, Any]:
    directory = ROOT / name
    report_path = directory / "benchmark-report.json"
    expected_sha = (directory / "benchmark-report.sha256").read_text().split()[0]
    actual_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(f"Checksum mismatch for {name}: {actual_sha} != {expected_sha}")
    report = json.loads(report_path.read_text())
    if not report["valid"] or report["service_integrity"]["errors"]:
        raise RuntimeError(f"Invalid benchmark phase: {name}")
    return report


def wave_metrics(report: dict[str, Any]) -> dict[str, list[float]]:
    result = {metric: [] for metric in METRICS}
    for wave in report["measured_waves"]:
        makespan = float(wave["service_makespan_s"])
        result["service_makespan_s"].append(makespan)
        result["service_chunks_per_s"].append(float(wave["service_chunks_per_s"]))
        result["service_audio_s_per_s"].append(float(wave["service_audio_s_per_s"]))
        completion_tokens = sum(int(job["completion_tokens"]) for job in wave["jobs"])
        result["service_completion_tokens_per_s"].append(completion_tokens / makespan)
    return result


def bootstrap_improvement(
    control: list[float], candidate: list[float], higher_is_better: bool, seed: int
) -> dict[str, float]:
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(50_000):
        control_mean = statistics.fmean(rng.choice(control) for _ in control)
        candidate_mean = statistics.fmean(rng.choice(candidate) for _ in candidate)
        if higher_is_better:
            improvement = (candidate_mean / control_mean - 1) * 100
        else:
            improvement = (1 - candidate_mean / control_mean) * 100
        estimates.append(improvement)
    point_control = statistics.fmean(control)
    point_candidate = statistics.fmean(candidate)
    point = (
        (point_candidate / point_control - 1) * 100
        if higher_is_better
        else (1 - point_candidate / point_control) * 100
    )
    return {
        "percent": point,
        "ci95_low": percentile(estimates, 2.5),
        "ci95_high": percentile(estimates, 97.5),
        "probability_improvement": sum(value > 0 for value in estimates) / len(estimates),
    }


def phase_quality(report: dict[str, Any]) -> dict[str, Any]:
    profile = report["request_profile"]
    return {
        "valid": report["valid"],
        "waves": report["counts_measured"]["waves"],
        "jobs": report["counts_measured"]["jobs"],
        "chunks": report["counts_measured"]["chunks"],
        "chunks_failed": report["counts_measured"]["chunks_failed"],
        "chunks_degraded": report["counts_measured"]["chunks_degraded"],
        "poll_errors": report["counts_measured"]["poll_errors"],
        "retry_matches": report["log_audit"]["retry_matches"],
        "error_matches": report["log_audit"]["error_matches"],
        "invalid_wavs": len(report["wav_validation_all_runs"]["invalid"]),
        "queue_wait_ms": profile["queue_wait_ms"],
        "engine_complete_after_queue_ms": profile["engine_complete_after_queue_ms"],
        "max_client_submit_skew_ms": max(
            float(wave["client_submit_skew_s"]) for wave in report["measured_waves"]
        )
        * 1000,
        "max_server_create_skew_ms": max(
            float(wave["server_create_skew_s"]) for wave in report["measured_waves"]
        )
        * 1000,
    }


def main() -> None:
    output: dict[str, Any] = {
        "control": "pooled k0_start + k0_end",
        "bootstrap_iterations": 50_000,
        "modes": {},
    }
    for mode, phase_names in PHASES.items():
        reports = {label: load_phase(name) for label, name in phase_names.items()}
        arrays = {label: wave_metrics(report) for label, report in reports.items()}
        control = {
            metric: arrays["k0_start"][metric] + arrays["k0_end"][metric]
            for metric in METRICS
        }
        mode_output: dict[str, Any] = {
            "control_pooled": {
                metric: distribution(values) for metric, values in control.items()
            },
            "phases": {},
        }
        for label, report in reports.items():
            phase = {
                "directory": phase_names[label],
                "metrics": {
                    metric: distribution(values)
                    for metric, values in arrays[label].items()
                },
                "quality": phase_quality(report),
            }
            if label in {"k8", "k12", "k16", "k20"}:
                phase["vs_control_pooled"] = {
                    metric: bootstrap_improvement(
                        control[metric],
                        arrays[label][metric],
                        higher_is_better,
                        seed=int(label[1:]) * 100 + len(metric) + len(mode),
                    )
                    for metric, higher_is_better in METRICS.items()
                }
            mode_output["phases"][label] = phase
        output["modes"][mode] = mode_output
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
