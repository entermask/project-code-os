#!/usr/bin/env python3
"""Production-shaped benchmark for the local TTS wrapper.

The benchmark submits N jobs at once. Each job contains M chunks, matching the
production pattern where one logical TTS request fans out to multiple backend
speech requests.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import hashlib
import http.server
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


DEFAULT_LONG_CHUNKS = [
    "On a cold morning near the harbor, the city sounded different from the way it did in summer. The streets were still busy, but every engine, footstep, and conversation seemed softened by the fog.",
    "A ferry moved slowly across the gray water while workers carried crates from the market into narrow side streets. Nobody was in a hurry, yet everything kept moving with quiet precision.",
    "Inside a small recording room above a closed bookstore, Daniel listened to the same sentence again and again.",
    "He was not looking for a perfect voice; he was looking for a voice that stayed natural when the paragraph became longer. Short samples were easy to impress with, but long passages revealed the truth.",
    "A model could sound confident for ten seconds and then lose rhythm, swallow consonants, or drift away from the original speaker. He wrote each result into a notebook.",
    "First came the response time, then the number of chunks, then the total duration of the generated audio. After that he added comments about pacing, breath, emphasis, and fatigue.",
    "The best system was not always the one with the fastest single request.",
    "It was the one that remained predictable when several requests arrived at once, and still produced speech that people could listen to without distraction.",
    "By the afternoon, the fog had lifted and sunlight reached the windows. Daniel played the final sample from beginning to end without touching the keyboard. The voice held together across the whole passage.",
    "It was not flawless, but it was steady, clear, and fast enough to be useful. That was the result he needed: not a miracle, just a reliable tool that could survive real work.",
]

DEFAULT_SHORT_CHUNKS = [
    "The harbor lights turned on before sunrise.",
    "Every operator checked the same short list.",
    "A quiet signal moved through the control room.",
    "The next request arrived without warning.",
    "Nobody paused, but the rhythm stayed clean.",
    "The speaker kept a steady natural pace.",
    "Each sentence was brief and easy to verify.",
    "The system answered with a clear voice.",
    "Latency mattered more than decoration.",
    "The final sample finished on time.",
]

DEFAULT_CHUNKS_PER_JOB = 10
MAX_CHUNKS_PER_JOB = min(len(DEFAULT_LONG_CHUNKS), len(DEFAULT_SHORT_CHUNKS))
RUN_MARKER_LENGTH = 8

REFS = {
    "d6905d7a": {
        "file": "ref_d6905d7a.mp3",
        "text": "Our distrust is very expensive.",
    },
    "b152986c": {
        "file": "ref_b152986c.mp3",
        "text": "Công cha như núi Thái Sơn, nghĩa mẹ như nước trong nguồn chảy ra.",
    },
}


@dataclass
class JobResult:
    index: int
    accepted: bool
    status_code: int
    submit_latency_s: float
    submit_started_s: float
    accepted_s: float | None = None
    finished_s: float | None = None
    request_id: str | None = None
    status_url: str | None = None
    final_status: str | None = None
    detail: str | None = None
    chunks_completed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    engine_time_s: float = 0.0
    cache_hit: bool | None = None
    submit_started_monotonic_ns: int | None = None
    submit_started_wall_time_ns: int | None = None
    created_at: float | None = None
    started_at: float | None = None
    updated_at: float | None = None
    lane: str | None = None
    chunks_failed: int | None = None
    chunks_degraded: int | None = None

    @property
    def job_latency_s(self) -> float | None:
        if self.finished_s is None:
            return None
        return self.finished_s - self.submit_started_s

    @property
    def server_job_latency_s(self) -> float | None:
        if self.created_at is None or self.updated_at is None:
            return None
        return self.updated_at - self.created_at

    @property
    def server_queue_latency_s(self) -> float | None:
        if self.created_at is None or self.started_at is None:
            return None
        return self.started_at - self.created_at

    @property
    def server_service_latency_s(self) -> float | None:
        if self.started_at is None or self.updated_at is None:
            return None
        return self.updated_at - self.started_at


class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize_values(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "avg": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def ref_http_server(ref_dir: Path, port: int):
    selected_port = port or find_free_port()
    handler = functools.partial(QuietHTTPRequestHandler, directory=str(ref_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", selected_port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield selected_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def prepare_many_refs(ref_dir: Path, base_file: str, count: int) -> list[str]:
    root = ref_dir.resolve()
    source = resolve_reference_path(ref_dir, base_file)
    target_dir = root / f"bench_refs_{Path(base_file).stem}_{count}"
    if target_dir.is_symlink():
        raise ValueError(f"Refusing symlinked many-ref directory: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir = target_dir.resolve()
    try:
        target_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Many-ref directory escapes --ref-dir: {target_dir}") from exc
    files: list[str] = []
    source_digest = file_sha256(source)
    for index in range(count):
        target = target_dir / f"ref_{index:04d}{source.suffix}"
        if target.is_symlink():
            raise ValueError(f"Refusing symlinked many-ref target: {target}")
        try:
            target.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Many-ref target escapes --ref-dir: {target}") from exc
        if (
            not target.is_file()
            or target.stat().st_size != source.stat().st_size
            or file_sha256(target) != source_digest
        ):
            shutil.copy2(source, target)
        files.append(str(target.relative_to(root)))
    return files


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_reference_path(ref_dir: Path, filename: str) -> Path:
    root = ref_dir.resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Reference audio escapes --ref-dir: {filename}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing reference audio: {candidate}")
    if candidate.suffix.lower() not in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}:
        raise ValueError(f"Unsupported reference-audio suffix: {candidate.suffix}")
    return candidate


def reference_record(ref_dir: Path, key: str, filename: str, text: str) -> dict[str, Any]:
    path = resolve_reference_path(ref_dir, filename)
    return {
        "key": key,
        "file": str(path.relative_to(ref_dir.resolve())),
        "text": text,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def ref_pool(args: argparse.Namespace) -> list[dict[str, Any]]:
    ref_dir = Path(args.ref_dir)
    if args.ref_file:
        base_file = str(resolve_reference_path(ref_dir, args.ref_file).relative_to(ref_dir.resolve()))
        if args.ref_profile == "hot":
            files = [base_file]
        elif args.ref_profile == "many-url":
            files = prepare_many_refs(
                Path(args.ref_dir), base_file, args.many_ref_count
            )
        else:
            raise ValueError("--ref-file supports ref profiles hot and many-url")
        return [reference_record(ref_dir, "custom", filename, args.ref_text) for filename in files]

    if args.ref_profile == "hot":
        keys = [args.ref_key]
        files = [REFS[args.ref_key]["file"]]
    elif args.ref_profile == "two":
        keys = ["d6905d7a", "b152986c"]
        files = [REFS[key]["file"] for key in keys]
    elif args.ref_profile == "many-url":
        keys = [args.ref_key] * args.many_ref_count
        files = prepare_many_refs(Path(args.ref_dir), REFS[args.ref_key]["file"], args.many_ref_count)
    else:
        raise ValueError(f"Unknown ref profile: {args.ref_profile}")

    refs: list[dict[str, Any]] = []
    for key, filename in zip(keys, files):
        refs.append(reference_record(ref_dir, key, filename, REFS[key]["text"]))
    return refs


def parse_chunks_per_job(value: str) -> int:
    try:
        chunks_per_job = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= chunks_per_job <= MAX_CHUNKS_PER_JOB:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {MAX_CHUNKS_PER_JOB}"
        )
    return chunks_per_job


def parse_nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_run_marker(value: str) -> str:
    if value == "":
        return value
    if len(value) != RUN_MARKER_LENGTH:
        raise argparse.ArgumentTypeError(
            f"must contain exactly {RUN_MARKER_LENGTH} characters"
        )
    if not value.isascii() or not all(
        character.isalnum() or character in {"_", "-"} for character in value
    ):
        raise argparse.ArgumentTypeError(
            "must contain only ASCII letters, digits, underscores, or hyphens"
        )
    return value


def chunks_for_job(
    profile: str,
    job_index: int,
    unique_text: bool,
    chunks_per_job: int = DEFAULT_CHUNKS_PER_JOB,
    run_marker: str = "",
) -> list[str]:
    if profile == "long-en":
        chunks = list(DEFAULT_LONG_CHUNKS)
    elif profile == "short-en":
        chunks = list(DEFAULT_SHORT_CHUNKS)
    else:
        raise ValueError(f"Unknown text profile: {profile}")

    if not 1 <= chunks_per_job <= len(chunks):
        raise ValueError(
            f"chunks_per_job must be between 1 and {len(chunks)} for {profile}"
        )
    chunks = chunks[:chunks_per_job]

    if not unique_text and not run_marker:
        return chunks

    if run_marker and unique_text:
        marker = f" Batch marker {run_marker}-{job_index:04d}."
    elif run_marker:
        marker = f" Batch marker {run_marker}."
    else:
        marker = f" Batch marker {job_index:04d}."
    return [f"{chunk}{marker}" for chunk in chunks]


async def get_json(client: httpx.AsyncClient, url: str, headers: dict[str, str]) -> Any:
    response = await client.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


async def wait_until_monotonic_ns(target_ns: int) -> int | None:
    """Wait until an absolute host monotonic timestamp used by paired processes."""
    if target_ns <= 0:
        return None
    while True:
        now_ns = time.monotonic_ns()
        remaining_ns = target_ns - now_ns
        if remaining_ns <= 0:
            return now_ns
        await asyncio.sleep(remaining_ns / 1_000_000_000)


def update_result_from_job_body(result: JobResult, body: dict[str, Any]) -> None:
    """Capture optional job metadata without requiring older API responses to have it."""
    if "status" in body:
        result.final_status = body.get("status")
    if "chunks_completed" in body:
        with contextlib.suppress(TypeError, ValueError):
            result.chunks_completed = int(body.get("chunks_completed") or 0)
    if "cache_hit" in body:
        result.cache_hit = body.get("cache_hit")

    for field in ("created_at", "started_at", "updated_at"):
        value = body.get(field)
        if value is not None:
            with contextlib.suppress(TypeError, ValueError):
                setattr(result, field, float(value))

    lane = body.get("lane")
    if lane is not None:
        result.lane = str(lane)

    for field in ("chunks_failed", "chunks_degraded"):
        if field in body:
            with contextlib.suppress(TypeError, ValueError):
                setattr(result, field, int(body.get(field) or 0))


async def submit_one(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    index: int,
    refs: list[dict[str, str]],
    base_url: str,
    headers: dict[str, str],
) -> JobResult:
    ref = refs[index % len(refs)]
    chunks = chunks_for_job(
        args.text_profile,
        index,
        args.unique_text,
        args.chunks_per_job,
        getattr(args, "run_marker", ""),
    )
    payload: dict[str, Any] = {
        "chunks": chunks,
        "ref_audio_url": (
            f"{args.ref_base_url}/{urllib.parse.quote(ref['file'], safe='/')}"
        ),
        "ref_text": ref["text"],
        "format": args.audio_format,
    }
    if args.af_filter:
        payload["af_filter"] = args.af_filter
    if args.seed is not None:
        payload["seed"] = args.seed + index
    for field in ("speed", "max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty"):
        value = getattr(args, field)
        if value is not None:
            payload[field] = value

    started = time.perf_counter()
    submit_started_monotonic_ns = time.monotonic_ns()
    submit_started_wall_time_ns = time.time_ns()
    try:
        response = await client.post(f"{base_url}/v1/tts", json=payload, headers=headers)
    except Exception as exc:
        now = time.perf_counter()
        return JobResult(
            index=index,
            accepted=False,
            status_code=0,
            submit_latency_s=now - started,
            submit_started_s=started,
            submit_started_monotonic_ns=submit_started_monotonic_ns,
            submit_started_wall_time_ns=submit_started_wall_time_ns,
            detail=repr(exc),
        )

    now = time.perf_counter()
    if response.status_code != 202:
        return JobResult(
            index=index,
            accepted=False,
            status_code=response.status_code,
            submit_latency_s=now - started,
            submit_started_s=started,
            submit_started_monotonic_ns=submit_started_monotonic_ns,
            submit_started_wall_time_ns=submit_started_wall_time_ns,
            detail=response.text[:500],
        )

    body = response.json()
    result = JobResult(
        index=index,
        accepted=True,
        status_code=response.status_code,
        submit_latency_s=now - started,
        submit_started_s=started,
        submit_started_monotonic_ns=submit_started_monotonic_ns,
        submit_started_wall_time_ns=submit_started_wall_time_ns,
        accepted_s=now,
        request_id=body.get("request_id"),
        status_url=body.get("status_url"),
    )
    update_result_from_job_body(result, body)
    return result


async def poll_one(
    client: httpx.AsyncClient,
    result: JobResult,
    base_url: str,
    headers: dict[str, str],
    poll_interval: float,
    timeout_s: float,
) -> JobResult:
    if not result.accepted or not result.status_url:
        return result

    deadline = time.perf_counter() + timeout_s
    url = f"{base_url}{result.status_url}"
    last_body: dict[str, Any] | None = None
    while time.perf_counter() < deadline:
        try:
            response = await client.get(url, headers=headers)
            if response.status_code >= 400:
                result.detail = response.text[:500]
                await asyncio.sleep(poll_interval)
                continue
            body = response.json()
            last_body = body
            update_result_from_job_body(result, body)
            status = result.final_status
            usage = body.get("usage") or {}
            result.prompt_tokens = int(usage.get("prompt_tokens") or 0)
            result.completion_tokens = int(usage.get("completion_tokens") or 0)
            result.total_tokens = int(usage.get("total_tokens") or 0)
            result.engine_time_s = float(usage.get("engine_time_s") or 0.0)
            if status in {"succeeded", "failed"}:
                result.finished_s = time.perf_counter()
                result.detail = body.get("detail")
                return result
        except Exception as exc:
            result.detail = repr(exc)
        await asyncio.sleep(poll_interval)

    result.finished_s = time.perf_counter()
    result.final_status = "timeout"
    if last_body:
        result.detail = json.dumps(last_body, ensure_ascii=False)[:500]
    return result


async def gpu_sampler(stop_event: asyncio.Event, interval_s: float) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if interval_s <= 0 or not shutil.which("nvidia-smi"):
        return samples
    query = (
        "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,"
        "clocks.sm,clocks.mem,pstate"
    )
    while not stop_event.is_set():
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            row = output.strip().splitlines()[0].split(",")
            values = [item.strip() for item in row]
            sample: dict[str, Any] = {
                "t": time.time(),
                "gpu_util_pct": float(values[0]),
                "memory_used_mb": float(values[1]),
                "memory_total_mb": float(values[2]),
                "temperature_c": float(values[4]),
                "pstate": values[7],
            }
            with contextlib.suppress(ValueError):
                sample["power_w"] = float(values[3])
            with contextlib.suppress(ValueError):
                sample["sm_clock_mhz"] = float(values[5])
            with contextlib.suppress(ValueError):
                sample["memory_clock_mhz"] = float(values[6])
            samples.append(sample)
        except Exception as exc:
            samples.append(
                {
                    "t": time.time(),
                    "sampler_error": f"{type(exc).__name__}: {exc}",
                }
            )
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass
    return samples


def summarize_gpu(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"samples": 0, "sampler_errors": []}
    valid = [sample for sample in samples if "sampler_error" not in sample]
    errors = [str(sample["sampler_error"]) for sample in samples if "sampler_error" in sample]
    util = [float(sample["gpu_util_pct"]) for sample in valid if "gpu_util_pct" in sample]
    mem = [float(sample["memory_used_mb"]) for sample in valid if "memory_used_mb" in sample]
    power = [float(sample["power_w"]) for sample in valid if "power_w" in sample]
    temperature = [
        float(sample["temperature_c"])
        for sample in valid
        if "temperature_c" in sample
    ]
    sm_clock = [
        float(sample["sm_clock_mhz"])
        for sample in valid
        if "sm_clock_mhz" in sample
    ]
    memory_clock = [
        float(sample["memory_clock_mhz"])
        for sample in valid
        if "memory_clock_mhz" in sample
    ]
    return {
        "samples": len(valid),
        "sampler_errors": errors,
        "gpu_util_pct": summarize_values(util),
        "memory_used_mb": summarize_values(mem),
        "power_w": summarize_values(power),
        "temperature_c": summarize_values(temperature),
        "sm_clock_mhz": summarize_values(sm_clock),
        "memory_clock_mhz": summarize_values(memory_clock),
        "pstate_counts": {
            pstate: sum(1 for sample in valid if sample.get("pstate") == pstate)
            for pstate in sorted(
                {str(sample["pstate"]) for sample in valid if "pstate" in sample}
            )
        },
        "last": valid[-1] if valid else None,
    }


def summarize_results(
    args: argparse.Namespace,
    started: float,
    finished: float,
    results: list[JobResult],
    gpu_samples: list[dict[str, Any]],
    barrier_released_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    accepted = [result for result in results if result.accepted]
    succeeded = [result for result in accepted if result.final_status == "succeeded"]
    failed = [result for result in accepted if result.final_status not in {"succeeded"}]
    rejected = [result for result in results if not result.accepted]
    job_latencies = [result.job_latency_s for result in succeeded if result.job_latency_s is not None]
    submit_latencies = [result.submit_latency_s for result in results]
    server_job_latencies = [
        result.server_job_latency_s
        for result in succeeded
        if result.server_job_latency_s is not None
    ]
    server_queue_latencies = [
        result.server_queue_latency_s
        for result in succeeded
        if result.server_queue_latency_s is not None
    ]
    server_service_latencies = [
        result.server_service_latency_s
        for result in succeeded
        if result.server_service_latency_s is not None
    ]
    total_wall_s = finished - started
    total_chunks = sum(result.chunks_completed for result in succeeded)
    prompt_tokens = sum(result.prompt_tokens for result in succeeded)
    completion_tokens = sum(result.completion_tokens for result in succeeded)
    engine_time_s = sum(result.engine_time_s for result in succeeded)
    submit_monotonic_ns = [
        result.submit_started_monotonic_ns
        for result in results
        if result.submit_started_monotonic_ns is not None
    ]
    submit_wall_time_ns = [
        result.submit_started_wall_time_ns
        for result in results
        if result.submit_started_wall_time_ns is not None
    ]
    target_monotonic_ns = int(getattr(args, "start_at_monotonic_ns", 0) or 0)
    server_created_at = [
        result.created_at for result in succeeded if result.created_at is not None
    ]
    server_updated_at = [
        result.updated_at for result in succeeded if result.updated_at is not None
    ]
    server_window_s = (
        max(server_updated_at) - min(server_created_at)
        if server_created_at and server_updated_at
        else None
    )

    return {
        "label": args.label,
        "config": {
            "jobs": args.jobs,
            "chunks_per_job": args.chunks_per_job,
            "text_profile": args.text_profile,
            "unique_text": args.unique_text,
            "run_marker": getattr(args, "run_marker", ""),
            "ref_profile": args.ref_profile,
            "ref_key": args.ref_key,
            "ref_file": args.ref_file or None,
            "many_ref_count": args.many_ref_count if args.ref_profile == "many-url" else None,
            "audio_format": args.audio_format,
            "af_filter": args.af_filter or None,
            "api_base_url": args.api_base_url,
            "submit_concurrency": args.submit_concurrency,
            "poll_concurrency": args.poll_concurrency,
            "poll_interval": args.poll_interval,
            "job_timeout": args.job_timeout,
            "http_timeout": args.http_timeout,
            "gpu_poll_interval": args.gpu_poll_interval,
            "start_at_monotonic_ns": target_monotonic_ns,
            "seed": args.seed,
            "sampling": {
                "speed": args.speed,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
            },
        },
        "timing": {
            "started_wall_time": time.time() - total_wall_s,
            "total_wall_s": total_wall_s,
            "server_window_s": server_window_s,
            "submission": {
                "target_monotonic_ns": target_monotonic_ns or None,
                "barrier_released_monotonic_ns": barrier_released_monotonic_ns,
                "barrier_lateness_ms": (
                    (barrier_released_monotonic_ns - target_monotonic_ns)
                    / 1_000_000
                    if target_monotonic_ns and barrier_released_monotonic_ns is not None
                    else None
                ),
                "first_submit_monotonic_ns": (
                    min(submit_monotonic_ns) if submit_monotonic_ns else None
                ),
                "last_submit_monotonic_ns": (
                    max(submit_monotonic_ns) if submit_monotonic_ns else None
                ),
                "submit_fanout_ms": (
                    (max(submit_monotonic_ns) - min(submit_monotonic_ns))
                    / 1_000_000
                    if submit_monotonic_ns
                    else None
                ),
                "first_submit_offset_from_target_ms": (
                    (min(submit_monotonic_ns) - target_monotonic_ns) / 1_000_000
                    if target_monotonic_ns and submit_monotonic_ns
                    else None
                ),
                "first_submit_wall_time_ns": (
                    min(submit_wall_time_ns) if submit_wall_time_ns else None
                ),
                "last_submit_wall_time_ns": (
                    max(submit_wall_time_ns) if submit_wall_time_ns else None
                ),
            },
        },
        "counts": {
            "submitted": len(results),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "succeeded": len(succeeded),
            "failed_or_timeout": len(failed),
            "chunks_completed": total_chunks,
            "chunks_failed": sum(result.chunks_failed or 0 for result in accepted),
            "chunks_degraded": sum(
                result.chunks_degraded or 0 for result in accepted
            ),
        },
        "rates": {
            "jobs_per_s": len(succeeded) / total_wall_s if total_wall_s > 0 else None,
            "chunks_per_s": total_chunks / total_wall_s if total_wall_s > 0 else None,
            "estimated_audio_s_per_s": (completion_tokens / 25.0) / total_wall_s if total_wall_s > 0 else None,
            "completion_tokens_per_s": completion_tokens / total_wall_s if total_wall_s > 0 else None,
            "server_completion_tokens_per_s": (
                completion_tokens / server_window_s
                if server_window_s is not None and server_window_s > 0
                else None
            ),
        },
        "latency_s": {
            "job": summarize_values([float(value) for value in job_latencies]),
            "submit": summarize_values(submit_latencies),
            "server_job": summarize_values(
                [float(value) for value in server_job_latencies]
            ),
            "server_queue": summarize_values(
                [float(value) for value in server_queue_latencies]
            ),
            "server_service": summarize_values(
                [float(value) for value in server_service_latencies]
            ),
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "engine_time_s": engine_time_s,
            "avg_engine_time_per_chunk_s": engine_time_s / total_chunks if total_chunks else None,
            "estimated_audio_s": completion_tokens / 25.0,
        },
        "cache": {
            "cache_hits": sum(1 for result in succeeded if result.cache_hit is True),
            "cache_misses": sum(1 for result in succeeded if result.cache_hit is False),
        },
        "gpu": summarize_gpu(gpu_samples),
        "errors": [asdict(result) for result in rejected[:10] + failed[:10]],
        "jobs_sample": [asdict(result) for result in results[: min(20, len(results))]],
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    env = parse_env(Path(args.env_file))
    token = args.token or os.getenv("API_TOKEN") or env.get("API_TOKEN")
    if not token:
        raise RuntimeError("API token is required. Pass --token or set API_TOKEN/.env.")

    base_url = args.api_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"}
    refs = ref_pool(args)

    timeout = httpx.Timeout(connect=10, read=args.http_timeout, write=30, pool=args.http_timeout)
    limits = httpx.Limits(max_connections=max(args.submit_concurrency + args.poll_concurrency + 20, 100), max_keepalive_connections=50)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        health_before = await get_json(client, f"{base_url}/health", headers)
        submit_sem = asyncio.Semaphore(args.submit_concurrency)

        async def bounded_submit(index: int) -> JobResult:
            async with submit_sem:
                return await submit_one(client, args, index, refs, base_url, headers)

        barrier_released_monotonic_ns = await wait_until_monotonic_ns(
            getattr(args, "start_at_monotonic_ns", 0)
        )
        stop_gpu = asyncio.Event()
        gpu_task = asyncio.create_task(gpu_sampler(stop_gpu, args.gpu_poll_interval))
        started = time.perf_counter()
        submitted = await asyncio.gather(*(bounded_submit(index) for index in range(args.jobs)))

        poll_sem = asyncio.Semaphore(args.poll_concurrency)

        async def bounded_poll(result: JobResult) -> JobResult:
            async with poll_sem:
                return await poll_one(client, result, base_url, headers, args.poll_interval, args.job_timeout)

        results = await asyncio.gather(*(bounded_poll(result) for result in submitted))
        finished = time.perf_counter()
        stop_gpu.set()
        gpu_samples = await gpu_task
        health_after = await get_json(client, f"{base_url}/health", headers)

    summary = summarize_results(
        args,
        started,
        finished,
        results,
        gpu_samples,
        barrier_released_monotonic_ns,
    )
    summary["health_before"] = health_before
    summary["health_after"] = health_after
    summary["ref_pool"] = refs[: min(10, len(refs))]
    summary["text_chunks"] = chunks_for_job(
        args.text_profile,
        0,
        args.unique_text,
        args.chunks_per_job,
        getattr(args, "run_marker", ""),
    )
    summary["jobs"] = [asdict(result) for result in results]
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--token", default="")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--jobs", type=int, default=50)
    parser.add_argument(
        "--chunks-per-job",
        type=parse_chunks_per_job,
        default=DEFAULT_CHUNKS_PER_JOB,
        metavar=f"1-{MAX_CHUNKS_PER_JOB}",
        help=(
            "Number of chunks in each submitted job. Values 1-4 exercise the "
            "API's short-job lane with the bundled text profiles (default: 10)."
        ),
    )
    parser.add_argument("--text-profile", choices=["short-en", "long-en"], default="long-en")
    parser.add_argument("--unique-text", action="store_true")
    parser.add_argument(
        "--run-marker",
        type=parse_run_marker,
        default="",
        metavar=f"MARKER{RUN_MARKER_LENGTH}",
        help=(
            f"Optional exactly-{RUN_MARKER_LENGTH}-character marker appended "
            "to every chunk to avoid cross-run prompt-cache reuse."
        ),
    )
    parser.add_argument("--ref-profile", choices=["hot", "two", "many-url"], default="hot")
    parser.add_argument("--ref-key", choices=sorted(REFS), default="d6905d7a")
    parser.add_argument(
        "--ref-file",
        default="",
        help="Reference filename under --ref-dir; overrides --ref-key.",
    )
    parser.add_argument(
        "--ref-text",
        default="",
        help="Transcript for --ref-file.",
    )
    parser.add_argument("--many-ref-count", type=int, default=32)
    parser.add_argument("--ref-dir", default="generated")
    parser.add_argument("--ref-port", type=int, default=0)
    parser.add_argument("--audio-format", choices=["wav", "mp3"], default="mp3")
    parser.add_argument(
        "--af-filter",
        default="",
        help="Optional ffmpeg audio-filter chain forwarded to the TTS bridge.",
    )
    parser.add_argument("--submit-concurrency", type=int, default=200)
    parser.add_argument("--poll-concurrency", type=int, default=200)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--job-timeout", type=float, default=900)
    parser.add_argument("--http-timeout", type=float, default=900)
    parser.add_argument("--gpu-poll-interval", type=float, default=0.5)
    parser.add_argument(
        "--start-at-monotonic-ns",
        type=parse_nonnegative_int,
        default=0,
        help=(
            "Absolute time.monotonic_ns() timestamp at which submissions may "
            "start; zero starts immediately."
        ),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--output", default="")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if bool(args.ref_file) != bool(args.ref_text):
        raise ValueError("--ref-file and --ref-text must be supplied together.")

    ref_dir = Path(args.ref_dir)
    if not ref_dir.exists():
        raise FileNotFoundError(f"Missing ref dir: {ref_dir}")
    ref_pool(args)

    with ref_http_server(ref_dir, args.ref_port) as port:
        args.ref_base_url = f"http://127.0.0.1:{port}"
        summary = asyncio.run(run(args))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    printable = {
        "label": summary["label"],
        "counts": summary["counts"],
        "rates": summary["rates"],
        "job_latency_s": summary["latency_s"]["job"],
        "usage": summary["usage"],
        "gpu": summary["gpu"],
        "output": args.output,
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    expected_chunks = args.jobs * args.chunks_per_job
    if (
        summary["counts"]["succeeded"] != args.jobs
        or summary["counts"]["chunks_completed"] != expected_chunks
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
