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

    @property
    def job_latency_s(self) -> float | None:
        if self.finished_s is None:
            return None
        return self.finished_s - self.submit_started_s


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
    source = ref_dir / base_file
    if not source.exists():
        raise FileNotFoundError(f"Missing reference audio: {source}")
    target_dir = ref_dir / f"bench_refs_{Path(base_file).stem}_{count}"
    target_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for index in range(count):
        target = target_dir / f"ref_{index:04d}{source.suffix}"
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)
        files.append(str(target.relative_to(ref_dir)))
    return files


def ref_pool(args: argparse.Namespace) -> list[dict[str, str]]:
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

    refs: list[dict[str, str]] = []
    for key, filename in zip(keys, files):
        refs.append(
            {
                "key": key,
                "file": filename,
                "text": REFS[key]["text"],
            }
        )
    return refs


def chunks_for_job(profile: str, job_index: int, unique_text: bool) -> list[str]:
    if profile == "long-en":
        chunks = list(DEFAULT_LONG_CHUNKS)
    elif profile == "short-en":
        chunks = list(DEFAULT_SHORT_CHUNKS)
    else:
        raise ValueError(f"Unknown text profile: {profile}")

    if not unique_text:
        return chunks

    marker = f" Batch marker {job_index:04d}."
    return [f"{chunk}{marker}" for chunk in chunks]


async def get_json(client: httpx.AsyncClient, url: str, headers: dict[str, str]) -> Any:
    response = await client.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


async def submit_one(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    index: int,
    refs: list[dict[str, str]],
    base_url: str,
    headers: dict[str, str],
) -> JobResult:
    ref = refs[index % len(refs)]
    chunks = chunks_for_job(args.text_profile, index, args.unique_text)
    payload: dict[str, Any] = {
        "chunks": chunks,
        "ref_audio_url": f"{args.ref_base_url}/{ref['file']}",
        "ref_text": ref["text"],
        "format": args.audio_format,
    }
    if args.seed is not None:
        payload["seed"] = args.seed + index
    for field in ("speed", "max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty"):
        value = getattr(args, field)
        if value is not None:
            payload[field] = value

    started = time.perf_counter()
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
            detail=response.text[:500],
        )

    body = response.json()
    return JobResult(
        index=index,
        accepted=True,
        status_code=response.status_code,
        submit_latency_s=now - started,
        submit_started_s=started,
        accepted_s=now,
        request_id=body.get("request_id"),
        status_url=body.get("status_url"),
        final_status=body.get("status"),
        chunks_completed=int(body.get("chunks_completed") or 0),
    )


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
            status = body.get("status")
            result.final_status = status
            result.chunks_completed = int(body.get("chunks_completed") or 0)
            result.cache_hit = body.get("cache_hit")
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
    query = "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
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
            }
            with contextlib.suppress(ValueError):
                sample["power_w"] = float(values[3])
            samples.append(sample)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass
    return samples


def summarize_gpu(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"samples": 0}
    util = [float(sample["gpu_util_pct"]) for sample in samples if "gpu_util_pct" in sample]
    mem = [float(sample["memory_used_mb"]) for sample in samples if "memory_used_mb" in sample]
    power = [float(sample["power_w"]) for sample in samples if "power_w" in sample]
    return {
        "samples": len(samples),
        "gpu_util_pct": summarize_values(util),
        "memory_used_mb": summarize_values(mem),
        "power_w": summarize_values(power),
        "last": samples[-1],
    }


def summarize_results(args: argparse.Namespace, started: float, finished: float, results: list[JobResult], gpu_samples: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [result for result in results if result.accepted]
    succeeded = [result for result in accepted if result.final_status == "succeeded"]
    failed = [result for result in accepted if result.final_status not in {"succeeded"}]
    rejected = [result for result in results if not result.accepted]
    job_latencies = [result.job_latency_s for result in succeeded if result.job_latency_s is not None]
    submit_latencies = [result.submit_latency_s for result in results]
    total_wall_s = finished - started
    total_chunks = sum(result.chunks_completed for result in succeeded)
    prompt_tokens = sum(result.prompt_tokens for result in succeeded)
    completion_tokens = sum(result.completion_tokens for result in succeeded)
    engine_time_s = sum(result.engine_time_s for result in succeeded)

    return {
        "label": args.label,
        "config": {
            "jobs": args.jobs,
            "chunks_per_job": args.chunks_per_job,
            "text_profile": args.text_profile,
            "unique_text": args.unique_text,
            "ref_profile": args.ref_profile,
            "ref_key": args.ref_key,
            "many_ref_count": args.many_ref_count if args.ref_profile == "many-url" else None,
            "audio_format": args.audio_format,
            "api_base_url": args.api_base_url,
        },
        "timing": {
            "started_wall_time": time.time() - total_wall_s,
            "total_wall_s": total_wall_s,
        },
        "counts": {
            "submitted": len(results),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "succeeded": len(succeeded),
            "failed_or_timeout": len(failed),
            "chunks_completed": total_chunks,
        },
        "rates": {
            "jobs_per_s": len(succeeded) / total_wall_s if total_wall_s > 0 else None,
            "chunks_per_s": total_chunks / total_wall_s if total_wall_s > 0 else None,
            "estimated_audio_s_per_s": (completion_tokens / 25.0) / total_wall_s if total_wall_s > 0 else None,
            "completion_tokens_per_s": completion_tokens / total_wall_s if total_wall_s > 0 else None,
        },
        "latency_s": {
            "job": summarize_values([float(value) for value in job_latencies]),
            "submit": summarize_values(submit_latencies),
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

    summary = summarize_results(args, started, finished, results, gpu_samples)
    summary["health_before"] = health_before
    summary["health_after"] = health_after
    summary["ref_pool"] = refs[: min(10, len(refs))]
    summary["text_chunks"] = chunks_for_job(args.text_profile, 0, args.unique_text)
    summary["jobs"] = [asdict(result) for result in results]
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--token", default="")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--jobs", type=int, default=50)
    parser.add_argument("--chunks-per-job", type=int, default=10)
    parser.add_argument("--text-profile", choices=["short-en", "long-en"], default="long-en")
    parser.add_argument("--unique-text", action="store_true")
    parser.add_argument("--ref-profile", choices=["hot", "two", "many-url"], default="hot")
    parser.add_argument("--ref-key", choices=sorted(REFS), default="d6905d7a")
    parser.add_argument("--many-ref-count", type=int, default=32)
    parser.add_argument("--ref-dir", default="generated")
    parser.add_argument("--ref-port", type=int, default=0)
    parser.add_argument("--audio-format", choices=["wav", "mp3"], default="mp3")
    parser.add_argument("--submit-concurrency", type=int, default=200)
    parser.add_argument("--poll-concurrency", type=int, default=200)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--job-timeout", type=float, default=900)
    parser.add_argument("--http-timeout", type=float, default=900)
    parser.add_argument("--gpu-poll-interval", type=float, default=0.5)
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
    if args.chunks_per_job != 10:
        raise ValueError("This benchmark currently defines exactly 10 chunks per job.")

    ref_dir = Path(args.ref_dir)
    if not ref_dir.exists():
        raise FileNotFoundError(f"Missing ref dir: {ref_dir}")

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


if __name__ == "__main__":
    main()
