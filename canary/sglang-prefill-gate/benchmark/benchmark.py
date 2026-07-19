#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import threading
import time
import urllib.request
import wave
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


HERE = Path(__file__).resolve().parent
OUT = Path(os.environ["PREFILL_BENCH_OUT"])
FIXTURE_DIR = Path("/root/autodl-tmp/fixture-task-7b66bc83")
RUN_DIRECT = HERE / "run-direct.py"
BASE_URL = "http://127.0.0.1:6006"
ENV_FILE = Path("/root/autodl-tmp/Fish-Audio/.env")
JOB_ROOT = Path("/root/autodl-tmp/tts-cache/jobs")
POLL_SECONDS = float(os.environ.get("PREFILL_POLL_SECONDS", "0.05"))
WARMUP_RUNS = int(os.environ.get("PREFILL_WARMUP_RUNS", "1"))
MEASURED_RUNS = int(os.environ.get("PREFILL_MEASURED_RUNS", "10"))
PARALLEL_JOBS = int(os.environ.get("PREFILL_PARALLEL_JOBS", "1"))
BENCH_LABEL = os.environ["PREFILL_BENCH_LABEL"]
COALESCE_REQUESTS = int(os.environ["PREFILL_COALESCE_REQUESTS"])
COALESCE_WAIT_MS = float(os.environ.get("PREFILL_COALESCE_WAIT_MS", "60"))
SGLANG_URL = "http://127.0.0.1:8000"
API_LOG = Path("/root/autodl-tmp/logs/higgs_api.log")
SGLANG_LOG = Path("/root/autodl-tmp/logs/higgs_sglang.log")
BRIDGE_SOURCE = Path("/root/autodl-tmp/Fish-Audio/app.py")
MAX_SUBMIT_SKEW_S = float(os.environ.get("PREFILL_MAX_SUBMIT_SKEW_S", "0.05"))
DIRTY_SENTINEL = Path("/root/autodl-tmp/prod-sim/prefill-benchmark-dirty.json")
RECOVERY_TIMEOUT_S = 1200.0

EXPECTED_TASK_ID = "7b66bc83-e5d5-48be-af5a-306a72d26bd4"
EXPECTED_VOICE_ID = "fcxDguohxleZaemvsuHB"
EXPECTED_CUE_COUNT = 69
EXPECTED_FIXTURE_SHA256 = "217ec831dd9b934cb70b3ce95bbe95b7336c0a1a353df01d3e754fd64025d051"
EXPECTED_SRT_SHA256 = "3e3532076ce72ee843a1fa6d36726a0eac65b928c11a8e22aaf9a38d34bd4b18"
EXPECTED_REFERENCE_SHA256 = "f5f702503b19eb0980467c8e9e395a7503a12f918bcf43cba1993fbe7fc08c80"
EXPECTED_SGLANG_HEAD = "df62e91a00d383e6f73ab9604386ffac6c520529"
EXPECTED_SGLANG_DIFF_SHA256 = "304eb276c6d3f19acbfa1bb32723f9c533b6d88be40a5f536cb83cf1ed9d097a"
EXPECTED_SOURCE_ROOT = "/root/autodl-tmp/sglang-omni-prefill-gate-canary"
EXPECTED_BRIDGE_APP_SHA256 = "00f37e06ac27d62f03d68686f1991d526db4884f7eec42b49398640885b44a37"
EXPECTED_BRIDGE_CACHE_DIR = "/root/autodl-tmp/tts-cache"
EXPECTED_SGLANG_STATUS_SHA256 = "abfbe6c8cdb655cfad3c9604dac169c3526bbd8b6b717f5cbd4337e3c12ac55c"
EXPECTED_RUNTIME_FILE_SHA256 = {
    "sglang_omni/scheduling/omni_scheduler.py": "ce0707c75ef193184b65af705192160ee25c694c13122bfd7e7733e5c026d6f3",
    "sglang_omni/models/higgs_tts/stages.py": "d5b8ae1fe4b7a34baacd294779a8a9fa0711a58fe27bc25b725a67fa513a73b9",
}
EXPECTED_BRIDGE_HEALTH = {
    "status": "ok",
    "sglang_ready": True,
    "sglang_base_url": "http://127.0.0.1:8000",
    "max_concurrent_chunks": 96,
    "short_reserved_chunks": 4,
    "long_concurrent_chunks": 92,
    "max_in_flight_chunks_per_job": 10,
    "max_burst_in_flight_chunks_per_job": 20,
    "max_burst_active_jobs": 2,
    "busy_backlog_chunks": 2000,
}
SUPPORTED_REF_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

if (
    WARMUP_RUNS < 1
    or MEASURED_RUNS < 1
    or PARALLEL_JOBS not in {1, 2}
    or MAX_SUBMIT_SKEW_S <= 0
):
    raise ValueError(
        "warmup/measured runs must be >= 1, parallel jobs must be 1 or 2, "
        "and max submit skew must be positive"
    )


def load_runner():
    spec = importlib.util.spec_from_file_location("fixture_runner", RUN_DIRECT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {RUN_DIRECT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()
request_journal_lock = threading.Lock()
submitted_requests: dict[str, str] = {}


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def proc_rss_kb(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
        match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", text, re.MULTILINE)
        return int(match.group(1)) if match else None
    except (FileNotFoundError, PermissionError):
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_api_pid() -> int:
    output = subprocess.check_output(
        ["pgrep", "-f", "/uvicorn app:app --host 0.0.0.0 --port 6006"],
        text=True,
    )
    pids = [int(line) for line in output.splitlines() if line.strip()]
    if len(pids) != 1:
        raise RuntimeError(f"Expected exactly one API PID, got {pids}")
    return pids[0]


def find_sglang_pid() -> int:
    output = subprocess.check_output(["pgrep", "-f", "sgl-omni serve"], text=True)
    pids = [int(line) for line in output.splitlines() if line.strip()]
    if len(pids) != 1:
        raise RuntimeError(f"Expected exactly one SGLang PID, got {pids}")
    return pids[0]


def process_environ(pid: int) -> dict[str, str]:
    environ: dict[str, str] = {}
    for raw in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        if b"=" in raw:
            key, value = raw.split(b"=", 1)
            environ[key.decode("utf-8")] = value.decode("utf-8")
    return environ


def process_start_unix(pid: int) -> float:
    stat_text = Path(f"/proc/{pid}/stat").read_text()
    fields_after_comm = stat_text[stat_text.rfind(")") + 2 :].split()
    start_ticks = int(fields_after_comm[19])
    boot_time = next(
        int(line.split()[1])
        for line in Path("/proc/stat").read_text().splitlines()
        if line.startswith("btime ")
    )
    return boot_time + start_ticks / os.sysconf("SC_CLK_TCK")


def bridge_runtime_snapshot(pid: int) -> dict[str, Any]:
    environ = process_environ(pid)
    cache_dir = environ.get("TTS_CACHE_DIR")
    if cache_dir != EXPECTED_BRIDGE_CACHE_DIR:
        raise RuntimeError(f"Unexpected bridge TTS_CACHE_DIR: {cache_dir!r}")
    cmdline = [
        value.decode("utf-8")
        for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if value
    ]
    cwd = os.readlink(f"/proc/{pid}/cwd")
    if Path(cwd) != BRIDGE_SOURCE.parent:
        raise RuntimeError(
            f"Bridge cwd {cwd} does not load the pinned source {BRIDGE_SOURCE}"
        )
    started_unix = process_start_unix(pid)
    source_mtime_unix = BRIDGE_SOURCE.stat().st_mtime
    if source_mtime_unix > started_unix:
        raise RuntimeError(
            "Pinned bridge source changed after the active API process started: "
            f"source_mtime={source_mtime_unix} process_start={started_unix}"
        )
    return {
        "pid": pid,
        "cmdline": cmdline,
        "cmdline_sha256": hashlib.sha256(
            b"\0".join(value.encode() for value in cmdline)
        ).hexdigest(),
        "cwd": cwd,
        "tts_cache_dir": cache_dir,
        "process_started_unix": started_unix,
        "source_mtime_unix": source_mtime_unix,
    }


def bridge_source_snapshot() -> dict[str, Any]:
    app_sha256 = file_sha256(BRIDGE_SOURCE)
    if app_sha256 != EXPECTED_BRIDGE_APP_SHA256:
        raise RuntimeError(
            f"Unexpected bridge app fingerprint: {app_sha256} at {BRIDGE_SOURCE}"
        )
    return {
        "path": str(BRIDGE_SOURCE),
        "bytes": BRIDGE_SOURCE.stat().st_size,
        "sha256": app_sha256,
    }


def fetch_bridge_health() -> dict[str, Any]:
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=30) as response:
        return json.loads(response.read())


def bridge_is_idle(health: dict[str, Any]) -> bool:
    tts_jobs = health.get("tts_jobs") or {}
    return (
        health.get("active_tts_jobs") == 0
        and health.get("outstanding_chunks") == 0
        and tts_jobs.get("queued") == 0
        and tts_jobs.get("running") == 0
        and not any((health.get("lane_inflight") or {}).values())
        and not any((health.get("lane_waiting") or {}).values())
    )


def bridge_health_snapshot(label: str) -> dict[str, Any]:
    health = fetch_bridge_health()

    mismatches = {
        key: {"actual": health.get(key), "expected": expected}
        for key, expected in EXPECTED_BRIDGE_HEALTH.items()
        if health.get(key) != expected
    }
    tts_jobs = health.get("tts_jobs") or {}
    sglang_status = health.get("sglang_status") or {}
    expected_sglang_status = {
        "status": "healthy",
        "running": True,
        "pending_completions": 0,
        "request_states": {},
    }
    mismatches.update(
        {
            f"sglang_status.{key}": {
                "actual": sglang_status.get(key),
                "expected": expected,
            }
            for key, expected in expected_sglang_status.items()
            if sglang_status.get(key) != expected
        }
    )
    busy = {
        "active_tts_jobs": health.get("active_tts_jobs"),
        "outstanding_chunks": health.get("outstanding_chunks"),
        "tts_jobs_queued": tts_jobs.get("queued"),
        "tts_jobs_running": tts_jobs.get("running"),
        "lane_inflight": health.get("lane_inflight"),
        "lane_waiting": health.get("lane_waiting"),
    }
    idle = bridge_is_idle(health)
    if mismatches or not idle:
        raise RuntimeError(
            f"Bridge health guard failed at {label}: "
            f"config={mismatches or 'ok'} idle={idle} state={busy}"
        )
    return health


def journal_request(event: str, request_id: str, **details: Any) -> None:
    record = {
        "event": event,
        "request_id": request_id,
        "unix": time.time(),
        **details,
    }
    with request_journal_lock:
        if event == "submitted":
            submitted_requests[request_id] = str(details.get("label") or "")
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "request-journal.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def wait_for_bridge_quiescence(token: str) -> dict[str, Any]:
    with request_journal_lock:
        pending = set(submitted_requests)
    last_status: dict[str, Any] = {}
    last_health: dict[str, Any] | None = None
    last_error: str | None = None
    deadline = time.monotonic() + RECOVERY_TIMEOUT_S
    while time.monotonic() < deadline:
        for request_id in list(pending):
            try:
                status_code, _, body = runner.http_request(
                    "GET",
                    f"{BASE_URL}/v1/tts/jobs/{request_id}",
                    token,
                    timeout=30,
                )
                if status_code == 404:
                    last_status[request_id] = {"status": "expired"}
                    pending.remove(request_id)
                    journal_request("recovered-terminal", request_id, status="expired")
                    continue
                if status_code != 200:
                    last_error = f"request {request_id}: HTTP {status_code}"
                    continue
                status = json.loads(body)
                last_status[request_id] = {
                    "status": status.get("status"),
                    "chunks_completed": status.get("chunks_completed"),
                    "chunks_failed": status.get("chunks_failed"),
                }
                if status.get("status") in {"succeeded", "failed"}:
                    pending.remove(request_id)
                    journal_request(
                        "recovered-terminal",
                        request_id,
                        **last_status[request_id],
                    )
            except Exception as exc:
                last_error = f"request {request_id}: {exc!r}"
        try:
            last_health = fetch_bridge_health()
            if not pending and bridge_is_idle(last_health):
                strict_health = bridge_health_snapshot("cleanup")
                return {
                    "clean": True,
                    "requests": last_status,
                    "health": strict_health,
                    "last_error": last_error,
                }
        except Exception as exc:
            last_error = f"health: {exc!r}"
        time.sleep(0.25)

    dirty = {
        "created_at_unix": time.time(),
        "benchmark_label": BENCH_LABEL,
        "pending_request_ids": sorted(pending),
        "last_status": last_status,
        "last_health": last_health,
        "last_error": last_error,
    }
    DIRTY_SENTINEL.write_text(json.dumps(dirty, indent=2) + "\n", encoding="utf-8")
    raise RuntimeError(
        f"Bridge did not become quiescent; dirty sentinel written to {DIRTY_SENTINEL}: {dirty}"
    )


def reference_cache_snapshot(
    cache_dir: Path, ref_audio_url: str, require_exists: bool
) -> dict[str, Any]:
    ref_audio_url = ref_audio_url.strip()
    suffix = Path(urlparse(ref_audio_url).path).suffix.lower()
    if suffix not in SUPPORTED_REF_AUDIO_SUFFIXES:
        suffix = ".audio"
    cache_name = f"{hashlib.sha256(ref_audio_url.encode('utf-8')).hexdigest()}{suffix}"
    cache_path = cache_dir / "ref-audio" / cache_name
    exists = cache_path.exists() and cache_path.stat().st_size > 0
    snapshot: dict[str, Any] = {
        "path": str(cache_path),
        "exists": exists,
    }
    if exists:
        snapshot.update(
            {
                "bytes": cache_path.stat().st_size,
                "sha256": file_sha256(cache_path),
            }
        )
        if snapshot["sha256"] != EXPECTED_REFERENCE_SHA256:
            raise RuntimeError(
                f"Bridge cached reference SHA mismatch: {snapshot['sha256']} at {cache_path}"
            )
    elif require_exists:
        raise RuntimeError(f"Bridge reference cache was not populated: {cache_path}")
    return snapshot


def active_sglang_snapshot() -> dict[str, Any]:
    pid = find_sglang_pid()
    cmdline = [
        value.decode("utf-8")
        for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if value
    ]
    environ = process_environ(pid)
    expected_prefix = [
        "/root/autodl-tmp/sglang-omni/.venv/bin/python3",
        "/root/autodl-tmp/sglang-omni/.venv/bin/sgl-omni",
        "serve",
    ]
    if cmdline[:3] != expected_prefix or (len(cmdline) - 3) % 2:
        raise RuntimeError(f"Unexpected SGLang command prefix/shape: {cmdline}")
    options: dict[str, str] = {}
    for index in range(3, len(cmdline), 2):
        name, value = cmdline[index : index + 2]
        if not name.startswith("--") or name in options:
            raise RuntimeError(f"Invalid or duplicate SGLang option {name}: {cmdline}")
        options[name] = value
    expected_options = {
        "--model-path": "bosonai/higgs-audio-v3-tts-4b",
        "--host": "127.0.0.1",
        "--port": "8000",
        "--allowed-local-media-path": "/root/autodl-tmp/tts-cache",
        "--stages.2.factory_args.server_args_overrides.attention_backend": "triton",
        "--stages.2.factory_args.prefill_coalesce_requests": str(COALESCE_REQUESTS),
        "--stages.2.factory_args.prefill_coalesce_wait_ms": f"{COALESCE_WAIT_MS:g}",
    }
    if options != expected_options:
        raise RuntimeError(
            f"Active SGLang CLI is not the exact prod-sim canary config: {options}"
        )
    active_k = int(options["--stages.2.factory_args.prefill_coalesce_requests"])
    active_wait_ms = float(
        options["--stages.2.factory_args.prefill_coalesce_wait_ms"]
    )
    source_root = environ.get("PYTHONPATH", "").split(":", 1)[0]
    head = subprocess.check_output(
        ["git", "-C", source_root, "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(
        ["git", "-C", source_root, "diff", "HEAD", "--no-ext-diff"]
    )
    diff_sha256 = hashlib.sha256(diff).hexdigest()
    status = subprocess.check_output(
        [
            "git",
            "-C",
            source_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ]
    )
    status_sha256 = hashlib.sha256(status).hexdigest()
    if status_sha256 != EXPECTED_SGLANG_STATUS_SHA256:
        raise RuntimeError(f"Unexpected SGLang worktree status: {status.decode()}")
    process_started_unix = process_start_unix(pid)
    runtime_files: dict[str, dict[str, Any]] = {}
    for relative_path, expected_sha256 in EXPECTED_RUNTIME_FILE_SHA256.items():
        path = Path(source_root) / relative_path
        actual_sha256 = file_sha256(path)
        modified_unix = path.stat().st_mtime
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Unexpected runtime source SHA for {relative_path}: {actual_sha256}"
            )
        if modified_unix > process_started_unix:
            raise RuntimeError(
                f"Runtime source {relative_path} changed after SGLang started"
            )
        runtime_files[relative_path] = {
            "sha256": actual_sha256,
            "mtime_unix": modified_unix,
        }
    changed_paths = subprocess.check_output(
        ["git", "-C", source_root, "diff", "HEAD", "--name-only"], text=True
    ).splitlines()
    runtime_mtime_after_start = [
        relative_path
        for relative_path in changed_paths
        if not relative_path.startswith("tests/")
        and (Path(source_root) / relative_path).stat().st_mtime > process_started_unix
    ]
    if runtime_mtime_after_start:
        raise RuntimeError(
            "SGLang runtime files changed after process start: "
            f"{runtime_mtime_after_start}"
        )
    snapshot = {
        "pid": pid,
        "cmdline": cmdline,
        "cmdline_sha256": hashlib.sha256(b"\0".join(value.encode() for value in cmdline)).hexdigest(),
        "pythonpath": environ.get("PYTHONPATH"),
        "source_root": source_root,
        "head": head,
        "diff_sha256": diff_sha256,
        "status_sha256": status_sha256,
        "process_started_unix": process_started_unix,
        "runtime_files": runtime_files,
        "coalesce_requests": active_k,
        "coalesce_wait_ms": active_wait_ms,
    }
    if active_k != COALESCE_REQUESTS or active_wait_ms != COALESCE_WAIT_MS:
        raise RuntimeError(
            "Benchmark K/T does not match active SGLang: "
            f"benchmark={COALESCE_REQUESTS}/{COALESCE_WAIT_MS}, "
            f"active={active_k}/{active_wait_ms}"
        )
    if source_root != EXPECTED_SOURCE_ROOT:
        raise RuntimeError(f"Unexpected active SGLang source: {source_root}")
    if head != EXPECTED_SGLANG_HEAD or diff_sha256 != EXPECTED_SGLANG_DIFF_SHA256:
        raise RuntimeError(
            f"Unexpected SGLang fingerprint: head={head} diff={diff_sha256}"
        )
    return snapshot


def profile_control(action: str, run_id: str, event_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"run_id": run_id}
    if action == "start":
        payload["event_dir"] = str(event_dir)
    request = urllib.request.Request(
        f"{SGLANG_URL}/{action}_request_profile",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read())
    if result.get("run_id") != run_id:
        raise RuntimeError(f"Unexpected profiler {action} response: {result}")
    return result


def wait_for_profile_flush(event_dir: Path) -> list[Path]:
    previous: tuple[tuple[str, int], ...] | None = None
    stable = 0
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        files = sorted(event_dir.glob("*.jsonl"))
        current = tuple((path.name, path.stat().st_size) for path in files)
        if files and current == previous:
            stable += 1
            if stable >= 3:
                return files
        else:
            stable = 0
        previous = current
        time.sleep(0.25)
    raise RuntimeError(f"Profiler files did not stabilize in {event_dir}")


def download_sha256(url: str) -> str:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log_checkpoint(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size


def read_log_delta(path: Path, checkpoint: tuple[int, int, int]) -> str:
    device, inode, offset = checkpoint
    stat = path.stat()
    if (stat.st_dev, stat.st_ino) != (device, inode) or stat.st_size < offset:
        raise RuntimeError(f"Log rotated during benchmark: {path}")
    with path.open("rb") as handle:
        handle.seek(offset)
        return handle.read().decode("utf-8", errors="replace")


def sample_resources(api_pid: int, stop: threading.Event, samples: list[dict[str, Any]]) -> None:
    while not stop.is_set():
        sample: dict[str, Any] = {"unix": time.time(), "api_pid": api_pid}
        rss = proc_rss_kb(api_pid)
        if rss is not None:
            sample["api_rss_kb"] = rss
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            values = [part.strip() for part in output.strip().splitlines()[0].split(",")]
            sample.update(
                {
                    "gpu_util_pct": float(values[0]),
                    "gpu_memory_used_mib": float(values[1]),
                    "gpu_memory_total_mib": float(values[2]),
                    "gpu_power_w": float(values[3]),
                }
            )
        except Exception as exc:
            sample["gpu_error"] = repr(exc)
        samples.append(sample)
        stop.wait(0.10)


def wav_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        compression = handle.getcomptype()
    return {
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate_hz": rate,
        "frames": frames,
        "compression": compression,
        "duration_s": frames / rate,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_payload() -> tuple[dict[str, Any], list[int], dict[str, Any], dict[str, str]]:
    fixture_path = FIXTURE_DIR / "fixture.json"
    manifest_sha256 = runner.sha256(fixture_path)
    if manifest_sha256 != EXPECTED_FIXTURE_SHA256:
        raise RuntimeError(f"Unexpected fixture manifest SHA: {manifest_sha256}")
    fixture = json.loads(fixture_path.read_text())
    if fixture["task"]["id"] != EXPECTED_TASK_ID:
        raise RuntimeError(f"Unexpected task fixture: {fixture['task']['id']}")
    if fixture["voice"]["voiceId"] != EXPECTED_VOICE_ID:
        raise RuntimeError(f"Unexpected voice fixture: {fixture['voice']['voiceId']}")
    if fixture["srt"]["cueCount"] != EXPECTED_CUE_COUNT:
        raise RuntimeError(f"Unexpected cue count: {fixture['srt']['cueCount']}")
    if fixture["srt"]["sha256"] != EXPECTED_SRT_SHA256:
        raise RuntimeError(f"Unexpected SRT manifest SHA: {fixture['srt']['sha256']}")
    if fixture["voice"]["reference"]["sha256"] != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError(
            f"Unexpected reference manifest SHA: {fixture['voice']['reference']['sha256']}"
        )
    srt_path = FIXTURE_DIR / fixture["srt"]["file"]
    ref_path = FIXTURE_DIR / fixture["voice"]["reference"]["localFile"]
    if runner.sha256(srt_path) != EXPECTED_SRT_SHA256:
        raise RuntimeError("SRT fixture SHA mismatch")
    if runner.sha256(ref_path) != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("Reference fixture SHA mismatch")
    reference_url_sha256 = download_sha256(fixture["voice"]["reference"]["audioUrl"])
    if reference_url_sha256 != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError(f"Reference URL bytes changed: {reference_url_sha256}")
    cues = runner.parse_srt(srt_path)
    if len(cues) != EXPECTED_CUE_COUNT:
        raise RuntimeError(f"Parsed {len(cues)} cues, expected {EXPECTED_CUE_COUNT}")
    payload = dict(fixture["bridgeRequest"]["payload"])
    payload.update(
        {
            "ref_audio_url": fixture["voice"]["reference"]["audioUrl"],
            "ref_text": fixture["voice"]["reference"]["transcript"],
            "chunks": [f"{cue['text'].strip()}," for cue in cues],
        }
    )
    if "seed" in payload:
        raise RuntimeError("Benchmark payload must not claim/use a seed")
    verification = {
        "fixture_manifest_sha256": manifest_sha256,
        "reference_url_sha256": reference_url_sha256,
    }
    return payload, [int(cue["number"]) for cue in cues], fixture, verification


def run_one(label: str, payload: dict[str, Any], cue_numbers: list[int], token: str) -> dict[str, Any]:
    output_dir = OUT / label
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_started_perf = time.perf_counter()
    runner_started_unix = time.time()
    client_submit_started_unix = time.time()
    created = runner.require_json("POST", f"{BASE_URL}/v1/tts", token, payload, expected={202})
    request_id = str(created.get("request_id") or "")
    if not request_id:
        raise RuntimeError("Bridge create response is missing request_id")
    server_created_unix = float(created["created_at"])
    response_received_unix = time.time()
    journal_request(
        "submitted",
        request_id,
        label=label,
        client_submit_started_unix=client_submit_started_unix,
        server_created_unix=server_created_unix,
        response_received_unix=response_received_unix,
    )
    final: dict[str, Any] | None = None
    poll_count = 0
    poll_errors: list[str] = []
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        poll_count += 1
        try:
            status = runner.require_json(
                "GET", f"{BASE_URL}/v1/tts/jobs/{request_id}", token, timeout=30
            )
        except Exception as exc:
            poll_errors.append(repr(exc))
            time.sleep(POLL_SECONDS)
            continue
        if status.get("status") in {"succeeded", "failed"}:
            final = status
            journal_request(
                "terminal",
                request_id,
                label=label,
                status=status.get("status"),
                chunks_completed=status.get("chunks_completed"),
                chunks_failed=status.get("chunks_failed"),
            )
            break
        time.sleep(POLL_SECONDS)
    if final is None:
        raise TimeoutError(request_id)

    query = f"from=0&chunks={len(cue_numbers)}"
    status_code, response_headers, body = runner.http_request(
        "GET", f"{BASE_URL}/v1/tts/jobs/{request_id}/audio?{query}", token, timeout=180
    )
    if status_code != 200:
        raise RuntimeError(f"audio HTTP {status_code}: {body[:500]!r}")
    buffers = runner.parse_length_prefixed_audio(body)
    if len(buffers) != len(cue_numbers):
        raise RuntimeError(f"Expected {len(cue_numbers)} buffers, got {len(buffers)}")
    wavs: list[dict[str, Any]] = []
    for cue_number, buffer in zip(cue_numbers, buffers, strict=True):
        path = output_dir / f"cue-{cue_number:02d}-bridge.wav"
        path.write_bytes(buffer)
        info = wav_info(path)
        info.update({"cue": cue_number, "file": path.name})
        wavs.append(info)
    runner_finished_unix = time.time()
    runner_wall_s = time.perf_counter() - runner_started_perf

    job_started = float(final.get("started_at") or final["created_at"])
    job_updated = float(final["updated_at"])
    job_dir = JOB_ROOT / request_id
    completion_offsets = sorted(
        path.stat().st_mtime - job_started for path in job_dir.glob("chunk_*.wav")
    )
    usage = final.get("usage") or {}
    run_result = {
        "label": label,
        "request_id": request_id,
        "runner_started_unix": runner_started_unix,
        "client_submit_started_unix": client_submit_started_unix,
        "server_created_unix": server_created_unix,
        "response_received_unix": response_received_unix,
        "runner_finished_unix": runner_finished_unix,
        "runner_wall_s": runner_wall_s,
        "job_started_unix": job_started,
        "job_updated_unix": job_updated,
        "status": final.get("status"),
        "chunks_total": int(final.get("chunks_total") or 0),
        "chunks_completed": int(final.get("chunks_completed") or 0),
        "chunks_failed": int(final.get("chunks_failed") or 0),
        "chunks_degraded": int(final.get("chunks_degraded") or 0),
        "audio_cache_hit": final.get("cache_hit"),
        "in_flight_limit": int(final.get("in_flight_limit") or 0),
        "job_makespan_s": job_updated - job_started,
        "chunks_per_s": len(wavs) / (job_updated - job_started),
        "audio_duration_s": sum(float(item["duration_s"]) for item in wavs),
        "audio_s_per_s": sum(float(item["duration_s"]) for item in wavs) / (job_updated - job_started),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "engine_time_s": float(usage.get("engine_time_s") or 0.0),
        "poll_count": poll_count,
        "poll_interval_s": POLL_SECONDS,
        "poll_errors": poll_errors,
        "completion_trace_files": len(completion_offsets),
        "completion_offset_s": distribution(completion_offsets),
        "wav_validation": {
            "files": len(wavs),
            "valid_pcm16_24k_mono": sum(
                1
                for item in wavs
                if item["channels"] == 1
                and item["sample_width_bytes"] == 2
                and item["sample_rate_hz"] == 24000
                and item["compression"] == "NONE"
            ),
            "invalid": [
                item
                for item in wavs
                if not (
                    item["channels"] == 1
                    and item["sample_width_bytes"] == 2
                    and item["sample_rate_hz"] == 24000
                    and item["compression"] == "NONE"
                )
            ],
        },
        "response_headers": {
            key: value
            for key, value in response_headers.items()
            if key.lower().startswith("x-") and key.lower() != "x-transcript"
        },
    }
    (output_dir / "run-result.json").write_text(json.dumps(run_result, indent=2) + "\n")
    print(
        f"{label}: status={run_result['status']} makespan={run_result['job_makespan_s']:.6f}s "
        f"wall={runner_wall_s:.6f}s retries=pending-log-audit failed={run_result['chunks_failed']} "
        f"degraded={run_result['chunks_degraded']}",
        flush=True,
    )
    return run_result


def metric_distribution(runs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return distribution([float(run[key]) for run in runs])


def resource_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"samples": len(samples), "sampling_interval_s": 0.10}
    for key in ("api_rss_kb", "gpu_util_pct", "gpu_memory_used_mib", "gpu_power_w"):
        result[key] = distribution([float(sample[key]) for sample in samples if key in sample])
    return result


def run_wave(
    label: str,
    payload: dict[str, Any],
    cue_numbers: list[int],
    token: str,
) -> dict[str, Any]:
    health_before = bridge_health_snapshot(f"before-{label}")
    barrier = threading.Barrier(PARALLEL_JOBS)

    def worker(index: int) -> dict[str, Any]:
        barrier.wait()
        return run_one(
            f"{label}-job-{index:02d}",
            payload,
            cue_numbers,
            token,
        )

    started_unix = time.time()
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as pool:
        futures = [pool.submit(worker, index) for index in range(1, PARALLEL_JOBS + 1)]
        jobs = [future.result() for future in futures]
    wall_s = time.perf_counter() - started
    finished_unix = time.time()
    service_started = min(float(job["job_started_unix"]) for job in jobs)
    service_finished = max(float(job["job_updated_unix"]) for job in jobs)
    service_makespan_s = service_finished - service_started
    chunks = sum(job["chunks_completed"] for job in jobs)
    audio_duration_s = sum(job["audio_duration_s"] for job in jobs)
    client_submit_times = [float(job["client_submit_started_unix"]) for job in jobs]
    server_create_times = [float(job["server_created_unix"]) for job in jobs]
    client_submit_skew_s = max(client_submit_times) - min(client_submit_times)
    server_create_skew_s = max(server_create_times) - min(server_create_times)
    if PARALLEL_JOBS > 1 and max(client_submit_skew_s, server_create_skew_s) > MAX_SUBMIT_SKEW_S:
        raise RuntimeError(
            "Concurrent submit skew exceeds limit: "
            f"client={client_submit_skew_s:.6f}s server={server_create_skew_s:.6f}s "
            f"limit={MAX_SUBMIT_SKEW_S:.6f}s"
        )
    job_makespans = [float(job["job_makespan_s"]) for job in jobs]
    return {
        "label": label,
        "parallel_jobs": PARALLEL_JOBS,
        "runner_started_unix": started_unix,
        "runner_finished_unix": finished_unix,
        "client_wall_s": wall_s,
        "client_submit_skew_s": client_submit_skew_s,
        "server_create_skew_s": server_create_skew_s,
        "health_before": health_before,
        "service_started_unix": service_started,
        "service_finished_unix": service_finished,
        "service_makespan_s": service_makespan_s,
        "job_makespan_spread_s": max(job_makespans) - min(job_makespans),
        "chunks": chunks,
        "service_chunks_per_s": chunks / service_makespan_s,
        "audio_duration_s": audio_duration_s,
        "service_audio_s_per_s": audio_duration_s / service_makespan_s,
        "jobs": jobs,
    }


def parse_profile_events(
    files: list[Path],
    measured_waves: list[dict[str, Any]],
    measured_guard_started_unix: float,
    measured_guard_finished_unix: float,
) -> dict[str, Any]:
    required = {
        "scheduler_queue_enter",
        "scheduler_prefill_start",
        "stage_complete",
    }
    by_request: dict[str, dict[str, int]] = {}
    duplicate_events: list[str] = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("stage") != "tts_engine" or event.get("event_name") not in required:
                continue
            request_id = str(event["request_id"])
            event_name = str(event["event_name"])
            request_events = by_request.setdefault(request_id, {})
            if event_name in request_events:
                duplicate_events.append(f"{request_id}:{event_name}")
            request_events[event_name] = int(event["timestamp_ns"])
    if duplicate_events:
        raise RuntimeError(f"Duplicate profiler events: {duplicate_events[:10]}")

    expected_per_wave = EXPECTED_CUE_COUNT * PARALLEL_JOBS
    all_queue_wait_ms: list[float] = []
    all_engine_complete_ms: list[float] = []
    all_inclusive_complete_ms: list[float] = []
    wave_reports: list[dict[str, Any]] = []
    measured_ids: set[str] = set()
    for wave in measured_waves:
        start_ns = int(float(wave["runner_started_unix"]) * 1e9)
        finish_ns = int(float(wave["runner_finished_unix"]) * 1e9)
        wave_ids = {
            request_id
            for request_id, events in by_request.items()
            if start_ns <= events.get("scheduler_queue_enter", -1) <= finish_ns
        }
        if len(wave_ids) != expected_per_wave:
            raise RuntimeError(
                f"Profiler request count mismatch for {wave['label']}: "
                f"got {len(wave_ids)}, expected {expected_per_wave}"
            )
        missing = {
            request_id: sorted(required - by_request[request_id].keys())
            for request_id in wave_ids
            if required - by_request[request_id].keys()
        }
        if missing:
            raise RuntimeError(f"Incomplete profiler events for {wave['label']}: {missing}")
        if measured_ids & wave_ids:
            raise RuntimeError(f"Profiler request appeared in multiple waves: {wave['label']}")
        measured_ids.update(wave_ids)

        queue_wait_ms: list[float] = []
        engine_complete_ms: list[float] = []
        inclusive_complete_ms: list[float] = []
        for request_id in wave_ids:
            events = by_request[request_id]
            queue_enter = events["scheduler_queue_enter"]
            prefill_start = events["scheduler_prefill_start"]
            stage_complete = events["stage_complete"]
            if not queue_enter <= prefill_start <= stage_complete:
                raise RuntimeError(f"Invalid profiler event order for {request_id}: {events}")
            queue_wait_ms.append((prefill_start - queue_enter) / 1e6)
            engine_complete_ms.append((stage_complete - prefill_start) / 1e6)
            inclusive_complete_ms.append((stage_complete - queue_enter) / 1e6)
        all_queue_wait_ms.extend(queue_wait_ms)
        all_engine_complete_ms.extend(engine_complete_ms)
        all_inclusive_complete_ms.extend(inclusive_complete_ms)
        wave_reports.append(
            {
                "label": wave["label"],
                "requests": len(wave_ids),
                "queue_wait_ms": distribution(queue_wait_ms),
                "engine_complete_after_prefill_ms": distribution(engine_complete_ms),
                "engine_complete_after_queue_ms": distribution(inclusive_complete_ms),
            }
        )

    expected_total = expected_per_wave * len(measured_waves)
    if len(measured_ids) != expected_total:
        raise RuntimeError(
            f"Measured profiler total mismatch: got {len(measured_ids)}, expected {expected_total}"
        )
    guard_start_ns = int(measured_guard_started_unix * 1e9)
    guard_finish_ns = int(measured_guard_finished_unix * 1e9)
    guard_ids = {
        request_id
        for request_id, events in by_request.items()
        if guard_start_ns <= events.get("scheduler_queue_enter", -1) <= guard_finish_ns
    }
    if guard_ids != measured_ids:
        raise RuntimeError(
            "Profiler exclusivity guard found requests outside measured waves: "
            f"extra={sorted(guard_ids - measured_ids)[:20]} "
            f"missing={sorted(measured_ids - guard_ids)[:20]}"
        )
    return {
        "semantics": {
            "queue_wait_ms": "scheduler_prefill_start - scheduler_queue_enter",
            "engine_complete_after_prefill_ms": "tts_engine stage_complete - scheduler_prefill_start",
            "engine_complete_after_queue_ms": "tts_engine stage_complete - scheduler_queue_enter",
            "first_emit_note": (
                "scheduler_first_emit is intentionally absent on this non-streaming SRT path"
            ),
        },
        "event_files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
        "measured_requests": len(measured_ids),
        "expected_measured_requests": expected_total,
        "guard_window": {
            "started_unix": measured_guard_started_unix,
            "finished_unix": measured_guard_finished_unix,
            "requests": len(guard_ids),
        },
        "queue_wait_ms": distribution(all_queue_wait_ms),
        "engine_complete_after_prefill_ms": distribution(all_engine_complete_ms),
        "engine_complete_after_queue_ms": distribution(all_inclusive_complete_ms),
        "per_wave": wave_reports,
    }


def validate_jobs(jobs: list[dict[str, Any]], require_cache_hit: bool) -> None:
    errors: list[str] = []
    for job in jobs:
        label = str(job["label"])
        expected = {
            "status": "succeeded",
            "chunks_total": EXPECTED_CUE_COUNT,
            "chunks_completed": EXPECTED_CUE_COUNT,
            "chunks_failed": 0,
            "chunks_degraded": 0,
            "in_flight_limit": 20,
            "completion_trace_files": EXPECTED_CUE_COUNT,
        }
        for key, value in expected.items():
            if job.get(key) != value:
                errors.append(f"{label}:{key}={job.get(key)!r}, expected {value!r}")
        if job["poll_errors"]:
            errors.append(f"{label}:poll_errors={job['poll_errors']!r}")
        validation = job["wav_validation"]
        if validation["files"] != EXPECTED_CUE_COUNT:
            errors.append(f"{label}:wav_files={validation['files']}")
        if validation["valid_pcm16_24k_mono"] != EXPECTED_CUE_COUNT:
            errors.append(
                f"{label}:valid_wavs={validation['valid_pcm16_24k_mono']}"
            )
        if validation["invalid"]:
            errors.append(f"{label}:invalid_wavs={validation['invalid']!r}")
        if require_cache_hit and job.get("audio_cache_hit") is not True:
            errors.append(f"{label}:audio_cache_hit={job.get('audio_cache_hit')!r}")
    if errors:
        raise RuntimeError(f"Benchmark job validation failed: {errors[:20]}")


def main() -> None:
    if DIRTY_SENTINEL.exists():
        raise RuntimeError(
            f"Refusing benchmark while dirty sentinel exists: {DIRTY_SENTINEL}"
        )
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to mix benchmark data in non-empty {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    payload, cue_numbers, fixture, fixture_verification = build_payload()
    token = runner.resolve_token(ENV_FILE)
    api_pid_before = find_api_pid()
    bridge_runtime_before = bridge_runtime_snapshot(api_pid_before)
    bridge_cache_dir = Path(bridge_runtime_before["tts_cache_dir"])
    bridge_source_before = bridge_source_snapshot()
    bridge_health_before = bridge_health_snapshot("before-warmup")
    reference_cache_before = reference_cache_snapshot(
        bridge_cache_dir, payload["ref_audio_url"], require_exists=False
    )
    sglang_before = active_sglang_snapshot()
    api_log_checkpoint = log_checkpoint(API_LOG)
    sglang_log_checkpoint = log_checkpoint(SGLANG_LOG)
    profile_run_id = f"{BENCH_LABEL}-events"
    event_dir = OUT / "events"
    profile_start: dict[str, Any] | None = None
    profile_stop: dict[str, Any] | None = None
    profile_start_attempted = False
    resources: list[dict[str, Any]] = []
    stop = threading.Event()
    monitor = threading.Thread(
        target=sample_resources,
        args=(api_pid_before, stop, resources),
        daemon=True,
    )
    monitor_started = False
    warmup_waves: list[dict[str, Any]] = []
    measured_waves: list[dict[str, Any]] = []
    measured_guard_started_unix: float | None = None
    measured_guard_finished_unix: float | None = None
    bridge_health_after_warmup: dict[str, Any] | None = None
    reference_cache_after_warmup: dict[str, Any] | None = None
    traffic_error: BaseException | None = None
    cleanup_errors: list[str] = []
    recovery_report: dict[str, Any] | None = None
    try:
        profile_start_attempted = True
        profile_start = profile_control("start", profile_run_id, event_dir)
        monitor.start()
        monitor_started = True
        warmup_waves = [
            run_wave(f"warmup-{index:02d}", payload, cue_numbers, token)
            for index in range(1, WARMUP_RUNS + 1)
        ]
        measured_guard_started_unix = time.time()
        bridge_health_after_warmup = bridge_health_snapshot("after-warmup")
        reference_cache_after_warmup = reference_cache_snapshot(
            bridge_cache_dir, payload["ref_audio_url"], require_exists=True
        )
        if (
            bridge_health_after_warmup["tts_jobs"]["failed"]
            != bridge_health_before["tts_jobs"]["failed"]
        ):
            raise RuntimeError("Bridge failed-job count increased during warmup")
        measured_waves = [
            run_wave(f"run-{index:02d}", payload, cue_numbers, token)
            for index in range(1, MEASURED_RUNS + 1)
        ]
        measured_guard_finished_unix = time.time()
    except BaseException as exc:
        traffic_error = exc
    finally:
        stop.set()
        if monitor_started:
            monitor.join(timeout=5)
        if profile_start_attempted:
            try:
                profile_stop = profile_control("stop", profile_run_id, event_dir)
            except Exception as exc:
                cleanup_errors.append(f"profile stop failed: {exc!r}")
        try:
            recovery_report = wait_for_bridge_quiescence(token)
        except Exception as exc:
            cleanup_errors.append(f"bridge recovery failed: {exc!r}")

    if traffic_error is not None or cleanup_errors:
        if cleanup_errors and not DIRTY_SENTINEL.exists():
            DIRTY_SENTINEL.write_text(
                json.dumps(
                    {
                        "created_at_unix": time.time(),
                        "benchmark_label": BENCH_LABEL,
                        "cleanup_errors": cleanup_errors,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        failure = {
            "label": BENCH_LABEL,
            "traffic_error": repr(traffic_error) if traffic_error else None,
            "cleanup_errors": cleanup_errors,
            "recovery": recovery_report,
            "dirty_sentinel": str(DIRTY_SENTINEL),
        }
        (OUT / "failure-cleanup.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        if traffic_error is not None:
            if cleanup_errors:
                raise RuntimeError(
                    f"Benchmark failed and cleanup was incomplete: {failure}"
                ) from traffic_error
            raise traffic_error
        raise RuntimeError(f"Benchmark cleanup failed: {failure}")

    if (
        measured_guard_started_unix is None
        or measured_guard_finished_unix is None
        or bridge_health_after_warmup is None
        or reference_cache_after_warmup is None
        or profile_start is None
        or profile_stop is None
        or recovery_report is None
    ):
        raise RuntimeError("Benchmark completed without required phase metadata")

    event_files = wait_for_profile_flush(event_dir)

    warmup = [job for wave in warmup_waves for job in wave["jobs"]]
    measured = [job for wave in measured_waves for job in wave["jobs"]]
    validate_jobs(warmup, require_cache_hit=False)
    validate_jobs(measured, require_cache_hit=True)
    profile_report = parse_profile_events(
        event_files,
        measured_waves,
        measured_guard_started_unix,
        measured_guard_finished_unix,
    )

    api_pid_after = find_api_pid()
    bridge_runtime_after = bridge_runtime_snapshot(api_pid_after)
    bridge_source_after = bridge_source_snapshot()
    bridge_health_after = bridge_health_snapshot("after-measured")
    reference_cache_after = reference_cache_snapshot(
        bridge_cache_dir, payload["ref_audio_url"], require_exists=True
    )
    sglang_after = active_sglang_snapshot()
    service_errors: list[str] = []
    if api_pid_after != api_pid_before:
        service_errors.append(f"API restarted: {api_pid_before} -> {api_pid_after}")
    if bridge_runtime_after != bridge_runtime_before:
        service_errors.append("Bridge process/runtime changed during phase")
    if bridge_source_after != bridge_source_before:
        service_errors.append("Bridge app source changed during phase")
    if sglang_after != sglang_before:
        service_errors.append("SGLang process/config/source changed during phase")
    if (
        bridge_health_after["tts_jobs"]["failed"]
        != bridge_health_before["tts_jobs"]["failed"]
    ):
        service_errors.append("Bridge failed-job count increased during phase")

    wav_files = list(OUT.glob("*/cue-*-bridge.wav"))
    invalid_wav: list[str] = []
    total_wav_duration = 0.0
    for path in wav_files:
        try:
            info = wav_info(path)
            total_wav_duration += float(info["duration_s"])
            if not (
                info["channels"] == 1
                and info["sample_width_bytes"] == 2
                and info["sample_rate_hz"] == 24000
                and info["compression"] == "NONE"
            ):
                invalid_wav.append(str(path))
        except Exception as exc:
            invalid_wav.append(f"{path}: {exc!r}")

    expected_wav_files = (
        (WARMUP_RUNS + MEASURED_RUNS) * PARALLEL_JOBS * EXPECTED_CUE_COUNT
    )
    if len(wav_files) != expected_wav_files:
        service_errors.append(
            f"WAV file count mismatch: {len(wav_files)} != {expected_wav_files}"
        )
    if invalid_wav:
        service_errors.append(f"Invalid WAV files: {invalid_wav[:10]}")

    api_log_window = read_log_delta(API_LOG, api_log_checkpoint)
    sglang_log_window = read_log_delta(SGLANG_LOG, sglang_log_checkpoint)
    (OUT / "api-log-window.log").write_text(api_log_window, encoding="utf-8")
    (OUT / "sglang-log-window.log").write_text(sglang_log_window, encoding="utf-8")
    retry_pattern = re.compile(r"retrying|re-render|attempt\s+[23]/3", re.IGNORECASE)
    error_pattern = re.compile(
        r"\bERROR\b|Traceback|out of memory|\bOOM\b", re.IGNORECASE
    )
    retry_lines = [line for line in api_log_window.splitlines() if retry_pattern.search(line)]
    error_lines = [
        line
        for line in (api_log_window + "\n" + sglang_log_window).splitlines()
        if error_pattern.search(line)
    ]
    if retry_lines:
        service_errors.append(f"Quality retries in API log: {len(retry_lines)}")
    if error_lines:
        service_errors.append(f"Errors in benchmark log window: {len(error_lines)}")

    report = {
        "label": BENCH_LABEL,
        "role": "prefill-admission-canary",
        "prefill_gate": {
            "requests": COALESCE_REQUESTS,
            "wait_ms": COALESCE_WAIT_MS,
        },
        "fixture": {
            "task_id": fixture["task"]["id"],
            "voice_id": fixture["voice"]["voiceId"],
            **fixture_verification,
            "runner_sha256": runner.sha256(RUN_DIRECT),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reference_audio_sha256": fixture["voice"]["reference"]["sha256"],
            "srt_sha256": fixture["srt"]["sha256"],
            "batch_size": len(cue_numbers),
            "format": payload["format"],
            "num_step": payload["num_step"],
            "seed_sent": "seed" in payload,
            "poll_interval_s": POLL_SECONDS,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
            "parallel_jobs": PARALLEL_JOBS,
            "max_submit_skew_s": MAX_SUBMIT_SKEW_S,
        },
        "distribution_measured": {
            key: metric_distribution(measured, key)
            for key in (
                "job_makespan_s",
                "runner_wall_s",
                "chunks_per_s",
                "audio_s_per_s",
                "completion_tokens",
                "audio_duration_s",
                "engine_time_s",
            )
        },
        "completion_offsets_measured": {
            "per_run_p50_s": distribution(
                [float(run["completion_offset_s"]["p50"]) for run in measured]
            ),
            "per_run_p95_s": distribution(
                [float(run["completion_offset_s"]["p95"]) for run in measured]
            ),
        },
        "wave_distribution_measured": {
            key: distribution([float(wave[key]) for wave in measured_waves])
            for key in (
                "client_wall_s",
                "client_submit_skew_s",
                "server_create_skew_s",
                "service_makespan_s",
                "job_makespan_spread_s",
                "service_chunks_per_s",
                "service_audio_s_per_s",
            )
        },
        "counts_measured": {
            "waves": len(measured_waves),
            "jobs": len(measured),
            "succeeded": sum(run["status"] == "succeeded" for run in measured),
            "failed_runs": sum(run["status"] != "succeeded" for run in measured),
            "chunks": sum(run["chunks_completed"] for run in measured),
            "chunks_failed": sum(run["chunks_failed"] for run in measured),
            "chunks_degraded": sum(run["chunks_degraded"] for run in measured),
            "poll_errors": sum(len(run["poll_errors"]) for run in measured),
        },
        "warmup_waves": warmup_waves,
        "measured_waves": measured_waves,
        "warmup": warmup,
        "per_run": measured,
        "request_profile": profile_report,
        "profile_control": {
            "start": profile_start,
            "stop": profile_stop,
        },
        "failure_safety": {
            "request_journal": str(OUT / "request-journal.jsonl"),
            "recovery": recovery_report,
            "dirty_sentinel": str(DIRTY_SENTINEL),
        },
        "wav_validation_all_runs": {
            "expected_files": expected_wav_files,
            "files": len(wav_files),
            "invalid": invalid_wav,
            "total_duration_s": total_wav_duration,
        },
        "log_audit": {
            "retry_matches": len(retry_lines),
            "error_matches": len(error_lines),
            "retry_lines": retry_lines,
            "error_lines": error_lines,
        },
        "service_integrity": {
            "api_pid_before": api_pid_before,
            "api_pid_after": api_pid_after,
            "bridge_runtime_before": bridge_runtime_before,
            "bridge_runtime_after": bridge_runtime_after,
            "bridge_source_before": bridge_source_before,
            "bridge_source_after": bridge_source_after,
            "bridge_health_before": bridge_health_before,
            "bridge_health_after_warmup": bridge_health_after_warmup,
            "bridge_health_after": bridge_health_after,
            "reference_cache_before": reference_cache_before,
            "reference_cache_after_warmup": reference_cache_after_warmup,
            "reference_cache_after": reference_cache_after,
            "sglang_before": sglang_before,
            "sglang_after": sglang_after,
            "errors": service_errors,
        },
        "resources_all_runs": resource_summary(resources),
        "valid": not service_errors,
    }
    resource_path = OUT / "resource-samples.jsonl"
    with resource_path.open("w") as handle:
        for sample in resources:
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
    report_path = OUT / "benchmark-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    (OUT / "benchmark-report.sha256").write_text(
        f"{report_sha256}  {report_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_path), "counts": report["counts_measured"]}, indent=2))
    if service_errors:
        raise RuntimeError(f"Benchmark phase failed integrity checks: {service_errors}")


if __name__ == "__main__":
    main()
