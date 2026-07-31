#!/usr/bin/env python3
"""Run a fixed, sequential Higgs quality gate directly against test SGLang."""

from __future__ import annotations

import argparse
import array
import hashlib
import io
import json
import math
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any

import httpx


BASE_URL = "http://127.0.0.1:8000"
DEFAULT_REF = Path("/root/autodl-tmp/bench-assets/ref_341c9392.mp3")
EXPECTED_REF_SHA256 = "0e58877044af78294f2164c7641e0bd1e2ef7e1510a360aa35c6f4f96a0aedf4"
REF_TEXT = (
    "The people who are crazy enough to think they can change the world "
    "are the ones who do."
)
STATE_DIR = Path("/root/autodl-tmp/higgs-test-control")

PROMPTS = [
    {
        "id": "en-long-01",
        "text": (
            "Do Wild Animals Understand When Humans Help Them? One night in 2013, "
            "off the coast of Hawaii, a group of divers was preparing to end their "
            "dive. The water was dark. Everything seemed normal. Until one of them "
            "noticed a dark shape racing toward the group at an incredible speed."
        ),
    },
    {
        "id": "en-long-02",
        "text": (
            "At first, they thought it was a shark. But as the creature got closer, "
            "they realized they were wrong. It was a dolphin. A wild dolphin. The "
            "strange thing was that it did not seem afraid at all. It swam directly "
            "toward the humans. No detours. No fleeing. No change of direction."
        ),
    },
    {
        "id": "en-long-03",
        "text": (
            "It kept approaching until it was only a few meters from one of the "
            "divers. Then it did something that left everyone stunned. The dolphin "
            "rolled onto its side. As if it was trying to show them something. When "
            "the divers shined their lights on its body, they immediately understood "
            "what was happening."
        ),
    },
    {
        "id": "en-long-04",
        "text": (
            "A large fishing hook was embedded deep in its fin. A long fishing line "
            "was tightly wrapped around its body. Every movement caused the line to "
            "cut in even deeper. The animal was seriously injured. But what made this "
            "incident famous around the world was not the fishing hook. It was what "
            "happened next."
        ),
    },
    {
        "id": "en-long-05",
        "text": (
            "Because throughout the entire rescue, the dolphin seemed to know exactly "
            "that the divers were trying to help it. And that raised a question that "
            "scientists have debated for years: Do wild animals truly understand when "
            "humans help them? Or is it all just a coincidence that we are imagining? "
            "The answer may change the way you see the animal world forever."
        ),
    },
    {"id": "en-short-01", "text": "The harbor lights turned on before sunrise."},
    {"id": "en-short-02", "text": "Every operator checked the same short list."},
    {"id": "en-short-03", "text": "A quiet signal moved through the control room."},
    {"id": "en-short-04", "text": "The next request arrived without warning."},
    {
        "id": "vi-baseline",
        "text": "Xin chào, đây là bài kiểm tra điều khiển giọng nói bằng tiếng Việt.",
    },
    {
        "id": "vi-long-pause",
        "text": (
            "Xin chào <|prosody:long_pause|> đây là bài kiểm tra điều khiển "
            "giọng nói bằng tiếng Việt."
        ),
    },
    {
        "id": "vi-whisper",
        "text": "<|style:whispering|>Nói nhỏ thôi, đây là một bí mật cần được giữ kín.",
    },
    {
        "id": "vi-laughter",
        "text": (
            "<|emotion:amusement|><|sfx:laughter|>Haha, chuyện này thật bất ngờ "
            "nhưng cũng rất vui."
        ),
    },
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields_after_comm = raw.rsplit(")", 1)[1].split()
    return int(fields_after_comm[19])


def require_quality_process(arm: str, expected_ras: int) -> dict[str, Any]:
    pid_fields = (STATE_DIR / "sglang.pid").read_text(encoding="utf-8").split()
    if len(pid_fields) != 4:
        raise RuntimeError("Invalid SGLang PID state")
    pid, start_ticks, pgid = (int(value) for value in pid_fields[:3])
    launch_token = pid_fields[3]
    if len(launch_token) != 64 or any(char not in "0123456789abcdef" for char in launch_token):
        raise RuntimeError("Invalid SGLang launch token")
    if process_start_ticks(pid) != start_ticks or os.getpgid(pid) != pgid or pid != pgid:
        raise RuntimeError("SGLang process identity does not match controller state")
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "sgl-omni serve" not in cmdline or "--host 127.0.0.1" not in cmdline:
        raise RuntimeError("Unexpected SGLang command line")
    environ = {
        item.decode("utf-8", errors="strict")
        for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        if item
    }
    expected_env = f"HIGGS_RAS_WIN_LEN={expected_ras}"
    if expected_env not in environ:
        raise RuntimeError(f"SGLang environment is missing {expected_env}")
    if f"HIGGS_TEST_LAUNCH_TOKEN={launch_token}" not in environ:
        raise RuntimeError("SGLang environment does not match its launch token")
    active_arm = (STATE_DIR / "arm").read_text(encoding="utf-8").strip()
    if active_arm != arm:
        raise RuntimeError(f"Active arm is {active_arm!r}, expected {arm!r}")
    return {
        "pid": pid,
        "start_ticks": start_ticks,
        "pgid": pgid,
        "arm": active_arm,
        "ras_win_len": expected_ras,
        "cmdline_sha256": sha256_bytes(cmdline.encode()),
    }


def full_token_cap(text: str) -> int:
    # Matches the bridge's Latin/Vietnamese path: ~3.2 codec tokens per char,
    # 2x safety, +96 base, clamped to [256, 2048].
    return max(256, min(2048, int(len(text) * 3.2 * 2.0) + 96))


def inspect_wav(audio: bytes) -> tuple[dict[str, Any], bytes]:
    with wave.open(io.BytesIO(audio), "rb") as reader:
        metadata = {
            "channels": reader.getnchannels(),
            "sample_width": reader.getsampwidth(),
            "sample_rate": reader.getframerate(),
            "frame_count": reader.getnframes(),
            "compression": reader.getcomptype(),
        }
        frames = reader.readframes(reader.getnframes())
    if metadata != {
        "channels": 1,
        "sample_width": 2,
        "sample_rate": 24000,
        "frame_count": metadata["frame_count"],
        "compression": "NONE",
    }:
        raise ValueError(f"Unexpected WAV format: {metadata}")
    if not frames:
        raise ValueError("Empty decoded PCM")
    samples = array.array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    peak = max(max(samples), -min(samples))
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    metadata.update(
        {
            "duration_s": metadata["frame_count"] / metadata["sample_rate"],
            "pcm_sha256": sha256_bytes(frames),
            "peak_dbfs": (
                20.0 * math.log10(peak / 32768.0)
                if peak > 0
                else float("-inf")
            ),
            "rms_dbfs": (
                20.0 * math.log10(rms / 32768.0)
                if rms > 0
                else float("-inf")
            ),
        }
    )
    return metadata, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--mode", choices=["prefix", "full", "production"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ref", type=Path, default=DEFAULT_REF)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    expected_ras = 7 if args.mode == "production" else 0
    process = require_quality_process(args.arm, expected_ras)
    if not args.ref.is_file() or sha256_file(args.ref) != EXPECTED_REF_SHA256:
        raise ValueError("Reference file is missing or has the wrong content hash")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    top_k = 50 if args.mode == "production" else 1
    results: list[dict[str, Any]] = []
    started = time.time()
    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(f"{BASE_URL}/health")
        health.raise_for_status()
        for prompt in PROMPTS:
            for repeat in range(args.repeats):
                max_new_tokens = (
                    12
                    if args.mode == "prefix"
                    else full_token_cap(str(prompt["text"]))
                )
                payload = {
                    "input": prompt["text"],
                    "response_format": "wav",
                    "references": [{"audio_path": str(args.ref), "text": REF_TEXT}],
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.8,
                    "top_k": top_k,
                }
                request_started = time.perf_counter()
                response = client.post(f"{BASE_URL}/v1/audio/speech", json=payload)
                elapsed = time.perf_counter() - request_started
                response.raise_for_status()
                wav_metadata, _ = inspect_wav(response.content)
                output_name = f"{prompt['id']}--r{repeat:02d}.wav"
                (args.output_dir / output_name).write_bytes(response.content)
                results.append(
                    {
                        "prompt_id": prompt["id"],
                        "prompt_sha256": sha256_bytes(str(prompt["text"]).encode()),
                        "repeat": repeat,
                        "file": output_name,
                        "wav_sha256": sha256_bytes(response.content),
                        "wav_bytes": len(response.content),
                        "wav": wav_metadata,
                        "elapsed_s": elapsed,
                        "usage": {
                            "prompt_tokens": int(
                                response.headers.get("x-prompt-tokens", "0")
                            ),
                            "completion_tokens": int(
                                response.headers.get("x-completion-tokens", "0")
                            ),
                            "engine_time_s": float(
                                response.headers.get("x-engine-time-s", "0")
                            ),
                        },
                        "request": {
                            "max_new_tokens": max_new_tokens,
                            "temperature": 0.8,
                            "top_k": top_k,
                        },
                    }
                )
                print(
                    f"{prompt['id']} r{repeat:02d}: "
                    f"{wav_metadata['duration_s']:.3f}s "
                    f"{elapsed:.3f}s wall",
                    flush=True,
                )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(str(row["prompt_id"]), []).append(row)
    repeat_exact = {
        prompt_id: len({row["wav"]["pcm_sha256"] for row in rows}) == 1
        for prompt_id, rows in grouped.items()
    }
    summary = {
        "schema_version": 1,
        "arm": args.arm,
        "mode": args.mode,
        "created_at": time.time(),
        "wall_s": time.time() - started,
        "process": process,
        "reference": {
            "path": str(args.ref),
            "sha256": EXPECTED_REF_SHA256,
            "text": REF_TEXT,
        },
        "settings": {
            "repeats": args.repeats,
            "temperature": 0.8,
            "top_k": top_k,
            "ras_win_len": expected_ras,
        },
        "repeat_exact_by_prompt": repeat_exact,
        "all_repeat_exact": all(repeat_exact.values()),
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), "repeat_exact": repeat_exact}))
    # Greedy semantic-prefix output must be exact. Full synthesis can still
    # differ at the vocoder level on the BF16 baseline, so full/production
    # modes are judged structurally and by paired duration/token ratios.
    if args.mode == "prefix" and not summary["all_repeat_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
