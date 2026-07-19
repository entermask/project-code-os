#!/usr/bin/env python3
"""Submit the exact task fixture directly to an isolated OmniVoice bridge.

The token is read from an environment variable or the canary's .env file and is
never printed. The output protocol is the bridge's 4-byte-count + repeated
4-byte-length-prefixed audio payload, not multipart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_ENV = Path("/root/autodl-tmp/Fish-Audio/.env")
DEFAULT_FIXTURE = Path(
    os.environ.get(
        "PREFILL_FIXTURE",
        "/root/autodl-tmp/fixture-task-7b66bc83/fixture.json",
    )
)
TERMINAL_STATUSES = {"succeeded", "failed"}
CANARY_BASE_URL = "http://127.0.0.1:6006"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", choices=[CANARY_BASE_URL], default=CANARY_BASE_URL)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--output-dir", type=Path, default=HERE / "runs" / "baseline")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--cues",
        default="all",
        help="all, a range such as 47-55, or a comma list such as 10,24,47-55",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--max-wait-seconds", type=float, default=1200.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dotenv_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    prefix = f"{key}="
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        if not line.startswith(prefix):
            continue
        value = line[len(prefix):].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value
    return None


def resolve_token(env_file: Path) -> str:
    token = os.environ.get("OMNIVOICE_CANARY_TOKEN") or dotenv_value(env_file, "API_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing canary token: set OMNIVOICE_CANARY_TOKEN or pass --env-file containing API_TOKEN"
        )
    return token


def parse_srt(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\r?\n\r?\n", text.strip())
    cues: list[dict[str, Any]] = []
    timing = re.compile(
        r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s+-->\s+"
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})$"
    )

    def seconds(parts: tuple[str, ...]) -> float:
        hours, minutes, secs, millis = (int(value) for value in parts)
        return hours * 3600 + minutes * 60 + secs + millis / 1000

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            raise ValueError(f"Invalid SRT block: {block!r}")
        number = int(lines[0].strip())
        match = timing.match(lines[1].strip())
        if not match:
            raise ValueError(f"Invalid SRT timing for cue {number}: {lines[1]!r}")
        start = seconds(match.groups()[:4])
        end = seconds(match.groups()[4:])
        cue_text = "\n".join(lines[2:]).strip()
        cues.append({"number": number, "start": start, "end": end, "text": cue_text})
    return cues


def parse_cue_selection(value: str, count: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(1, count + 1))
    selected: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            selected.update(range(int(left), int(right) + 1))
        else:
            selected.add(int(item))
    if not selected or min(selected) < 1 or max(selected) > count:
        raise ValueError(f"Cue selection must stay within 1-{count}")
    return sorted(selected)


def http_request(
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> tuple[int, dict[str, str], bytes]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {token}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers.items()), error.read()


def require_json(
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
    expected: set[int] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    status, _, body = http_request(method, url, token, payload, timeout)
    allowed = expected or {200}
    if status not in allowed:
        bounded = body.decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"{method} {url} returned HTTP {status}: {bounded}")
    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{method} {url} returned invalid JSON") from error


def parse_length_prefixed_audio(body: bytes) -> list[bytes]:
    if len(body) < 4:
        raise ValueError("Audio response is shorter than its chunk-count prefix")
    count = struct.unpack(">I", body[:4])[0]
    offset = 4
    chunks: list[bytes] = []
    for index in range(count):
        if offset + 4 > len(body):
            raise ValueError(f"Missing length prefix for audio chunk {index + 1}")
        length = struct.unpack(">I", body[offset:offset + 4])[0]
        offset += 4
        end = offset + length
        if end > len(body):
            raise ValueError(f"Truncated audio chunk {index + 1}: need {length} bytes")
        chunks.append(body[offset:end])
        offset = end
    if offset != len(body):
        raise ValueError(f"Audio response has {len(body) - offset} trailing bytes")
    return chunks


def run_batch(
    *,
    base_url: str,
    token: str,
    payload: dict[str, Any],
    cue_numbers: list[int],
    output_dir: Path,
    poll_seconds: float,
    max_wait_seconds: float,
) -> dict[str, Any]:
    created = require_json(
        "POST",
        f"{base_url}/v1/tts",
        token,
        payload,
        expected={202},
    )
    request_id = str(created.get("request_id") or "")
    if not request_id:
        raise RuntimeError("Bridge create response is missing request_id")
    print(f"submitted {request_id}: cues {cue_numbers[0]}-{cue_numbers[-1]}", flush=True)

    deadline = time.monotonic() + max_wait_seconds
    last_progress: tuple[str, int] | None = None
    final_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        status_payload = require_json(
            "GET",
            f"{base_url}/v1/tts/jobs/{request_id}",
            token,
            timeout=30.0,
        )
        status = str(status_payload.get("status") or "unknown")
        completed = int(status_payload.get("chunks_completed") or 0)
        progress = (status, completed)
        if progress != last_progress:
            print(
                f"{request_id}: {status} {completed}/{status_payload.get('chunks_total')}",
                flush=True,
            )
            last_progress = progress
        if status in TERMINAL_STATUSES:
            final_status = status_payload
            break
        time.sleep(poll_seconds)

    if final_status is None:
        raise TimeoutError(f"Bridge job {request_id} exceeded {max_wait_seconds}s")
    if final_status.get("status") != "succeeded":
        raise RuntimeError(f"Bridge job {request_id} failed: {final_status.get('detail', 'unknown')}")

    query = urllib.parse.urlencode({"from": 0, "chunks": len(cue_numbers)})
    status, headers, body = http_request(
        "GET",
        f"{base_url}/v1/tts/jobs/{request_id}/audio?{query}",
        token,
        timeout=180.0,
    )
    if status != 200:
        bounded = body.decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Audio download for {request_id} returned HTTP {status}: {bounded}")
    buffers = parse_length_prefixed_audio(body)
    if len(buffers) != len(cue_numbers):
        raise RuntimeError(
            f"Bridge returned {len(buffers)} audio chunks, expected {len(cue_numbers)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for cue_number, buffer in zip(cue_numbers, buffers, strict=True):
        path = output_dir / f"cue-{cue_number:02d}-bridge.wav"
        path.write_bytes(buffer)
        files.append(
            {
                "cue": cue_number,
                "file": path.name,
                "bytes": len(buffer),
                "sha256": hashlib.sha256(buffer).hexdigest(),
            }
        )

    return {
        "requestId": request_id,
        "cues": cue_numbers,
        "status": final_status,
        "responseHeaders": {
            key: value
            for key, value in headers.items()
            if key.lower().startswith("x-") and key.lower() != "x-transcript"
        },
        "files": files,
    }


def main() -> int:
    args = parse_args()
    fixture_path = args.fixture.resolve()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture_dir = fixture_path.parent
    srt_path = fixture_dir / fixture["srt"]["file"]
    reference_path = fixture_dir / fixture["voice"]["reference"]["localFile"]

    if sha256(srt_path) != fixture["srt"]["sha256"]:
        raise RuntimeError("SRT fixture SHA-256 mismatch")
    if sha256(reference_path) != fixture["voice"]["reference"]["sha256"]:
        raise RuntimeError("Reference audio fixture SHA-256 mismatch")

    cues = parse_srt(srt_path)
    if len(cues) != fixture["srt"]["cueCount"]:
        raise RuntimeError(f"Parsed {len(cues)} SRT cues, expected {fixture['srt']['cueCount']}")
    selected_numbers = parse_cue_selection(args.cues, len(cues))
    selected = [cues[number - 1] for number in selected_numbers]
    token = resolve_token(args.env_file)
    base_url = args.base_url.rstrip("/")
    output_dir = args.output_dir.resolve()
    request_config = fixture["bridgeRequest"]
    batch_size = args.batch_size or int(request_config["batchSize"])
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    common_payload = dict(request_config["payload"])
    common_payload.update(
        {
            "ref_audio_url": fixture["voice"]["reference"]["audioUrl"],
            "ref_text": fixture["voice"]["reference"]["transcript"],
        }
    )
    started_at = time.time()
    jobs: list[dict[str, Any]] = []
    for offset in range(0, len(selected), batch_size):
        batch = selected[offset:offset + batch_size]
        payload = dict(common_payload)
        payload["chunks"] = [f"{cue['text'].strip()}," for cue in batch]
        jobs.append(
            run_batch(
                base_url=base_url,
                token=token,
                payload=payload,
                cue_numbers=[int(cue["number"]) for cue in batch],
                output_dir=output_dir,
                poll_seconds=args.poll_seconds,
                max_wait_seconds=args.max_wait_seconds,
            )
        )

    run_manifest = {
        "fixtureTaskId": fixture["task"]["id"],
        "voiceId": fixture["voice"]["voiceId"],
        "referenceAudioSha256": fixture["voice"]["reference"]["sha256"],
        "srtSha256": fixture["srt"]["sha256"],
        "baseUrl": base_url,
        "selection": selected_numbers,
        "batchSize": batch_size,
        "startedAtUnix": started_at,
        "finishedAtUnix": time.time(),
        "jobs": jobs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run-manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(selected_numbers)} raw WAV files and {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
