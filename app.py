import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import random
import re
import shutil
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlparse

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sglang-tts-api")


API_TOKEN = os.getenv("API_TOKEN", "")
TTS_BACKEND_NAME = os.getenv("TTS_BACKEND_NAME", os.getenv("MODEL_PATH", "bosonai/higgs-audio-v3-tts-4b"))
SGLANG_BASE_URL = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
SPEECH_MODEL = os.getenv("SPEECH_MODEL", "").strip()
CACHE_DIR = Path(os.getenv("TTS_CACHE_DIR", "/ephemeral/tts-cache"))
REF_AUDIO_DIR = CACHE_DIR / "ref-audio"
TRANSCRIPT_DIR = CACHE_DIR / "transcripts"
JOB_DIR = CACHE_DIR / "jobs"
TMP_DIR = CACHE_DIR / "tmp"

DOWNLOAD_TIMEOUT = float(os.getenv("DOWNLOAD_TIMEOUT", "60"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))
JOB_CLEANUP_INTERVAL_SECONDS = int(os.getenv("JOB_CLEANUP_INTERVAL_SECONDS", "60"))
MAX_CONCURRENT_CHUNKS = max(1, int(os.getenv("MAX_CONCURRENT_CHUNKS", "4")))
SHORT_RESERVED_CHUNKS = max(0, int(os.getenv("SHORT_RESERVED_CHUNKS", "0")))
if SHORT_RESERVED_CHUNKS >= MAX_CONCURRENT_CHUNKS:
    SHORT_RESERVED_CHUNKS = MAX_CONCURRENT_CHUNKS - 1
SHORT_REQUEST_MAX_CHARS = max(1, int(os.getenv("SHORT_REQUEST_MAX_CHARS", "1000")))
SHORT_REQUEST_MAX_CHUNKS = max(1, int(os.getenv("SHORT_REQUEST_MAX_CHUNKS", "4")))
LONG_CONCURRENT_CHUNKS = MAX_CONCURRENT_CHUNKS - SHORT_RESERVED_CHUNKS
MAX_IN_FLIGHT_CHUNKS_PER_JOB = max(1, int(os.getenv("MAX_IN_FLIGHT_CHUNKS_PER_JOB", "12")))
BUSY_BACKLOG_CHUNKS = max(1, int(os.getenv("BUSY_BACKLOG_CHUNKS", "32")))
# Per-chunk retry: lỗi tạm từ SGLang (5xx/CUDA/network) hoặc audio rỗng/quá nhỏ
# sẽ được sinh lại tối đa CHUNK_RETRY_ATTEMPTS lần thay vì giết cả job ngay.
CHUNK_RETRY_ATTEMPTS = max(1, int(os.getenv("CHUNK_RETRY_ATTEMPTS", "3")))
CHUNK_RETRY_BASE_DELAY = float(os.getenv("CHUNK_RETRY_BASE_DELAY", "1.0"))
# Audio nhỏ hơn ngưỡng này coi như chunk hỏng (gần-câm/cụt) → retry. 0 = tắt kiểm tra.
CHUNK_MIN_BYTES = max(0, int(os.getenv("CHUNK_MIN_BYTES", "512")))
# Higgs đôi khi không emit EOS → tuôn tới max_new_tokens ra audio SIZE HỢP LỆ nhưng CÂM
# (byte-size không bắt được). Đo max_volume: nếu < ngưỡng này (dBFS) coi là câm → retry seed khác.
# Speech thật có peak > -20dB nên -50 tách sạch. Đặt <= -90 để TẮT kiểm tra.
CHUNK_SILENCE_MAX_DBFS = float(os.getenv("CHUNK_SILENCE_MAX_DBFS", "-50"))
# Dynamic max_new_tokens theo độ dài + script của chunk (codec Higgs = 25 fps; xem docs higgs_tts).
# Đo thực: latin ~1.4-1.66 tok/char, Hangul ~3.6, CJK ~4.8-5.1. Dùng rate + margin RỘNG để không
# cắt cụt chunk thật, nhưng chặn runaway (default 2048 ≈ 82s) xuống sát nhu cầu (~3-4× với latin).
TOK_PER_CHAR_LATIN = float(os.getenv("TOK_PER_CHAR_LATIN", "2.0"))
TOK_PER_CHAR_DENSE = float(os.getenv("TOK_PER_CHAR_DENSE", "6.0"))   # CJK + Hangul + kana
MAX_NEW_TOKENS_SAFETY = float(os.getenv("MAX_NEW_TOKENS_SAFETY", "1.5"))
MAX_NEW_TOKENS_BASE = int(os.getenv("MAX_NEW_TOKENS_BASE", "96"))
MAX_NEW_TOKENS_FLOOR = max(1, int(os.getenv("MAX_NEW_TOKENS_FLOOR", "256")))
MAX_NEW_TOKENS_CEIL = max(1, int(os.getenv("MAX_NEW_TOKENS_CEIL", "2048")))  # = model default, không vượt
# Sampling mặc định cho higgs. Worker không gửi → sgl-omni mặc định temp=1.0, top_p/top_k TẮT =
# phân bố khuếch tán → dễ kẹt "silence attractor" (không sample được EOC) → runaway câm. Theo ref
# boson voice-clone (temp 0.8 / top_k 50 / top_p 0.95). top_k>=50 để EOC không bị mask khỏi top-k.
HIGGS_TEMPERATURE = float(os.getenv("HIGGS_TEMPERATURE", "0.8"))
HIGGS_TOP_P = float(os.getenv("HIGGS_TOP_P", "0.95"))
HIGGS_TOP_K = int(os.getenv("HIGGS_TOP_K", "50"))
# Natural/multi-turn: ground each chunk on the last N seconds of audio already
# produced in this job (audio only, NO text — empirically cleaner). The window
# spans chunk boundaries, so a tiny prior chunk ("OK.") auto-merges with the one
# before it to fill the window.
MT_CONTEXT_TAIL_SEC = float(os.getenv("MT_CONTEXT_TAIL_SEC", "6"))
_DENSE_CHAR_RE = re.compile("[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")  # CJK/kana/Hangul/fullwidth
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")


def _estimate_max_new_tokens(text: str) -> int:
    """Cap động theo độ dài/script chunk: đủ dư cho audio thật, chặn runaway sát nhu cầu thực."""
    dense = len(_DENSE_CHAR_RE.findall(text))
    other = max(0, len(text) - dense)
    est = dense * TOK_PER_CHAR_DENSE + other * TOK_PER_CHAR_LATIN
    cap = int(est * MAX_NEW_TOKENS_SAFETY) + MAX_NEW_TOKENS_BASE
    return max(MAX_NEW_TOKENS_FLOOR, min(MAX_NEW_TOKENS_CEIL, cap))

SUPPORTED_FORMATS = {"wav", "mp3"}
SUPPORTED_REF_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
OPTIONAL_SGLANG_FIELDS = (
    "speed",
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "seed",
)


app = FastAPI(title="SGLang TTS API", version="0.2.0")

chunk_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)
short_chunk_semaphore = asyncio.Semaphore(SHORT_RESERVED_CHUNKS) if SHORT_RESERVED_CHUNKS else None
long_chunk_semaphore = asyncio.Semaphore(LONG_CONCURRENT_CHUNKS)
cache_locks: dict[str, asyncio.Lock] = {}
cache_locks_guard = asyncio.Lock()
jobs: dict[str, "TTSJob"] = {}
jobs_lock = asyncio.Lock()
outstanding_chunks = 0
outstanding_chunks_lock = asyncio.Lock()
cleanup_task: Optional[asyncio.Task] = None


class TTSRequest(BaseModel):
    chunks: list[str]
    ref_audio_url: str
    ref_text: str
    format: str = "mp3"
    speed: Optional[float] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    # Natural/multi-turn: generate chunks sequentially, grounding each on the
    # last MT_CONTEXT_TAIL_SEC of audio already produced this job (audio-only).
    # Consistent voice across the job; slower per-job (no chunk parallelism);
    # requires the multi-turn sgl-omni build.
    multi_turn: bool = False
    # Tag LẬT GIỌNG (emotion/style/pitch/expressive) kéo giọng của chain trôi/đổi giới
    # tính. Khi True: neo MỌI chunk vào 1 mỏ neo TRUNG TÍNH cố định (chunk0 đã strip hết
    # tag) thay vì chunk-liền-trước đã lật → khoá giọng, vẫn để cảm xúc từng chunk tự do.
    mt_neutral_anchor: bool = False


@dataclass
class ReferenceCacheEntry:
    audio_path: Path
    transcript: str
    audio_cache_hit: bool


@dataclass
class ChunkResult:
    audio_bytes: bytes
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0


@dataclass
class TTSJob:
    request_id: str
    status: str
    created_at: float
    updated_at: float
    format: str
    chunks_total: int
    chunks_completed: int = 0
    chunks_failed: int = 0
    input_chars: int = 0
    lane: str = "default"
    detail: Optional[str] = None
    chunk_paths: Optional[list[Path]] = None
    chunk_media_type: Optional[str] = None
    transcript: str = ""
    audio_cache_hit: Optional[bool] = None
    cleanup_paths: Optional[list[Path]] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    engine_time_s: float = 0.0


def _ensure_dirs() -> None:
    REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def _request_lane(req: TTSRequest) -> str:
    if not SHORT_RESERVED_CHUNKS:
        return "default"
    input_chars = sum(len(chunk) for chunk in req.chunks)
    if input_chars <= SHORT_REQUEST_MAX_CHARS and len(req.chunks) <= SHORT_REQUEST_MAX_CHUNKS:
        return "short"
    return "long"


def _lane_semaphore(lane: str) -> asyncio.Semaphore:
    if lane == "short" and short_chunk_semaphore is not None:
        return short_chunk_semaphore
    if lane == "long" and SHORT_RESERVED_CHUNKS:
        return long_chunk_semaphore
    return chunk_semaphore


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_audio_suffix(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in SUPPORTED_REF_AUDIO_SUFFIXES:
        return suffix
    return ".audio"


def _audio_cache_path(ref_audio_url: str) -> Path:
    return REF_AUDIO_DIR / f"{_sha256(ref_audio_url)}{_safe_audio_suffix(ref_audio_url)}"


def _transcript_cache_path(ref_audio_url: str) -> Path:
    return TRANSCRIPT_DIR / f"{_sha256(ref_audio_url)}.json"


def _write_transcript(ref_audio_url: str, transcript: str) -> None:
    payload = {
        "ref_audio_url": ref_audio_url,
        "transcript": transcript,
        "updated_at": time.time(),
    }
    path = _transcript_cache_path(ref_audio_url)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _write_transcript_if_needed(ref_audio_url: str, transcript: str) -> None:
    path = _transcript_cache_path(ref_audio_url)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("ref_audio_url") == ref_audio_url
            and existing.get("transcript") == transcript
        ):
            return
    except (OSError, ValueError, TypeError):
        pass
    _write_transcript(ref_audio_url, transcript)


def _validate_token(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="API_TOKEN is not configured.")
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _validate_request(req: TTSRequest) -> None:
    req.format = req.format.lower().strip()
    if req.format not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail="format must be wav or mp3.")

    req.chunks = [chunk.strip() for chunk in req.chunks if chunk and chunk.strip()]
    if not req.chunks:
        raise HTTPException(status_code=400, detail="chunks is required and must contain non-empty strings.")

    req.ref_audio_url = req.ref_audio_url.strip()
    parsed = urlparse(req.ref_audio_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="ref_audio_url must be http(s).")

    req.ref_text = req.ref_text.strip()
    if not req.ref_text:
        raise HTTPException(status_code=400, detail="ref_text is required.")

    if req.speed is not None and req.speed <= 0:
        raise HTTPException(status_code=400, detail="speed must be greater than 0.")
    if req.max_new_tokens is not None and req.max_new_tokens <= 0:
        raise HTTPException(status_code=400, detail="max_new_tokens must be greater than 0.")
    if req.temperature is not None and req.temperature <= 0:
        raise HTTPException(status_code=400, detail="temperature must be greater than 0.")
    if req.top_p is not None and (req.top_p <= 0 or req.top_p > 1):
        raise HTTPException(status_code=400, detail="top_p must be between 0 and 1.")
    if req.top_k is not None and req.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be greater than 0.")
    if req.repetition_penalty is not None and req.repetition_penalty <= 0:
        raise HTTPException(status_code=400, detail="repetition_penalty must be greater than 0.")


async def _get_cache_lock(key: str) -> asyncio.Lock:
    async with cache_locks_guard:
        lock = cache_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            cache_locks[key] = lock
        return lock


async def _download_ref_audio(ref_audio_url: str, target: Path) -> None:
    tmp_path = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    last_exc: Optional[BaseException] = None
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
                async with client.stream("GET", ref_audio_url) as response:
                    if response.status_code >= 400:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Could not download ref_audio_url: HTTP {response.status_code}",
                        )
                    with tmp_path.open("wb") as handle:
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                handle.write(chunk)
            if tmp_path.stat().st_size <= 0:
                raise HTTPException(status_code=400, detail="Downloaded ref_audio_url is empty.")
            tmp_path.replace(target)
            return
        except HTTPException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        except Exception as exc:
            last_exc = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < 3:
                await asyncio.sleep(attempt)
    raise HTTPException(
        status_code=400,
        detail=f"Could not download ref_audio_url after 3 attempts. Last error: {last_exc}",
    )


async def _prepare_reference(req: TTSRequest) -> ReferenceCacheEntry:
    cache_key = _sha256(req.ref_audio_url)
    audio_path = _audio_cache_path(req.ref_audio_url)
    if audio_path.exists() and audio_path.stat().st_size > 0:
        _write_transcript_if_needed(req.ref_audio_url, req.ref_text)
        return ReferenceCacheEntry(
            audio_path=audio_path,
            transcript=req.ref_text,
            audio_cache_hit=True,
        )

    lock = await _get_cache_lock(cache_key)
    async with lock:
        audio_cache_hit = audio_path.exists() and audio_path.stat().st_size > 0
        if not audio_cache_hit:
            await _download_ref_audio(req.ref_audio_url, audio_path)
        _write_transcript_if_needed(req.ref_audio_url, req.ref_text)
        return ReferenceCacheEntry(
            audio_path=audio_path,
            transcript=req.ref_text,
            audio_cache_hit=audio_cache_hit,
        )


async def _try_reserve_chunks(chunk_count: int) -> tuple[bool, int]:
    global outstanding_chunks
    async with outstanding_chunks_lock:
        # Shed tải khi backlog HIỆN TẠI đã chạm ngưỡng; nhưng LUÔN nhận 1 job khi còn dưới ngưỡng,
        # kể cả job có chunk_count > cap. Nếu so theo (outstanding + chunk_count) thì 1 job lớn
        # (vd 369 chunk) sẽ bị 429 vĩnh viễn → fail. Job lớn được nhận sẽ tự đẩy backlog vượt cap
        # và chặn job mới tới khi rút bớt (least-loaded routing đẩy job khác sang bridge rảnh).
        if outstanding_chunks >= BUSY_BACKLOG_CHUNKS:
            return False, outstanding_chunks
        outstanding_chunks += chunk_count
        return True, outstanding_chunks


async def _release_chunks(chunk_count: int) -> None:
    global outstanding_chunks
    async with outstanding_chunks_lock:
        outstanding_chunks = max(0, outstanding_chunks - chunk_count)


def _media_type_for_format(audio_format: str) -> str:
    return "audio/mpeg" if audio_format == "mp3" else "audio/wav"


def _is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _is_mp3(data: bytes) -> bool:
    if data.startswith(b"ID3"):
        return True
    return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0


def _unwrap_sglang_audio(data: bytes) -> bytes:
    if len(data) < 8:
        return data
    frame_count = int.from_bytes(data[:4], "big")
    frame_size = int.from_bytes(data[4:8], "big")
    if frame_count == 1 and frame_size == len(data) - 8:
        framed = data[8:]
        if _is_mp3(framed) or _is_wav(framed):
            return framed
    return data


async def _wav_to_mp3(wav_bytes: bytes) -> bytes:
    if not shutil.which(FFMPEG_BIN):
        raise RuntimeError("ffmpeg is required to convert WAV output to MP3.")

    process = await asyncio.create_subprocess_exec(
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "wav",
        "-i",
        "pipe:0",
        "-f",
        "mp3",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(wav_bytes)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to convert WAV to MP3: {stderr.decode('utf-8', 'replace')}")
    return stdout


def _wav_seconds(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as r:
            return r.getnframes() / float(r.getframerate())
    except Exception:
        return 0.0


def _wav_concat_tail(wavs: list[bytes], seconds: float) -> Optional[bytes]:
    """Concatenate WAV chunks (chronological) and return the last `seconds` as a
    WAV. Spans chunk boundaries, so a tiny trailing chunk merges with earlier
    ones to fill the window. Returns None if nothing decodable."""
    params = None
    frames = bytearray()
    for w in wavs:
        try:
            with wave.open(io.BytesIO(w), "rb") as r:
                if params is None:
                    params = r.getparams()
                frames += r.readframes(r.getnframes())
        except Exception:
            continue
    if params is None or not frames:
        return None
    frame_size = params.nchannels * params.sampwidth
    keep_bytes = int(seconds * params.framerate) * frame_size
    tail = bytes(frames[-keep_bytes:]) if 0 < keep_bytes < len(frames) else bytes(frames)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as o:
        o.setparams(params)
        o.writeframes(tail)
    return buf.getvalue()


_HIGGS_TOKEN_RE = re.compile(r"<\|[a-z_]+:[a-z_]+\|>", re.IGNORECASE)


def _strip_higgs_tags(text: str) -> str:
    """Bỏ MỌI token điều khiển <|cat:tag|> → text trung tính (dựng neutral anchor)."""
    return _HIGGS_TOKEN_RE.sub("", text).strip()


def _sglang_payload(
    chunk_text: str,
    req: TTSRequest,
    ref: ReferenceCacheEntry,
    context: Optional[list[tuple[str, bytes]]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input": chunk_text,
        # Multi-turn keeps chunks as WAV internally so the rolling context window
        # can be concatenated/trimmed losslessly; output is re-encoded later.
        "response_format": "wav" if req.multi_turn else req.format,
        "references": [
            {
                "audio_path": str(ref.audio_path),
                "text": ref.transcript,
            }
        ],
    }
    # Multi-turn grounding: prior (text, audio) turns sent via stage_params (a
    # first-class sgl-omni field) so the engine interleaves them before the
    # current text. Audio is encoded server-side by the audio_encoder stage.
    if context:
        payload["stage_params"] = {
            "preprocessing": {
                "context": [
                    {"text": t, "audio": {"base64": base64.b64encode(a).decode()}}
                    for t, a in context
                ]
            }
        }
    if SPEECH_MODEL:
        payload["model"] = SPEECH_MODEL
    for field in OPTIONAL_SGLANG_FIELDS:
        value = getattr(req, field)
        if value is not None:
            payload[field] = value
    # Cap động theo chunk nếu request không tự chỉ định → chặn runaway (default model 2048 ≈ 82s).
    if req.max_new_tokens is None:
        payload["max_new_tokens"] = _estimate_max_new_tokens(chunk_text)
    # Sampling mặc định khi client không gửi → tránh phân bố khuếch tán (temp=1.0, no top_p/top_k)
    # vốn dễ dẫn tới silence-attractor không emit EOC. setdefault → tôn trọng override của client.
    payload.setdefault("temperature", HIGGS_TEMPERATURE)
    payload.setdefault("top_p", HIGGS_TOP_P)
    payload.setdefault("top_k", HIGGS_TOP_K)
    return payload


def _header_int(headers: httpx.Headers, key: str) -> int:
    try:
        return int(headers.get(key, "0") or "0")
    except ValueError:
        return 0


def _header_float(headers: httpx.Headers, key: str) -> float:
    try:
        return float(headers.get(key, "0") or "0")
    except ValueError:
        return 0.0


async def _max_volume_dbfs(audio_bytes: bytes) -> Optional[float]:
    """max_volume (dBFS) của audio qua ffmpeg volumedetect. None nếu không phân tích được."""
    if not audio_bytes or not shutil.which(FFMPEG_BIN):
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            FFMPEG_BIN, "-hide_banner", "-nostats",
            "-i", "pipe:0", "-af", "volumedetect", "-f", "null", "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate(audio_bytes)
    except Exception as exc:
        logger.warning("volumedetect failed: %s", exc)
        return None
    text = stderr.decode("utf-8", "replace")
    match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", text)
    if match:
        return float(match.group(1))
    if re.search(r"max_volume:\s*-inf", text):
        return float("-inf")
    return None


async def _call_sglang(
    chunk_text: str, req: TTSRequest, ref: ReferenceCacheEntry, seed_override: Optional[int] = None,
    context: Optional[list[tuple[str, bytes]]] = None,
) -> ChunkResult:
    payload = _sglang_payload(chunk_text, req, ref, context=context)
    if seed_override is not None:
        payload["seed"] = seed_override
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(f"{SGLANG_BASE_URL}/v1/audio/speech", json=payload)

    if response.status_code >= 400:
        detail = response.text[:500]
        raise RuntimeError(f"SGLang returned HTTP {response.status_code}: {detail}")

    audio_bytes = _unwrap_sglang_audio(response.content)
    if not audio_bytes:
        raise RuntimeError("SGLang returned empty audio.")

    result = ChunkResult(
        audio_bytes=audio_bytes,
        prompt_tokens=_header_int(response.headers, "x-prompt-tokens"),
        completion_tokens=_header_int(response.headers, "x-completion-tokens"),
        engine_time_s=_header_float(response.headers, "x-engine-time"),
    )

    # Multi-turn keeps WAV (re-encoded to req.format when writing the chunk file).
    if not req.multi_turn and req.format == "mp3" and not _is_mp3(result.audio_bytes) and _is_wav(audio_bytes):
        result.audio_bytes = await _wav_to_mp3(audio_bytes)

    # Audio rỗng/quá nhỏ = chunk gần-câm/cụt → coi là lỗi để retry (xem _generate_one_chunk).
    if CHUNK_MIN_BYTES and len(result.audio_bytes) < CHUNK_MIN_BYTES:
        raise RuntimeError(
            f"SGLang returned undersized audio ({len(result.audio_bytes)} bytes < {CHUNK_MIN_BYTES})."
        )

    # Runaway EOS (tín hiệu CHÍNH, miễn phí): model không emit EOS → chạy tới đúng max_new_tokens.
    # Bắt được CẢ câm-toàn-phần LẪN "đọc một đoạn rồi câm tới hết cap" (max_volume bỏ sót vì có tiếng
    # ở đầu). Chỉ áp khi cap do wrapper tự đặt (req.max_new_tokens None) và cap < ceil model → tránh
    # nhầm chunk dài hợp lệ chạm trần thật. Retry đổi seed (xem _call_sglang_with_retry).
    cap = payload.get("max_new_tokens")
    if (
        req.max_new_tokens is None
        and isinstance(cap, int)
        and cap < MAX_NEW_TOKENS_CEIL
        and result.completion_tokens >= cap * 0.95
    ):
        raise RuntimeError(
            f"hit max_new_tokens cap ({result.completion_tokens}/{cap}); EOS-runaway → retry seed"
        )

    # Backstop: câm-toàn-phần mà vẫn emit EOS sớm (không chạm cap) → đo âm lượng.
    if CHUNK_SILENCE_MAX_DBFS > -90:
        max_db = await _max_volume_dbfs(result.audio_bytes)
        if max_db is not None and max_db < CHUNK_SILENCE_MAX_DBFS:
            raise RuntimeError(
                f"silent audio output (max_volume {max_db:.1f} dBFS < {CHUNK_SILENCE_MAX_DBFS}); likely EOS-runaway"
            )

    return result


async def _call_sglang_with_retry(
    request_id: str,
    chunk_index: int,
    text: str,
    req: TTSRequest,
    ref: ReferenceCacheEntry,
    context: Optional[list[tuple[str, bytes]]] = None,
) -> ChunkResult:
    last_exc: Optional[BaseException] = None
    for attempt in range(1, CHUNK_RETRY_ATTEMPTS + 1):
        try:
            # Attempt đầu dùng seed tự nhiên; retry ĐỔI seed để thoát runaway/silent (sampling fluke,
            # cùng seed thường câm lại). seed=None ở lần đầu → giữ hành vi mặc định của backend.
            seed_override = None if attempt == 1 else random.randint(1, 2_147_483_647)
            return await _call_sglang(text, req, ref, seed_override=seed_override, context=context)
        except Exception as exc:
            last_exc = exc
            if attempt < CHUNK_RETRY_ATTEMPTS:
                logger.warning(
                    "TTS job %s chunk %d attempt %d/%d failed: %s; retrying with new seed",
                    request_id, chunk_index, attempt, CHUNK_RETRY_ATTEMPTS, exc,
                )
                await asyncio.sleep(CHUNK_RETRY_BASE_DELAY * attempt)
    assert last_exc is not None
    raise last_exc


async def _generate_one_chunk(
    request_id: str,
    chunk_index: int,
    text: str,
    req: TTSRequest,
    ref: ReferenceCacheEntry,
    output_path: Path,
    lane: str,
    job_semaphore: asyncio.Semaphore,
    context: Optional[list[tuple[str, bytes]]] = None,
) -> ChunkResult:
    async with job_semaphore:
        async with _lane_semaphore(lane):
            result = await _call_sglang_with_retry(request_id, chunk_index, text, req, ref, context=context)
            # Multi-turn keeps result.audio_bytes as WAV (for the context window);
            # the chunk FILE still needs req.format. Re-encode only for the write.
            file_bytes = result.audio_bytes
            if req.multi_turn and req.format == "mp3" and _is_wav(file_bytes):
                file_bytes = await _wav_to_mp3(file_bytes)
            # Ghi atomic: /audio đọc theo exists() nên không được để lộ file ghi dở.
            tmp_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                tmp_path.write_bytes(file_bytes)
                os.replace(tmp_path, output_path)
            except BaseException:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise

    await _release_chunks(1)
    async with jobs_lock:
        job = jobs.get(request_id)
        if job is not None:
            job.chunks_completed += 1
            job.prompt_tokens += result.prompt_tokens
            job.completion_tokens += result.completion_tokens
            job.total_tokens = job.prompt_tokens + job.completion_tokens
            job.engine_time_s += result.engine_time_s
            job.updated_at = time.time()
    return result


async def _render_anchor(
    request_id: str,
    text: str,
    req: TTSRequest,
    ref: ReferenceCacheEntry,
    lane: str,
    job_semaphore: asyncio.Semaphore,
) -> bytes:
    """Render 1 đoạn TRUNG TÍNH (đã strip tag) làm mỏ neo cố định cho multi-turn khi có
    tag lật giọng. Trả WAV bytes (multi_turn giữ WAV); KHÔNG ghi file chunk. Có retry
    chống runaway/silent như chunk thường (chunk_index=-1 chỉ để log)."""
    async with job_semaphore:
        async with _lane_semaphore(lane):
            result = await _call_sglang_with_retry(request_id, -1, text, req, ref, context=None)
    return result.audio_bytes


async def _run_tts_job(request_id: str, req: TTSRequest) -> None:
    completed_outputs = 0
    try:
        async with jobs_lock:
            job = jobs.get(request_id)
            if job is not None:
                job.status = "running"
                job.updated_at = time.time()

        ref = await _prepare_reference(req)
        job_dir = JOB_DIR / request_id
        job_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".mp3" if req.format == "mp3" else ".wav"
        output_paths = [job_dir / f"chunk_{index:05d}{suffix}" for index in range(len(req.chunks))]

        async with jobs_lock:
            job = jobs.get(request_id)
            if job is not None:
                job.transcript = ref.transcript
                job.audio_cache_hit = ref.audio_cache_hit
                job.cleanup_paths = [job_dir]
                # Gán sớm để /audio phục vụ progressive khi job còn "running";
                # file chunk xuất hiện dần (ghi atomic), /audio gate theo exists().
                job.chunk_paths = output_paths
                job.chunk_media_type = _media_type_for_format(req.format)
                job.updated_at = time.time()

        lane = job.lane if job else _request_lane(req)
        job_semaphore = asyncio.Semaphore(min(MAX_IN_FLIGHT_CHUNKS_PER_JOB, len(req.chunks)))

        if req.multi_turn:
            # Natural mode: chunks run SEQUENTIALLY, each grounded on the
            # immediately-prior chunk as a FULL text↔audio PAIR (K1-full). This
            # mirrors Higgs' official long-form format (generation.py keeps every
            # turn as a (user text, assistant audio) pair, pruned 2:1) — the
            # transcript must stay paired with its audio. Audio-only or tail-
            # trimmed context breaks that pairing and makes the model drift or
            # re-read (text says more than the clipped audio contains).
            # result.audio_bytes is WAV in multi-turn → lossless context.
            results: list[Any] = []
            tasks = []
            # Neo trung tính khi có tag LẬT GIỌNG: chain neo vào chunk-liền-trước sẽ kế
            # thừa giọng đã lật (trôi / đổi giới tính). Thay vào đó dựng 1 mỏ neo TRUNG
            # TÍNH cố định = chunk0 đã strip hết tag, render 1 lần, rồi neo MỌI chunk vào
            # nó → khoá giọng, cảm xúc từng chunk vẫn tự do. Render neo fail → fallback chain.
            anchor: Optional[tuple[str, bytes]] = None
            if req.mt_neutral_anchor and req.chunks:
                neutral_text = _strip_higgs_tags(req.chunks[0])
                if neutral_text:
                    try:
                        anchor = (
                            neutral_text,
                            await _render_anchor(request_id, neutral_text, req, ref, lane, job_semaphore),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "TTS job %s neutral-anchor render failed: %s; fallback to chain",
                            request_id, exc,
                        )
            prev: Optional[tuple[str, bytes]] = None  # (full text, full WAV) of prior chunk
            for index, text in enumerate(req.chunks):
                # anchor cố định (voice-flip) > chain chunk-trước (K1-full thường).
                ctx = [anchor] if anchor is not None else ([prev] if prev else None)
                try:
                    result = await _generate_one_chunk(
                        request_id, index, text, req, ref, output_paths[index],
                        lane, job_semaphore, context=ctx,
                    )
                    results.append(result)
                    prev = (text, result.audio_bytes)
                except Exception as exc:  # noqa: BLE001 — surfaced via shared handler below
                    results.append(exc)
                    break
        else:
            tasks = [
                asyncio.create_task(_generate_one_chunk(request_id, index, text, req, ref, output_paths[index], lane, job_semaphore))
                for index, text in enumerate(req.chunks)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]
        completed_outputs = sum(1 for path in output_paths if path.exists())

        if errors:
            for task in tasks:
                if not task.done():
                    task.cancel()
            remaining = len(req.chunks) - completed_outputs
            if remaining > 0:
                await _release_chunks(remaining)
            detail = str(errors[0])
            async with jobs_lock:
                job = jobs.get(request_id)
                if job is not None:
                    job.status = "failed"
                    job.detail = detail
                    job.chunks_failed = len(errors)
                    job.updated_at = time.time()
            logger.error("TTS job %s failed: %s", request_id, detail)
            return

        async with jobs_lock:
            job = jobs.get(request_id)
            if job is not None:
                job.status = "succeeded"
                job.chunk_paths = output_paths
                job.chunk_media_type = _media_type_for_format(req.format)
                job.updated_at = time.time()
    except Exception as exc:
        remaining = len(req.chunks) - completed_outputs
        if remaining > 0:
            await _release_chunks(remaining)
        async with jobs_lock:
            job = jobs.get(request_id)
            if job is not None:
                job.status = "failed"
                job.detail = str(exc)
                job.chunks_failed = max(job.chunks_failed, remaining)
                job.updated_at = time.time()
        logger.exception("TTS job %s failed", request_id)


def _job_payload(job: TTSJob) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": job.request_id,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "status_url": f"/v1/tts/jobs/{job.request_id}",
        "chunks_total": job.chunks_total,
        "chunks_completed": job.chunks_completed,
        "chunks_failed": job.chunks_failed,
        "input_chars": job.input_chars,
        "lane": job.lane,
        "format": job.format,
    }
    if job.detail:
        payload["detail"] = job.detail
    if job.audio_cache_hit is not None:
        payload["cache_hit"] = job.audio_cache_hit
    if job.prompt_tokens or job.completion_tokens or job.engine_time_s:
        payload["usage"] = {
            "prompt_tokens": job.prompt_tokens,
            "completion_tokens": job.completion_tokens,
            "total_tokens": job.total_tokens,
            "engine_time_s": round(job.engine_time_s, 6),
        }
    if job.transcript:
        payload["transcript"] = job.transcript
    if job.status == "succeeded":
        payload["audio_url"] = f"/v1/tts/jobs/{job.request_id}/audio"
    return payload


async def _job_counts() -> dict[str, int]:
    async with jobs_lock:
        return {
            "queued": sum(1 for job in jobs.values() if job.status == "queued"),
            "running": sum(1 for job in jobs.values() if job.status == "running"),
            "succeeded": sum(1 for job in jobs.values() if job.status == "succeeded"),
            "failed": sum(1 for job in jobs.values() if job.status == "failed"),
        }


async def _sglang_health() -> tuple[bool, Optional[Any]]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{SGLANG_BASE_URL}/health")
        if response.status_code >= 400:
            return False, {"status_code": response.status_code, "body": response.text[:500]}
        try:
            return True, response.json()
        except ValueError:
            return True, response.text[:500]
    except Exception as exc:
        return False, str(exc)


async def _cleanup_expired_jobs() -> None:
    now = time.time()
    expired: list[TTSJob] = []
    async with jobs_lock:
        for request_id, job in list(jobs.items()):
            if job.status in {"succeeded", "failed"} and now - job.updated_at > JOB_TTL_SECONDS:
                expired.append(job)
                del jobs[request_id]

    for job in expired:
        for path in job.cleanup_paths or []:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            except OSError as exc:
                logger.warning("Could not remove expired job path %s: %s", path, exc)


async def _periodic_cleanup() -> None:
    while True:
        await asyncio.sleep(JOB_CLEANUP_INTERVAL_SECONDS)
        await _cleanup_expired_jobs()


@app.on_event("startup")
async def startup() -> None:
    global cleanup_task
    _ensure_dirs()
    cleanup_task = asyncio.create_task(_periodic_cleanup())


@app.on_event("shutdown")
async def shutdown() -> None:
    if cleanup_task is not None:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


@app.get("/health")
async def health() -> dict[str, Any]:
    _ensure_dirs()
    sglang_ready, sglang_status = await _sglang_health()
    counts = await _job_counts()
    async with outstanding_chunks_lock:
        current_outstanding = outstanding_chunks
    return {
        "status": "ok",
        "tts_backend_name": TTS_BACKEND_NAME,
        "sglang_ready": sglang_ready,
        "sglang_base_url": SGLANG_BASE_URL,
        "sglang_status": sglang_status,
        "cache_audio_count": len(list(REF_AUDIO_DIR.glob("*"))),
        "cache_transcript_count": len(list(TRANSCRIPT_DIR.glob("*.json"))),
        "active_tts_jobs": counts["queued"] + counts["running"],
        "tts_jobs": counts,
        "max_concurrent_chunks": MAX_CONCURRENT_CHUNKS,
        "short_reserved_chunks": SHORT_RESERVED_CHUNKS,
        "long_concurrent_chunks": LONG_CONCURRENT_CHUNKS,
        "short_request_max_chars": SHORT_REQUEST_MAX_CHARS,
        "short_request_max_chunks": SHORT_REQUEST_MAX_CHUNKS,
        "max_in_flight_chunks_per_job": MAX_IN_FLIGHT_CHUNKS_PER_JOB,
        "busy_backlog_chunks": BUSY_BACKLOG_CHUNKS,
        "outstanding_chunks": current_outstanding,
        "job_ttl_seconds": JOB_TTL_SECONDS,
    }


@app.post("/v1/cache/clear")
async def clear_cache(_: None = Depends(_validate_token)) -> dict[str, str]:
    for path in (REF_AUDIO_DIR, TRANSCRIPT_DIR):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    async with cache_locks_guard:
        cache_locks.clear()
    return {
        "status": "ok",
        "message": "All cached reference audios and transcripts cleared successfully.",
    }


@app.post("/v1/tts")
async def submit_tts(
    req: TTSRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(_validate_token),
) -> JSONResponse:
    _ensure_dirs()
    _validate_request(req)
    reserved, current_outstanding = await _try_reserve_chunks(len(req.chunks))
    if not reserved:
        raise HTTPException(
            status_code=429,
            detail=(
                "TTS chunk backlog is busy; retry later. "
                f"outstanding_chunks={current_outstanding}, limit={BUSY_BACKLOG_CHUNKS}."
            ),
            headers={
                "Retry-After": "1",
                "X-Busy-Backlog-Chunks": str(BUSY_BACKLOG_CHUNKS),
                "X-Outstanding-Chunks": str(current_outstanding),
            },
        )

    request_id = str(uuid.uuid4())
    now = time.time()
    lane = _request_lane(req)
    job = TTSJob(
        request_id=request_id,
        status="queued",
        created_at=now,
        updated_at=now,
        format=req.format,
        chunks_total=len(req.chunks),
        input_chars=sum(len(chunk) for chunk in req.chunks),
        lane=lane,
    )
    async with jobs_lock:
        jobs[request_id] = job

    background_tasks.add_task(_run_tts_job, request_id, req)
    return JSONResponse(
        status_code=202,
        content=_job_payload(job),
        headers={
            "X-Request-Id": request_id,
            "Location": f"/v1/tts/jobs/{request_id}",
        },
    )


@app.get("/v1/tts/jobs/{request_id}")
async def get_tts_job(
    request_id: str,
    _: None = Depends(_validate_token),
) -> JSONResponse:
    async with jobs_lock:
        job = jobs.get(request_id)
        if job is None:
            raise HTTPException(status_code=404, detail="TTS job not found.")
        return JSONResponse(content=_job_payload(job))


@app.get("/v1/tts/jobs/{request_id}/audio")
async def get_tts_job_audio(
    request_id: str,
    chunk_from: int = Query(0, alias="from"),
    chunks: Optional[int] = Query(None),
    _: None = Depends(_validate_token),
) -> StreamingResponse:
    if chunk_from < 0:
        raise HTTPException(status_code=400, detail="from must be >= 0.")
    if chunks is not None and chunks < 1:
        raise HTTPException(status_code=400, detail="chunks must be >= 1.")

    async with jobs_lock:
        job = jobs.get(request_id)
        if job is None:
            raise HTTPException(status_code=404, detail="TTS job not found.")
        if job.status == "queued":
            raise HTTPException(status_code=409, detail="TTS job is queued.")
        if job.status == "failed":
            raise HTTPException(status_code=409, detail=f"TTS job failed: {job.detail or 'unknown error'}.")
        # running | succeeded: chunk_paths đã được gán sớm trong _run_tts_job.
        if not job.chunk_paths:
            raise HTTPException(status_code=409, detail=f"TTS job is {job.status}.")
        job_status = job.status
        all_chunk_paths = list(job.chunk_paths)
        total_chunks = len(all_chunk_paths)
        if chunk_from >= total_chunks:
            raise HTTPException(status_code=416, detail="from is outside available chunks.")
        chunk_to = total_chunks if chunks is None else min(chunk_from + chunks, total_chunks)
        requested_paths = all_chunk_paths[chunk_from:chunk_to]
        media_type = job.chunk_media_type or _media_type_for_format(job.format)
        transcript = job.transcript
        cache_hit = job.audio_cache_hit
        prompt_tokens = job.prompt_tokens
        completion_tokens = job.completion_tokens
        total_tokens = job.total_tokens
        engine_time_s = job.engine_time_s

    # Chỉ phục vụ đoạn LIỀN-MẠCH đã ghi xong tính từ `from`; dừng ở chunk đầu tiên chưa có.
    # Worker tự cộng dồn `fetched += parsed.length` rồi xin tiếp from kế tiếp.
    chunk_paths: list[Path] = []
    for path in requested_paths:
        if path.exists():
            chunk_paths.append(path)
        else:
            break

    if not chunk_paths:
        if job_status == "succeeded":
            # succeeded mà file không còn → đã bị cleanup theo TTL.
            raise HTTPException(status_code=410, detail="TTS job audio expired.")
        # chunk `from` chưa sinh xong → worker poll lại.
        raise HTTPException(status_code=409, detail="TTS chunk not ready yet.")

    async def stream_length_prefixed():
        yield len(chunk_paths).to_bytes(4, "big")
        for path in chunk_paths:
            data = await asyncio.to_thread(path.read_bytes)
            yield len(data).to_bytes(4, "big")
            yield data

    headers = {
        "X-Request-Id": request_id,
        "X-Cache-Hit": str(bool(cache_hit)).lower(),
        "X-Transcript": quote(transcript or "", safe=""),
        "X-Transcript-Encoding": "urlencoded-utf8",
        "X-Chunk-From": str(chunk_from),
        "X-Chunks-Returned": str(len(chunk_paths)),
        "X-Chunks-Total": str(total_chunks),
        "X-Audio-Format": media_type,
        "X-Prompt-Tokens": str(prompt_tokens),
        "X-Completion-Tokens": str(completion_tokens),
        "X-Total-Tokens": str(total_tokens),
        "X-Engine-Time": f"{engine_time_s:.6f}",
    }
    return StreamingResponse(
        stream_length_prefixed(),
        media_type="application/octet-stream",
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8001")))
