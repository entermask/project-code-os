import array
import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import threading
import time
import uuid
import wave
from collections import deque
from contextlib import asynccontextmanager
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


PCM_STATS_AUDIOOP = os.getenv("HIGGS_PCM_STATS_AUDIOOP", "0").strip().lower()
if PCM_STATS_AUDIOOP not in ("0", "peak", "all"):
    raise RuntimeError("HIGGS_PCM_STATS_AUDIOOP must be 0, peak, or all")
if PCM_STATS_AUDIOOP != "0":
    try:
        import audioop as _audioop
    except ImportError as exc:
        raise RuntimeError(
            "HIGGS_PCM_STATS_AUDIOOP requires Python 3.12 audioop or audioop-lts"
        ) from exc
else:
    _audioop = None


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
STREAMED_JOB_TTL_SECONDS = int(os.getenv("STREAMED_JOB_TTL_SECONDS", "600"))
JOB_CLEANUP_INTERVAL_SECONDS = int(os.getenv("JOB_CLEANUP_INTERVAL_SECONDS", "60"))
STREAM_CHUNK_SIZE_BYTES = max(1, int(os.getenv("STREAM_CHUNK_SIZE_BYTES", str(1024 * 1024))))
MAX_CONCURRENT_CHUNKS = max(1, int(os.getenv("MAX_CONCURRENT_CHUNKS", "4")))
SHORT_RESERVED_CHUNKS = max(0, int(os.getenv("SHORT_RESERVED_CHUNKS", "0")))
if SHORT_RESERVED_CHUNKS >= MAX_CONCURRENT_CHUNKS:
    SHORT_RESERVED_CHUNKS = MAX_CONCURRENT_CHUNKS - 1
LANE_ADMISSION_MODE = (
    os.getenv("HIGGS_LANE_ADMISSION_MODE", "dual").strip().lower().replace("-", "_")
)
if LANE_ADMISSION_MODE not in {"dual", "soft_reserved"}:
    raise RuntimeError("HIGGS_LANE_ADMISSION_MODE must be dual or soft_reserved")
SHORT_REQUEST_MAX_CHARS = max(1, int(os.getenv("SHORT_REQUEST_MAX_CHARS", "1000")))
SHORT_REQUEST_MAX_CHUNKS = max(1, int(os.getenv("SHORT_REQUEST_MAX_CHUNKS", "4")))
LONG_CONCURRENT_CHUNKS = MAX_CONCURRENT_CHUNKS - SHORT_RESERVED_CHUNKS
MAX_IN_FLIGHT_CHUNKS_PER_JOB = max(1, int(os.getenv("MAX_IN_FLIGHT_CHUNKS_PER_JOB", "12")))
MAX_BURST_IN_FLIGHT_CHUNKS_PER_JOB = min(
    MAX_CONCURRENT_CHUNKS,
    max(
        MAX_IN_FLIGHT_CHUNKS_PER_JOB,
        int(os.getenv("MAX_BURST_IN_FLIGHT_CHUNKS_PER_JOB", str(MAX_IN_FLIGHT_CHUNKS_PER_JOB))),
    ),
)
MAX_BURST_ACTIVE_JOBS = max(1, int(os.getenv("MAX_BURST_ACTIVE_JOBS", "2")))
# Fair-share quota (opt-in, xem _job_in_flight_limit). Mặc định TẮT → giữ nguyên
# hai nấc burst/thường như cũ.
JOB_QUOTA_FAIR_SHARE = os.getenv("JOB_QUOTA_FAIR_SHARE", "0").strip().lower() in ("1", "true", "yes", "on")
JOB_QUOTA_MIN = max(1, int(os.getenv("JOB_QUOTA_MIN", "12")))
JOB_QUOTA_MAX = max(JOB_QUOTA_MIN, int(os.getenv("JOB_QUOTA_MAX", "24")))
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
# audio-token/char THEO TỪNG SCRIPT (codec Higgs 25 fps). ĐO THỰC trên model (chunk ngắn đọc sạch):
# Han(Trung) ~6.1, Kana(Nhật) ~4.8, Hangul(Hàn) ~3.7, Arabic ~2.9, Thai ~2.33, Devanagari(Hindi) ~2.27,
# Cyrillic ~2.1, Latin/Việt ~1.4-1.66. Phân loại 2-nhóm cũ (dense 6.0 / latin 2.0) quá thô: OVER cho
# Hàn/Nhật/Việt, UNDER cho Arabic/Thai/Hindi. Để hơi-cao (an toàn): cap clamp 2048 đằng nào cũng OK,
# còn sub-split & early-EOS bắn TRƯỚC khi cụt. Dùng cho cap động, early-EOS threshold, và sub-split gate.
TOK_PER_CHAR_LATIN = float(os.getenv("TOK_PER_CHAR_LATIN", "2.0"))            # default: latin/cyrillic/khác
TOK_PER_CHAR_HAN = float(os.getenv("TOK_PER_CHAR_HAN", "6.0"))                # CJK ideographs (Trung/kanji)
TOK_PER_CHAR_KANA = float(os.getenv("TOK_PER_CHAR_KANA", "4.8"))             # hiragana/katakana (Nhật)
TOK_PER_CHAR_HANGUL = float(os.getenv("TOK_PER_CHAR_HANGUL", "3.8"))          # Hàn
TOK_PER_CHAR_ARABIC = float(os.getenv("TOK_PER_CHAR_ARABIC", "3.0"))          # Arabic abjad
TOK_PER_CHAR_THAI = float(os.getenv("TOK_PER_CHAR_THAI", "2.5"))              # Thai
TOK_PER_CHAR_DEVANAGARI = float(os.getenv("TOK_PER_CHAR_DEVANAGARI", "2.4"))  # Hindi
TOK_PER_CHAR_DENSE = float(os.getenv("TOK_PER_CHAR_DENSE", "6.0"))            # (giữ tương thích env cũ)
MAX_NEW_TOKENS_SAFETY = float(os.getenv("MAX_NEW_TOKENS_SAFETY", "2.0"))
MAX_NEW_TOKENS_BASE = int(os.getenv("MAX_NEW_TOKENS_BASE", "96"))
MAX_NEW_TOKENS_FLOOR = max(1, int(os.getenv("MAX_NEW_TOKENS_FLOOR", "256")))
MAX_NEW_TOKENS_CEIL = max(1, int(os.getenv("MAX_NEW_TOKENS_CEIL", "2048")))  # = model default, không vượt
# Default max_new_tokens của higgs-audio-v3 (theo docs) = trần trên cho chunk có context.
HIGGS_DEFAULT_MAX_NEW_TOKENS = max(1, int(os.getenv("HIGGS_DEFAULT_MAX_NEW_TOKENS", "2048")))
# "CHẶN DƯỚI" (early-EOS / đọc THIẾU chữ): model phát EOS sớm → audio HỢP LỆ nhưng CỤT (có tiếng,
# chưa chạm cap → qua hết gate trên) → worker nhận đoạn thiếu. Bắt bằng completion_tokens thực <
# EARLY_EOS_RATIO × kỳ vọng-token-từ-text (codec 25 fps). Re-render đổi seed (EOS sớm là fluke
# sampling, seed khác thường đọc trọn). Chỉ áp khi text đủ dài để tránh false-pos ở chunk ngắn.
EARLY_EOS_RATIO = float(os.getenv("EARLY_EOS_RATIO", "0.5"))
EARLY_EOS_MIN_EXPECTED_TOKENS = max(1, int(os.getenv("EARLY_EOS_MIN_EXPECTED_TOKENS", "96")))
# Sub-split fallback (LƯỚI AN TOÀN cuối): sgl-omni KHÔNG tự chia — mỗi /v1/audio/speech là 1 generation,
# vượt trần max_new → CẮT CỤT ÂM THẦM (finished_reason=length). Nếu _expected_tokens(chunk) > effective_cap
# (single-turn 2048; multi-turn KV≈1535), reseed VÔ DỤNG (không nhét >82s vào trần) → wrapper tự tách chunk
# ở ranh giới câu, render từng phần, NỐI WAV, giữ 1:1 (1 file/chunk, vô hình với worker & /audio). Gate bằng
# expected > cap nên 99% chunk đúng cỡ KHÔNG vào nhánh này; runaway-mà-đáng-lẽ-vừa do reseed lo (không split).
SUBSPLIT_ENABLE = os.getenv("SUBSPLIT_ENABLE", "1").strip().lower() not in ("0", "false", "no", "")
SUBSPLIT_TARGET_RATIO = float(os.getenv("SUBSPLIT_TARGET_RATIO", "0.7"))  # sub-part ≤ ratio×cap (chừa headroom)
SUBSPLIT_MAX_PARTS = max(2, int(os.getenv("SUBSPLIT_MAX_PARTS", "8")))    # chặn fan-out vô hạn
# KV cache (thinker) chứa CẢ input LẪN generation: input_tokens + max_new_tokens phải ≤ kv_capacity,
# nếu không SGLang trả HTTP 500 "requires more tokens than KV cache can hold". Multi-turn context
# audio ăn input RẤT lớn (đo thực ~2459) → KHÔNG thể dùng higgs default 2048 (2459+2048>4095=tràn).
# Chunk CÓ context: cap max_new = kv_capacity − reserve(input) → vừa KV mà vẫn rộng hơn cap động cũ.
# Nếu context thực to hơn reserve → reactive refit từ thông số trong lỗi 500 (xem _call_sglang_with_retry).
KV_CACHE_CAPACITY = max(1, int(os.getenv("KV_CACHE_CAPACITY", "4095")))
KV_INPUT_RESERVE = max(1, int(os.getenv("KV_INPUT_RESERVE", "2560")))   # chừa cho ref+context+text input
KV_SAFETY_MARGIN = max(0, int(os.getenv("KV_SAFETY_MARGIN", "128")))
# Parse lỗi KV overflow của SGLang: "...input_tokens=2459...kv_capacity=4095..." → refit max_new.
_KV_OVERFLOW_RE = re.compile(r"input_tokens=(\d+).*?kv_capacity=(\d+)", re.IGNORECASE)
# Sampling mặc định cho higgs. Worker không gửi → sgl-omni mặc định temp=1.0, top_p/top_k TẮT =
# phân bố khuếch tán → dễ kẹt "silence attractor" (không sample được EOC) → runaway câm.
# temp=0.8 + top_k=50 là 2 đòn bẩy ổn định CHÍNH, khớp ref boson voice-clone + model card v3.
# top_p: MẶC ĐỊNH của higgs v3 = UNSET (nucleus OFF — xem sgl-omni cookbook `top_p|float|null` và
# sampler.py: nucleus chỉ áp khi top_p is not None and < 1.0). 0.95 là giá trị legacy v2/example,
# KHÔNG phải default v3. Để TRỐNG HIGGS_TOP_P → KHÔNG gửi top_p (đúng default model). Đặt số (vd
# 0.95) chỉ khi muốn bật lại nucleus như đòn anti-silence-attractor.
HIGGS_TEMPERATURE = float(os.getenv("HIGGS_TEMPERATURE", "0.8"))
_HIGGS_TOP_P_ENV = os.getenv("HIGGS_TOP_P", "").strip()
HIGGS_TOP_P: Optional[float] = float(_HIGGS_TOP_P_ENV) if _HIGGS_TOP_P_ENV else None
HIGGS_TOP_K = int(os.getenv("HIGGS_TOP_K", "50"))
# Phụ âm/âm tiết đầu bị nuốt: model thỉnh thoảng vào thẳng frame 0 (không chừa
# frame im lặng nào) → âm mở đầu không có chỗ đặt closure, nghe mất chữ ("đáng
# đời" → "áng đời"). Đo trên prod: 29% chunk_00000 có onset 0ms; theo từng câu tỉ
# lệ 0-100% (không theo lớp phụ âm — "ông" 80%, "anh" 0%). Prefix token prosody
# CHÍNH THỨC của v3 (id 151722, có trong tokenizer) ép model sinh lặng thật:
# gate 100 mẫu qua 6 câu → 0/100 vào frame 0, đổi lại ~12 token (~3.5%) mỗi chunk.
# Lặng thừa được cắt lại ở _trim_lead_silence. Mặc định TẮT.
HIGGS_PAUSE_PREFIX = os.getenv("HIGGS_PAUSE_PREFIX", "0").strip().lower() in ("1", "true", "yes", "on")
_PAUSE_PREFIX_TOKEN = "<|prosody:pause|>"
# Caller TỰ đặt pause đầu chunk → KHÔNG chèn thêm và KHÔNG cắt: nếu cắt thì đúng
# khoảng lặng họ cố ý yêu cầu bị san phẳng về LEAD_SILENCE_KEEP_MS.
_OWN_LEAD_PAUSE_RE = re.compile(r"^\s*(<\|prosody:(?:long_)?pause\|>|<break\b)", re.IGNORECASE)
# Giữ lại bấy nhiêu ms im lặng đầu sau khi cắt. 0 = tắt cắt.
LEAD_SILENCE_KEEP_MS = max(0, int(os.getenv("LEAD_SILENCE_KEEP_MS", "60")))
LEAD_SILENCE_DBFS = float(os.getenv("LEAD_SILENCE_DBFS", "-45"))
# Natural/multi-turn: ground each chunk on the last N seconds of audio already
# produced in this job (audio only, NO text — empirically cleaner). The window
# spans chunk boundaries, so a tiny prior chunk ("OK.") auto-merges with the one
# before it to fill the window.
MT_CONTEXT_TAIL_SEC = float(os.getenv("MT_CONTEXT_TAIL_SEC", "6"))
# Per-script ranges \u0111\u1ec3 \u01b0\u1edbc l\u01b0\u1ee3ng token (rate \u1edf TOK_PER_CHAR_*). KH\u00d4NG ch\u1ed3ng l\u1ea5n; k\u00fd t\u1ef1 kh\u00f4ng kh\u1edbp
# (latin, cyrillic, d\u1ea5u c\u00e2u, CJK-punct \u3000-\u303f, fullwidth) \u2192 default TOK_PER_CHAR_LATIN.
_RE_HAN = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")        # CJK ideographs (Trung/kanji)
_RE_KANA = re.compile("[\u3040-\u30ff\u31f0-\u31ff]")                     # hiragana + katakana (Nh\u1eadt)
_RE_HANGUL = re.compile("[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")     # H\u00e0n
_RE_ARABIC = re.compile("[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]")
_RE_THAI = re.compile("[\u0e00-\u0e7f]")
_RE_DEVANAGARI = re.compile("[\u0900-\u097f]")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFMPEG_THREADS = max(1, int(os.getenv("FFMPEG_THREADS", "1")))
# P1 speaker-boost-on-bridge: worker gửi af_filter (chuỗi -af gồm atempo+boost) → áp khi encode
# WAV→MP3 từng chunk, trả MP3 đã boost (worker chỉ concat-copy). Giới hạn riêng + nice để KHÔNG
# tranh CPU feed-GPU (container cgroup ~22 core).
FFMPEG_POST_CONCURRENCY = max(1, int(os.getenv("FFMPEG_POST_CONCURRENCY", "10")))
_FFMPEG_POST_NICE = os.getenv("FFMPEG_POST_NICE", "19").strip().lower()
_FFMPEG_NICE_PREFIX = (
    [] if _FFMPEG_POST_NICE in ("", "0", "off", "false", "no") else ["nice", "-n", _FFMPEG_POST_NICE]
)
# Test/canary-only observability for locating FFmpeg queue/process bottlenecks.
# Default OFF: when disabled, /health keeps its existing schema and counters do no clock/lock work.
FFMPEG_TIMING_ENABLED = os.getenv("HIGGS_FFMPEG_TIMING", "0").strip().lower() in (
    "1", "true", "yes", "on",
)
_FFMPEG_TIMING_KINDS = ("measure", "encode", "peak_fallback")
# loudnorm (trong af) là chuẩn-hoá ĐỘNG, cần đủ độ dài để đo ổn định; trên audio NGẮN (<~3s) nó
# under-measure → boost yếu/lệch. Chunk ngắn (vd segment SRT, câu cuối) bị nhất. Fix: CHỈ với chunk
# ngắn, pad silence cho loudnorm đủ buffer (LUFS gating bỏ qua silence → vẫn đo đúng speech) rồi atrim
# cắt lại đúng độ dài. Chunk dài giữ nguyên (không thêm chi phí). Tắt: SHORT_LOUDNORM_SEC=0.
SHORT_LOUDNORM_SEC = float(os.getenv("SHORT_LOUDNORM_SEC", "3.0"))
# SHARED-GAIN loudnorm: loudnorm động PER-CHUNK độc lập → mỗi mảnh một gain → bậc âm lượng giữa các
# mảnh ("to nhỏ" quanh <break time/>: mảnh dưới-câu bị cào bằng về -16 riêng lẻ, mất nhấn nhá + nhiễu
# đo mảnh ngắn). Fix: chunk ĐẦU TIÊN của job đo loudnorm (pass đo ~0.3s, 1 lần/job) → khoá measured_*
# → MỌI chunk áp linear=true (CÙNG 1 gain tĩnh) → đồng nhất by construction, prosody giữ nguyên.
# Worker có thể gửi sẵn `ln_measured` (đồng bộ gain qua NHIỀU batch/box của cùng 1 task) → bỏ qua đo.
# Đo fail/timeout → fallback per-chunk pad-cut dynamic (đúng hành vi cũ). Tắt: LOUDNORM_SHARED=off.
LOUDNORM_SHARED = os.getenv("LOUDNORM_SHARED", "on").strip().lower() not in ("off", "0", "false", "no")
LN_ANCHOR_WAIT_SEC = float(os.getenv("LN_ANCHOR_WAIT_SEC", "120"))
_ATEMPO_RE = re.compile(r"atempo=([0-9.]+)")
_LOUDNORM_RE = re.compile(r"loudnorm=[^,]*")


@dataclass
class _FFmpegTimingCounters:
    calls: int = 0
    failures: int = 0
    queue_wait_s_total: float = 0.0
    queue_wait_s_max: float = 0.0
    service_s_total: float = 0.0
    service_s_max: float = 0.0
    in_flight: int = 0
    max_in_flight: int = 0


@dataclass
class _FFmpegTimingToken:
    kind: str
    queued_at: float
    has_queue: bool
    started_at: Optional[float] = None
    failed: bool = False


_ffmpeg_timing_lock = threading.Lock()
_ffmpeg_timing_counters = {
    kind: _FFmpegTimingCounters() for kind in _FFMPEG_TIMING_KINDS
}


def _ffmpeg_timing_begin(kind: str, has_queue: bool) -> Optional[_FFmpegTimingToken]:
    if not FFMPEG_TIMING_ENABLED:
        return None
    token = _FFmpegTimingToken(kind=kind, queued_at=time.perf_counter(), has_queue=has_queue)
    with _ffmpeg_timing_lock:
        _ffmpeg_timing_counters[kind].calls += 1
    return token


def _ffmpeg_timing_start(token: Optional[_FFmpegTimingToken]) -> None:
    if token is None:
        return
    now = time.perf_counter()
    token.started_at = now
    queue_wait = max(0.0, now - token.queued_at) if token.has_queue else 0.0
    with _ffmpeg_timing_lock:
        counters = _ffmpeg_timing_counters[token.kind]
        counters.queue_wait_s_total += queue_wait
        counters.queue_wait_s_max = max(counters.queue_wait_s_max, queue_wait)
        counters.in_flight += 1
        counters.max_in_flight = max(counters.max_in_flight, counters.in_flight)


def _ffmpeg_timing_finish(token: Optional[_FFmpegTimingToken]) -> None:
    if token is None:
        return
    now = time.perf_counter()
    with _ffmpeg_timing_lock:
        counters = _ffmpeg_timing_counters[token.kind]
        if token.started_at is None:
            # Cancellation/error while waiting for a bounded FFmpeg slot.
            queue_wait = max(0.0, now - token.queued_at) if token.has_queue else 0.0
            counters.queue_wait_s_total += queue_wait
            counters.queue_wait_s_max = max(counters.queue_wait_s_max, queue_wait)
        else:
            service = max(0.0, now - token.started_at)
            counters.service_s_total += service
            counters.service_s_max = max(counters.service_s_max, service)
            counters.in_flight = max(0, counters.in_flight - 1)
        if token.failed:
            counters.failures += 1


@asynccontextmanager
async def _timed_ffmpeg_operation(
    kind: str,
    semaphore: Optional[asyncio.Semaphore] = None,
):
    """Observe an existing FFmpeg operation without changing its scheduling semantics."""
    token = _ffmpeg_timing_begin(kind, has_queue=semaphore is not None)
    acquired = False
    try:
        if semaphore is not None:
            await semaphore.acquire()
            acquired = True
        _ffmpeg_timing_start(token)
        yield token
    except BaseException:
        if token is not None:
            token.failed = True
        raise
    finally:
        if acquired and semaphore is not None:
            semaphore.release()
        _ffmpeg_timing_finish(token)


def _ffmpeg_timing_snapshot() -> dict[str, dict[str, Any]]:
    with _ffmpeg_timing_lock:
        return {
            kind: {
                "calls": counters.calls,
                "failures": counters.failures,
                "queue_wait_s_total": round(counters.queue_wait_s_total, 6),
                "queue_wait_s_max": round(counters.queue_wait_s_max, 6),
                "service_s_total": round(counters.service_s_total, 6),
                "service_s_max": round(counters.service_s_max, 6),
                "in_flight": counters.in_flight,
                "max_in_flight": counters.max_in_flight,
            }
            for kind, counters in _ffmpeg_timing_counters.items()
        }


def _atempo_product(af: str) -> float:
    """Tích các atempo trong chuỗi af (để tính độ dài sau khi tăng/giảm tốc). 1.0 nếu không có."""
    prod = 1.0
    for m in _ATEMPO_RE.finditer(af or ""):
        try:
            v = float(m.group(1))
            if v > 0:
                prod *= v
        except ValueError:
            pass
    return prod if prod > 0 else 1.0


def _expected_tokens(text: str) -> int:
    """Kỳ vọng audio-token cho text (codec 25 fps), TÍNH THEO SCRIPT (rate đo thực, xem TOK_PER_CHAR_*).
    Dùng cho cap động, 'chặn dưới' early-EOS, và sub-split gate. Rate hơi cao = an toàn (bắn sớm)."""
    han = len(_RE_HAN.findall(text))
    kana = len(_RE_KANA.findall(text))
    hangul = len(_RE_HANGUL.findall(text))
    arabic = len(_RE_ARABIC.findall(text))
    thai = len(_RE_THAI.findall(text))
    deva = len(_RE_DEVANAGARI.findall(text))
    other = max(0, len(text) - (han + kana + hangul + arabic + thai + deva))
    return int(
        han * TOK_PER_CHAR_HAN + kana * TOK_PER_CHAR_KANA + hangul * TOK_PER_CHAR_HANGUL
        + arabic * TOK_PER_CHAR_ARABIC + thai * TOK_PER_CHAR_THAI + deva * TOK_PER_CHAR_DEVANAGARI
        + other * TOK_PER_CHAR_LATIN
    )


def _estimate_max_new_tokens(text: str) -> int:
    """Cap động theo độ dài/script chunk: đủ dư cho audio thật, chặn runaway sát nhu cầu thực."""
    cap = int(_expected_tokens(text) * MAX_NEW_TOKENS_SAFETY) + MAX_NEW_TOKENS_BASE
    return max(MAX_NEW_TOKENS_FLOOR, min(MAX_NEW_TOKENS_CEIL, cap))


def _escalated_max_new_tokens(context: Optional[list] = None) -> int:
    """Trần token RỘNG NHẤT cho lần re-render: single-turn → ceil model (2048); chunk có context →
    sát KV (kv_capacity − reserve input). Cho chunk dài bị cắt ở attempt 1 có chỗ đọc trọn lần sau."""
    if context:
        return max(MAX_NEW_TOKENS_FLOOR, min(HIGGS_DEFAULT_MAX_NEW_TOKENS, KV_CACHE_CAPACITY - KV_INPUT_RESERVE))
    return MAX_NEW_TOKENS_CEIL


# Ranh giới câu cho sub-split (giữ delimiter). Ưu tiên: kết câu → mệnh đề/khoảng trắng → cắt cứng.
_SUBSPLIT_PRIMARY_RE = re.compile(r"(?<=[。．！？!?;；\n])")
_SUBSPLIT_SECONDARY_RE = re.compile(r"(?<=[、，,：:）)」』】\s])")


def _subsplit_pack(units: list[str], target: int) -> list[str]:
    """Gói các unit liên tiếp thành sub-part sao cho _expected_tokens mỗi part ≤ target."""
    parts: list[str] = []
    cur = ""
    for u in units:
        if not u:
            continue
        if cur and _expected_tokens(cur + u) > target:
            parts.append(cur)
            cur = u
        else:
            cur += u
    if cur:
        parts.append(cur)
    return parts


def _subsplit_oversized(unit: str, target: int) -> list[str]:
    """1 câu đơn vẫn > target (câu dài không dấu): tách theo ranh giới phụ, cuối cùng cắt cứng theo
    số ký tự ước tính cho `target` token."""
    out: list[str] = []
    for p in _subsplit_pack([s for s in _SUBSPLIT_SECONDARY_RE.split(unit) if s], target):
        if _expected_tokens(p) <= target or len(p) <= 1:
            out.append(p)
            continue
        rate = _expected_tokens(p) / len(p)            # token/char thực của đúng đoạn này
        n = max(1, int(target / max(0.1, rate)))
        out.extend(p[i:i + n] for i in range(0, len(p), n))
    return out


def _split_for_subchunk(text: str, target: int) -> list[str]:
    """Tách text thành các sub-part ≤ target token, ưu tiên ranh giới câu (script-aware)."""
    units: list[str] = []
    for u in _SUBSPLIT_PRIMARY_RE.split(text):
        if not u:
            continue
        if _expected_tokens(u) > target:
            units.extend(_subsplit_oversized(u, target))
        else:
            units.append(u)
    return _subsplit_pack(units, target)

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


@dataclass
class _LaneAdmissionWaiter:
    lane: str
    sequence: int
    future: asyncio.Future
    granted: bool = False


class SoftReservedLaneAdmission:
    """Unified, work-conserving capacity with a soft short-lane reservation.

    Long work may consume every slot while no short work is waiting. Once short
    work waits, released slots restore at least ``short_reserve`` concurrent
    short permits before admission resumes global FIFO order. There is no
    physical short pool, so every released slot can wake an eligible waiter.
    """

    def __init__(self, capacity: int, short_reserve: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if short_reserve < 0 or short_reserve > capacity:
            raise ValueError("short_reserve must be between 0 and capacity")
        self.capacity = capacity
        self.short_reserve = short_reserve
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._short_waiters: deque[_LaneAdmissionWaiter] = deque()
        self._regular_waiters: deque[_LaneAdmissionWaiter] = deque()
        self._inflight_total = 0
        self._inflight_by_lane: dict[str, int] = {
            "short": 0,
            "long": 0,
            "default": 0,
        }
        self._waiting_by_lane: dict[str, int] = {
            "short": 0,
            "long": 0,
            "default": 0,
        }

    def _queue_for_lane(self, lane: str) -> deque[_LaneAdmissionWaiter]:
        return self._short_waiters if lane == "short" else self._regular_waiters

    def _next_waiter_locked(self) -> Optional[_LaneAdmissionWaiter]:
        short = self._short_waiters[0] if self._short_waiters else None
        regular = self._regular_waiters[0] if self._regular_waiters else None
        if short is None:
            return regular
        if regular is None:
            return short
        if self._inflight_by_lane.get("short", 0) < self.short_reserve:
            return short
        return short if short.sequence < regular.sequence else regular

    def _dispatch_locked(self) -> None:
        while self._inflight_total < self.capacity:
            waiter = self._next_waiter_locked()
            if waiter is None:
                return
            queue = self._queue_for_lane(waiter.lane)
            popped = queue.popleft()
            if popped is not waiter:
                raise RuntimeError("lane admission queue order corrupted")
            self._waiting_by_lane[waiter.lane] = max(
                0, self._waiting_by_lane.get(waiter.lane, 0) - 1
            )
            self._inflight_total += 1
            self._inflight_by_lane[waiter.lane] = (
                self._inflight_by_lane.get(waiter.lane, 0) + 1
            )
            waiter.granted = True
            waiter.future.set_result(None)

    async def _cancel_waiter(self, waiter: _LaneAdmissionWaiter) -> None:
        async with self._lock:
            if waiter.granted:
                self._inflight_total -= 1
                self._inflight_by_lane[waiter.lane] -= 1
                waiter.granted = False
            else:
                queue = self._queue_for_lane(waiter.lane)
                try:
                    queue.remove(waiter)
                except ValueError:
                    pass
                else:
                    self._waiting_by_lane[waiter.lane] = max(
                        0, self._waiting_by_lane.get(waiter.lane, 0) - 1
                    )
            self._dispatch_locked()

    async def acquire(self, lane: str) -> str:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        async with self._lock:
            self._sequence += 1
            waiter = _LaneAdmissionWaiter(
                lane=lane,
                sequence=self._sequence,
                future=future,
            )
            self._queue_for_lane(lane).append(waiter)
            self._waiting_by_lane[lane] = self._waiting_by_lane.get(lane, 0) + 1
            self._dispatch_locked()
        try:
            await asyncio.shield(future)
        except BaseException:
            await asyncio.shield(self._cancel_waiter(waiter))
            raise
        return "unified"

    async def release(self, lane: str) -> None:
        async with self._lock:
            if (
                self._inflight_total < 1
                or self._inflight_by_lane.get(lane, 0) < 1
            ):
                raise RuntimeError(f"lane admission release without permit: {lane}")
            self._inflight_total -= 1
            self._inflight_by_lane[lane] -= 1
            self._dispatch_locked()

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "capacity": self.capacity,
                "short_reserve": self.short_reserve,
                "inflight_total": self._inflight_total,
                "inflight_by_lane": dict(self._inflight_by_lane),
                "waiting_by_lane": dict(self._waiting_by_lane),
            }


@dataclass(frozen=True)
class LaneAdmissionPermit:
    pool: str
    semaphore: Optional[asyncio.Semaphore] = None


app = FastAPI(title="SGLang TTS API", version="0.2.0")

chunk_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)
short_chunk_semaphore = asyncio.Semaphore(SHORT_RESERVED_CHUNKS) if SHORT_RESERVED_CHUNKS else None
long_chunk_semaphore = asyncio.Semaphore(LONG_CONCURRENT_CHUNKS)
soft_reserved_lane_admission = SoftReservedLaneAdmission(
    MAX_CONCURRENT_CHUNKS, SHORT_RESERVED_CHUNKS
)
# Gauge quan sát lane (expose qua /health): bao nhiêu chunk đang GIỮ slot / đang CHỜ slot mỗi lane.
# RCA 2026-07-10: /health cũ mù lane (outstanding gộp) → không thấy lane short nghẹt dù cụm "rảnh".
lane_inflight: dict[str, int] = {"short": 0, "long": 0, "default": 0}
lane_waiting: dict[str, int] = {"short": 0, "long": 0, "default": 0}
# Physical admission pool thực tế. Trong dual mode, short mượn long sẽ tăng
# ``dual_long``; trong soft_reserved, mọi permit nằm trong một pool unified.
lane_admission_inflight: dict[str, int] = {
    "dual_default": 0,
    "dual_short_reserved": 0,
    "dual_long": 0,
    "unified": 0,
}
# Bound riêng cho ffmpeg post-proc (boost-on-bridge) → KHÔNG tranh hết core với generation feed-GPU.
ffmpeg_post_semaphore = asyncio.Semaphore(FFMPEG_POST_CONCURRENCY)
cache_locks: dict[str, asyncio.Lock] = {}
cache_locks_guard = asyncio.Lock()
jobs: dict[str, "TTSJob"] = {}
jobs_lock = asyncio.Lock()
outstanding_chunks = 0
outstanding_chunks_lock = asyncio.Lock()
cleanup_task: Optional[asyncio.Task] = None
sglang_http_client: Optional[httpx.AsyncClient] = None
sglang_http_client_lock = asyncio.Lock()


class TTSRequest(BaseModel):
    chunks: list[str]
    ref_audio_url: str
    ref_text: str
    format: str = "mp3"
    # P1 boost-on-bridge: chuỗi ffmpeg -af (atempo+boost) worker tính sẵn (single source of truth ở
    # speaker-boost.ts). Có giá trị → áp khi encode WAV→MP3 từng chunk; None → hành vi cũ y nguyên.
    af_filter: Optional[str] = None
    # Shared-gain loudnorm: measured values {i,tp,lra,thresh,offset} worker lấy từ batch đầu (status
    # `ln_measured`) truyền cho batch sau → CÙNG gain tĩnh qua mọi batch/box của 1 task. None → job
    # tự anchor trên chunk đầu (xem _shared_loudnorm_af).
    ln_measured: Optional[dict] = None
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
    # Completeness annotations: set bởi _call_sglang, dùng bởi best-effort retry. quality_issue None
    # = audio hoàn chỉnh; 'early_eos' (đọc thiếu) | 'runaway' (đuôi câm) | 'silent' (câm toàn phần).
    quality_issue: Optional[str] = None
    is_silent: bool = False
    expected_tokens: int = 0


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
    # Chunk hoàn thành nhưng phải lấy BEST-EFFORT (issue early_eos/runaway/silent vẫn còn sau khi
    # đã re-render hết lượt). Đếm để theo dõi chất lượng / calib ngưỡng; không tính là fail.
    chunks_degraded: int = 0
    input_chars: int = 0
    lane: str = "default"
    in_flight_limit: Optional[int] = None
    detail: Optional[str] = None
    chunk_paths: Optional[list[Path]] = None
    chunk_media_type: Optional[str] = None
    transcript: str = ""
    audio_cache_hit: Optional[bool] = None
    cleanup_paths: Optional[list[Path]] = None
    active_audio_streams: int = 0
    audio_stream_completed_at: Optional[float] = None
    audio_streamed_chunks: Optional[set[int]] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    engine_time_s: float = 0.0
    # Shared-gain loudnorm per job: future do chunk ĐẦU claim (đo anchor), các chunk khác await;
    # ln_measured expose qua status để worker thread sang batch sau (xem _shared_loudnorm_af).
    ln_future: Optional[asyncio.Future] = None
    ln_measured: Optional[dict] = None
    # Thời điểm chunk ĐẦU TIÊN acquire được lane slot (job thật sự chạy). None = còn xếp hàng.
    # Expose qua _job_payload để worker watchdog phân biệt "queued" vs "running không tiến".
    started_at: Optional[float] = None


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


async def _acquire_lane_slot(lane: str) -> LaneAdmissionPermit:
    """Acquire one lane slot and return its physical admission permit.

    Default ``dual`` mode preserves the existing two-semaphore behavior exactly:
    lane "short" được ƯU TIÊN SHORT_RESERVED_CHUNKS slot dành riêng nhưng KHÔNG bị nhốt ở đó:
    hết slot riêng mà pool long còn rảnh thì MƯỢN slot long (RCA 2026-07-10: hard-partition
    4 slot làm job short chờ >120s trong khi 92 slot long rảnh → worker watchdog tưởng bridge
    treo, failover xoay vòng cả cụm). Chiều ngược lại giữ nguyên: long KHÔNG mượn slot short
    (đó mới là ý nghĩa "reserved" — chống long-stampede đè latency short).
    Check `locked()` rồi acquire ngay là an toàn trong asyncio single-thread: giữa check và
    fast-path decrement của acquire() không có await point nên không ai chen được.

    Opt-in ``soft_reserved`` mode uses one work-conserving capacity controller.
    """
    if LANE_ADMISSION_MODE == "soft_reserved":
        pool = await soft_reserved_lane_admission.acquire(lane)
        return LaneAdmissionPermit(pool=pool)
    if lane == "short" and short_chunk_semaphore is not None:
        if not short_chunk_semaphore.locked():
            await short_chunk_semaphore.acquire()
            return LaneAdmissionPermit(
                pool="dual_short_reserved",
                semaphore=short_chunk_semaphore,
            )
        if not long_chunk_semaphore.locked():
            await long_chunk_semaphore.acquire()
            return LaneAdmissionPermit(
                pool="dual_long",
                semaphore=long_chunk_semaphore,
            )
        await short_chunk_semaphore.acquire()
        return LaneAdmissionPermit(
            pool="dual_short_reserved",
            semaphore=short_chunk_semaphore,
        )
    sem = _lane_semaphore(lane)
    await sem.acquire()
    pool = (
        "dual_long"
        if sem is long_chunk_semaphore and SHORT_RESERVED_CHUNKS
        else "dual_default"
    )
    return LaneAdmissionPermit(pool=pool, semaphore=sem)


async def _release_soft_reserved_lane_slot(lane: str) -> None:
    """Return a unified permit even if its holder is cancelled during cleanup.

    Unlike ``Semaphore.release()``, the controller release awaits its internal
    lock. Run it in a shielded task so cancellation is delayed, not swallowed,
    until the permit has definitely been returned.
    """
    release_task = asyncio.create_task(soft_reserved_lane_admission.release(lane))
    pending_cancel: Optional[asyncio.CancelledError] = None
    while not release_task.done():
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError as exc:
            if release_task.cancelled():
                raise
            if pending_cancel is None:
                pending_cancel = exc
    release_task.result()
    if pending_cancel is not None:
        raise pending_cancel


@asynccontextmanager
async def _lane_slot(lane: str):
    """Context manager giữ 1 lane slot + cập nhật gauge lane_waiting/lane_inflight cho /health."""
    lane_waiting[lane] = lane_waiting.get(lane, 0) + 1
    try:
        permit = await _acquire_lane_slot(lane)
    finally:
        lane_waiting[lane] = max(0, lane_waiting.get(lane, 0) - 1)
    lane_inflight[lane] = lane_inflight.get(lane, 0) + 1
    lane_admission_inflight[permit.pool] = (
        lane_admission_inflight.get(permit.pool, 0) + 1
    )
    try:
        yield
    finally:
        lane_inflight[lane] = max(0, lane_inflight.get(lane, 0) - 1)
        lane_admission_inflight[permit.pool] = max(
            0, lane_admission_inflight.get(permit.pool, 0) - 1
        )
        if permit.semaphore is not None:
            permit.semaphore.release()
        else:
            await _release_soft_reserved_lane_slot(lane)


def _job_in_flight_limit(active_same_lane_jobs: int, chunks_total: int) -> int:
    """Chọn quota tĩnh khi job bắt đầu; không đánh thức hàng nghìn coroutine khi tải cao.

    FAIR_SHARE (opt-in): chia đều cap cho số job cùng lane, kẹp trong
    [JOB_QUOTA_MIN, JOB_QUOTA_MAX]. Cụm đo được p50 chỉ ~9 chunk/box nên quota
    cố định 10 để GPU rảnh; chia đều cho job ĐANG có mới tận dụng được lúc vắng
    mà vẫn co lại khi đông. Trần trên = 24 vì worker gửi tối đa 24 chunk/job
    (OMNIVOICE_TTS_BATCH_SIZE); multi-turn cap 40 nhưng chạy TUẦN TỰ nên quota
    không áp dụng. Quota vẫn chốt lúc job bắt đầu (dính khi tải đổi đột ngột),
    chấp nhận được vì job sống ~3s nên quota cũ tự hết hạn nhanh.
    """
    if JOB_QUOTA_FAIR_SHARE:
        share = MAX_CONCURRENT_CHUNKS // max(1, active_same_lane_jobs)
        configured_limit = max(JOB_QUOTA_MIN, min(JOB_QUOTA_MAX, share))
    else:
        configured_limit = MAX_IN_FLIGHT_CHUNKS_PER_JOB
        if active_same_lane_jobs <= MAX_BURST_ACTIVE_JOBS:
            configured_limit = MAX_BURST_IN_FLIGHT_CHUNKS_PER_JOB
    return max(1, min(configured_limit, chunks_total))


def _active_same_lane_job_count(lane: str) -> int:
    """Đếm job có nhu cầu cùng lane; caller production phải đang giữ jobs_lock."""
    return sum(
        1
        for candidate in jobs.values()
        if candidate.lane == lane and candidate.status in {"queued", "running"}
    )


async def _mark_job_started(request_id: str) -> None:
    """Chunk ĐẦU TIÊN của job acquire được lane slot → job thật sự bắt đầu chạy.

    Trước đó status giữ nguyên "queued" (đang xếp hàng lane / chuẩn bị reference) để worker
    watchdog phân biệt được "chờ đến lượt" (nới grace, đừng failover) với "đang chạy mà không
    tiến" (bridge treo thật → failover). RCA 2026-07-10: bản cũ set "running" ngay khi
    background task khởi động làm job xếp hàng hiện ra y hệt job treo.
    """
    async with jobs_lock:
        job = jobs.get(request_id)
        if job is not None and job.status == "queued":
            job.status = "running"
            job.started_at = time.time()
            job.updated_at = job.started_at


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


def _looks_degenerate_ref_text(text: str) -> bool:
    """ref_text bị STT hallucinate: 1 ký tự lặp hàng trăm lần ('。'×387 đo thực 2026-07-03).
    Run >= 20 cùng 1 ký tự không bao giờ là transcript thật của ref audio <=10s. ref_text rác
    làm generation kẹt 0-progress chiếm GPU ~10' mỗi attempt → chặn ngay từ cửa (400 fail-fast)."""
    run = 1
    for prev, cur in zip(text, text[1:]):
        run = run + 1 if prev == cur else 1
        if run >= 20:
            return True
    return False


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
    if _looks_degenerate_ref_text(req.ref_text):
        raise HTTPException(
            status_code=400,
            detail="ref_text looks degenerate (repeated-character spam from STT); refusing job.",
        )

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


def _short_loudnorm_padcut(af: str, wav_bytes: bytes) -> str:
    """Fallback single-pass: chunk NGẮN (<SHORT_LOUDNORM_SEC) pad silence cho loudnorm đủ buffer rồi
    atrim cắt lại. Duration FREE từ WAV header; atempo (nếu có) làm co/giãn → tính theo post-atempo."""
    if SHORT_LOUDNORM_SEC > 0 and "loudnorm" in af:
        d = _wav_seconds(wav_bytes)
        atempo = _atempo_product(af)
        post_dur = d / atempo
        if 0 < post_dur < SHORT_LOUDNORM_SEC:
            pad_to = SHORT_LOUDNORM_SEC * atempo
            return f"apad=whole_dur={pad_to:.3f},{af},atrim=0:{post_dur:.3f},asetpts=N/SR/TB"
    return af


# --- Shared-gain loudnorm (KHOÁ 1 gain tĩnh cho cả job/task) + outlier rescue ---
# Khác bản two-pass cũ (đã xoá ở d930983: đo PER-CHUNK → vẫn mỗi chunk một gain): ở đây đo ĐÚNG 1 LẦN
# trên chunk anchor rồi mọi chunk áp CÙNG volume=dB → hết bậc "to nhỏ" giữa các mảnh <break time/> /
# segment SRT ngắn, prosody giữ nguyên. RIÊNG chunk sinh LỆCH HẲN mức (generation lỗi — đo thực file
# khách: mảnh -38 LUFS cạnh mảnh -14, dropout 20-25 LU) thì shared gain sẽ GIỮ NGUYÊN dropout →
# screen RMS rẻ so với anchor, lệch > LN_OUTLIER_DB thì đo riêng + normalize riêng về target (rescue).
LN_OUTLIER_DB = float(os.getenv("LN_OUTLIER_DB", "6"))
# Cho phép gain vượt TP-cap tối đa chừng này dB — alimiter cuối chain bắt phần đỉnh (đúng vai trò nó
# làm dưới loudnorm động cũ). Cap cứng theo TP anchor làm CẢ job nhỏ đi khi anchor có peak cao (đo
# thực vbee: mean -20.4 thay vì -16). 0 = cap cứng như cũ.
LN_TP_OVER_DB = float(os.getenv("LN_TP_OVER_DB", "6"))


def _ln_values_valid(m: Optional[dict]) -> Optional[dict]:
    """Validate measured values {i,tp,lra,thresh,offset}: đủ key, hữu hạn, i trong (-70, 0).
    Giữ thêm `rms` (dBFS của anchor, optional) nếu hợp lệ — dùng cho screen outlier xuyên batch."""
    if not isinstance(m, dict):
        return None
    try:
        vals = {k: float(m[k]) for k in ("i", "tp", "lra", "thresh", "offset")}
    except (KeyError, TypeError, ValueError):
        return None
    if any(not math.isfinite(v) for v in vals.values()):
        return None
    if not (-70.0 < vals["i"] < 0.0):
        return None
    try:
        rms = float(m["rms"])
        if math.isfinite(rms) and -120.0 < rms < 0.0:
            vals["rms"] = rms
    except (KeyError, TypeError, ValueError):
        pass
    return vals


def _wav_rms_dbfs(wav_bytes: bytes) -> Optional[float]:
    """RMS dBFS của WAV 16-bit (mono/stereo) — screen outlier RẺ (thuần Python, không ffmpeg).
    None nếu không decode được / không phải s16 / câm."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as r:
            if r.getsampwidth() != 2:
                return None
            frames = r.readframes(r.getnframes())
        frames = frames[: len(frames) - (len(frames) % 2)]
        if not frames:
            return None
        if PCM_STATS_AUDIOOP == "all":
            assert _audioop is not None
            rms = float(_audioop.rms(frames, 2))
        else:
            samples = array.array("h")
            samples.frombytes(frames)
            acc = 0
            for s in samples:
                acc += s * s
            rms = math.sqrt(acc / len(samples))
        if rms <= 0:
            return None
        return 20.0 * math.log10(rms / 32768.0)
    except Exception:
        return None


def _apply_linear_af(af: str, m: dict) -> str:
    """Thay loudnorm=... bằng `volume=<gain>dB` — gain TĨNH tính từ measured của ANCHOR.

    KHÔNG dùng loudnorm linear=true với measured của anchor: khi CHUNK KHÁC vi phạm điều kiện nội bộ
    (measured_TP+gain > TP hoặc LRA > target) ffmpeg LẶNG LẼ rơi về dynamic — mà seed measured lại là
    của anchor → gain rác (đo thực: có chunk ra -49 LUFS). volume=dB thì tất định tuyệt đối cho mọi
    chunk; peak lệch đã có alimiter cuối chain (đúng vai trò của nó trong preset el)."""
    ln = _LOUDNORM_RE.search(af)
    target_i, target_tp = -16.0, -1.5
    if ln:
        mi = re.search(r"I=(-?[0-9.]+)", ln.group(0))
        mtp = re.search(r"TP=(-?[0-9.]+)", ln.group(0))
        if mi:
            target_i = float(mi.group(1))
        if mtp:
            target_tp = float(mtp.group(1))
    # Gain đưa integrated về target (+ offset pass-1); cap theo true-peak nhưng NỚI LN_TP_OVER_DB
    # (alimiter cuối chain bắt phần vượt) — cap cứng làm cả job nhỏ đi khi anchor peaky.
    gain = target_i - m["i"] + m["offset"]
    gain = min(gain, target_tp - m["tp"] + LN_TP_OVER_DB)
    return _LOUDNORM_RE.sub(f"volume={gain:.2f}dB", af, count=1)


async def _measure_loudnorm(af: str, wav_bytes: bytes) -> Optional[dict]:
    """Pass đo: chạy chain với loudnorm:print_format=json → parse input loudness. Chunk ngắn (<3s
    post-atempo) pad silence cho phép đo đủ block (gating bỏ silence → vẫn đo đúng speech). None nếu
    đo lỗi / kết quả vô nghĩa (silent/clip)."""
    measure_af = _LOUDNORM_RE.sub(lambda ln: f"{ln.group(0)}:print_format=json", af, count=1)
    if SHORT_LOUDNORM_SEC > 0:
        d = _wav_seconds(wav_bytes)
        atempo = _atempo_product(af)
        if 0 < d / atempo < SHORT_LOUDNORM_SEC:
            measure_af = f"apad=whole_dur={SHORT_LOUDNORM_SEC * atempo:.3f},{measure_af}"
    async with _timed_ffmpeg_operation("measure", ffmpeg_post_semaphore) as timing:
        args = _FFMPEG_NICE_PREFIX + [
            FFMPEG_BIN, "-hide_banner", "-nostats",
            "-threads", str(FFMPEG_THREADS), "-filter_threads", str(FFMPEG_THREADS),
            "-f", "wav", "-i", "pipe:0",
            "-af", measure_af, "-f", "null", "-",
        ]
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate(wav_bytes)
        if timing is not None and process.returncode != 0:
            timing.failed = True
    if process.returncode != 0:
        return None
    text = stderr.decode("utf-8", "replace")
    start, end = text.rfind("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
        return _ln_values_valid({
            "i": data.get("input_i"),
            "tp": data.get("input_tp"),
            "lra": data.get("input_lra"),
            "thresh": data.get("input_thresh"),
            "offset": data.get("target_offset"),
        })
    except (json.JSONDecodeError, ValueError):
        return None


async def _shared_loudnorm_af(request_id: str, req: "TTSRequest", wav_bytes: bytes) -> Optional[str]:
    """Trả chain -af đã khoá gain tĩnh chung cho job (linear=true), hoặc None → caller dùng đường cũ
    (per-chunk pad-cut dynamic). Ưu tiên ln_measured worker gửi (đồng bộ qua nhiều batch/box); không có
    thì chunk ĐẦU claim đo anchor, các chunk khác await (timeout → fallback, không kẹt job)."""
    if not LOUDNORM_SHARED or not req.af_filter or "loudnorm" not in req.af_filter:
        return None

    # Worker gửi sẵn measured values (batch 2+ của cùng task) → áp thẳng, không cần anchor.
    m = _ln_values_valid(req.ln_measured)
    if m:
        async with jobs_lock:
            job = jobs.get(request_id)
            if job is not None and job.ln_measured is None:
                job.ln_measured = m
        return await _shared_or_rescue_af(req.af_filter, m, wav_bytes)

    claimed = False
    async with jobs_lock:
        job = jobs.get(request_id)
        if job is None:
            return None
        if job.ln_future is None:
            job.ln_future = asyncio.get_running_loop().create_future()
            claimed = True
        fut = job.ln_future

    if claimed:
        m = None
        try:
            m = await _measure_loudnorm(req.af_filter, wav_bytes)
            if m:
                # RMS của anchor: mốc screen outlier cho MỌI chunk (đi kèm ln_measured qua status/batch).
                rms = await asyncio.get_running_loop().run_in_executor(None, _wav_rms_dbfs, wav_bytes)
                if rms is not None:
                    m["rms"] = rms
        finally:
            if not fut.done():
                fut.set_result(m)
        if m:
            async with jobs_lock:
                job = jobs.get(request_id)
                if job is not None:
                    job.ln_measured = m
    else:
        try:
            m = await asyncio.wait_for(asyncio.shield(fut), timeout=LN_ANCHOR_WAIT_SEC)
        except asyncio.TimeoutError:
            m = None

    return await _shared_or_rescue_af(req.af_filter, m, wav_bytes) if m else None


async def _shared_or_rescue_af(af: str, m: dict, wav_bytes: bytes) -> str:
    """Shared gain mặc định; RESCUE khi chunk sinh lệch hẳn mức so với anchor (generation lỗi —
    dropout -38 LUFS cạnh -14 trong file khách). Screen bằng RMS thuần Python (rẻ, chain giống nhau
    nên lệch RMS ≈ lệch loudness); vượt LN_OUTLIER_DB → đo riêng chunk + normalize riêng về target
    (cào bằng ĐÚNG chỗ cần cào — dropout thì phải kéo về, không giữ "prosody")."""
    if LN_OUTLIER_DB > 0 and m.get("rms") is not None:
        rms = await asyncio.get_running_loop().run_in_executor(None, _wav_rms_dbfs, wav_bytes)
        if rms is not None and abs(rms - m["rms"]) > LN_OUTLIER_DB:
            own = await _measure_loudnorm(af, wav_bytes)
            if own:
                return _apply_linear_af(af, own)
    return _apply_linear_af(af, m)


async def _wav_to_mp3(wav_bytes: bytes, af: Optional[str] = None, af_final: bool = False) -> bytes:
    if not shutil.which(FFMPEG_BIN):
        raise RuntimeError("ffmpeg is required to convert WAV output to MP3.")

    # Path KHÔNG boost: giữ nguyên perf (không nice/semaphore, không re-filter).
    if not af:
        async with _timed_ffmpeg_operation("encode") as timing:
            process = await asyncio.create_subprocess_exec(
                FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
                "-threads", str(FFMPEG_THREADS), "-filter_threads", str(FFMPEG_THREADS),
                "-f", "wav", "-i", "pipe:0", "-threads", str(FFMPEG_THREADS),
                "-f", "mp3", "pipe:1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate(wav_bytes)
            if timing is not None and process.returncode != 0:
                timing.failed = True
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to convert WAV to MP3: {stderr.decode('utf-8', 'replace')}")
        return stdout

    # Boost-path: nice + semaphore bao CẢ measure + encode (giữ concurrency bounded ~FFMPEG_POST_CONCURRENCY).
    async with _timed_ffmpeg_operation("encode", ffmpeg_post_semaphore) as timing:
        # af_final: chain ĐÃ khoá gain tĩnh shared (linear=true) → dùng verbatim, KHÔNG pad-cut
        # (linear không cần buffer đo). Ngược lại: pad-cut short-clip như bản d9cabb7a.
        eff_af = af if af_final else _short_loudnorm_padcut(af, wav_bytes)
        # Boost-on-bridge: ÉP codec KHỚP worker FINAL_MP3 (libmp3lame 128k 44.1k stereo) → worker concat-copy.
        args = _FFMPEG_NICE_PREFIX + [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
            "-threads", str(FFMPEG_THREADS), "-filter_threads", str(FFMPEG_THREADS),
            "-f", "wav", "-i", "pipe:0",
            "-af", eff_af, "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-threads", str(FFMPEG_THREADS),
            "-f", "mp3", "pipe:1",
        ]
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(wav_bytes)
        if timing is not None and process.returncode != 0:
            timing.failed = True
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


def _wav_concat_all(wavs: list[bytes]) -> Optional[bytes]:
    """Nối TOÀN BỘ WAV (chronological) thành 1 WAV — dùng cho sub-split. None nếu không decode được."""
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
    buf = io.BytesIO()
    with wave.open(buf, "wb") as o:
        o.setparams(params)
        o.writeframes(bytes(frames))
    return buf.getvalue()


_HIGGS_TOKEN_RE = re.compile(r"<\|[a-z_]+:[a-z_]+\|>", re.IGNORECASE)


def _strip_higgs_tags(text: str) -> str:
    """Bỏ MỌI token điều khiển <|cat:tag|> → text trung tính (dựng neutral anchor)."""
    return _HIGGS_TOKEN_RE.sub("", text).strip()


# Tag LẬT GIỌNG dẫn đầu chunk: emotion/style + prosody pitch|expressive. pause/speed/sfx KHÔNG lật.
_VOICE_FLIP_LEAD_RE = re.compile(
    r"^\s*<\|(?:(?:emotion|style):[a-z_]+|prosody:(?:pitch|expressive)_[a-z_]+)\|>",
    re.IGNORECASE,
)


def _starts_with_voice_flip(text: str) -> bool:
    """True nếu chunk BẮT ĐẦU bằng tag lật giọng → c0 (chỉ có ref slot) dễ lật, cần neo ref."""
    return bool(_VOICE_FLIP_LEAD_RE.match(text or ""))


def _sglang_payload(
    chunk_text: str,
    req: TTSRequest,
    ref: ReferenceCacheEntry,
    context: Optional[list[tuple[str, bytes]]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input": (
            _PAUSE_PREFIX_TOKEN + chunk_text
            if HIGGS_PAUSE_PREFIX and not _OWN_LEAD_PAUSE_RE.match(chunk_text)
            else chunk_text
        ),
        # Multi-turn keeps chunks as WAV internally so the rolling context window
        # can be concatenated/trimmed losslessly; output is re-encoded later.
        "response_format": "wav" if (req.multi_turn or req.af_filter) else req.format,
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
    # max_new_tokens nếu request không tự chỉ định:
    #  - CÓ context audio (multi-turn): cap theo KV (kv_capacity − reserve input), KHÔNG dùng 2048
    #    (input context lớn → 2048 tràn KV → HTTP 500). Vẫn rộng hơn cap động cũ (chống cắt cụt);
    #    false-runaway-flag TẮT cho context (xem _call_sglang) vì cap bị KV giới hạn, không đủ rộng
    #    cho heuristic. Context to hơn reserve → reactive refit (xem _call_sglang_with_retry).
    #  - KHÔNG context (single-turn): giữ cap động → chặn runaway sớm (~3-4× nhu cầu).
    if req.max_new_tokens is None:
        if context:
            payload["max_new_tokens"] = max(MAX_NEW_TOKENS_FLOOR, min(HIGGS_DEFAULT_MAX_NEW_TOKENS, KV_CACHE_CAPACITY - KV_INPUT_RESERVE))
        else:
            payload["max_new_tokens"] = _estimate_max_new_tokens(chunk_text)
    # Sampling mặc định khi client không gửi → tránh phân bố khuếch tán (temp=1.0, no top_p/top_k)
    # vốn dễ dẫn tới silence-attractor không emit EOC. setdefault → tôn trọng override của client.
    payload.setdefault("temperature", HIGGS_TEMPERATURE)
    # top_p MẶC ĐỊNH higgs v3 = unset (nucleus OFF) → chỉ gửi khi HIGGS_TOP_P được set qua env,
    # hoặc khi client tự gửi top_p (đã add ở vòng OPTIONAL_SGLANG_FIELDS bên trên).
    if HIGGS_TOP_P is not None:
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


async def _get_sglang_client() -> httpx.AsyncClient:
    """Một connection pool dùng chung cho traffic nội bộ tới SGLang."""
    global sglang_http_client
    if sglang_http_client is not None and not sglang_http_client.is_closed:
        return sglang_http_client
    async with sglang_http_client_lock:
        if sglang_http_client is None or sglang_http_client.is_closed:
            pool_size = MAX_CONCURRENT_CHUNKS + 4  # chừa headroom cho health/retry khi generation đầy lane
            sglang_http_client = httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                limits=httpx.Limits(
                    max_connections=pool_size,
                    max_keepalive_connections=pool_size,
                    keepalive_expiry=30.0,
                ),
            )
    return sglang_http_client


def _wav_peak_dbfs(audio_bytes: bytes) -> Optional[float]:
    """Peak dBFS của PCM16 WAV SGLang; None để caller fallback FFmpeg."""
    if not _is_wav(audio_bytes):
        return None
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as reader:
            if reader.getcomptype() != "NONE":
                return None
            if reader.getsampwidth() != 2:
                return None
            frames = reader.readframes(reader.getnframes())
    except (EOFError, OSError, wave.Error):
        return None

    if not frames:
        return None

    frames = frames[:len(frames) - len(frames) % 2]
    if not frames:
        return None
    if PCM_STATS_AUDIOOP in ("peak", "all"):
        assert _audioop is not None
        peak = _audioop.max(frames, 2)
    else:
        samples = array.array("h")
        samples.frombytes(frames)
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            return None
        peak = max(max(samples), -min(samples))

    if peak <= 0:
        return float("-inf")
    return 20.0 * math.log10(peak / 32768.0)


def _trim_lead_silence(wav_bytes: bytes) -> bytes:
    """Cắt bớt im lặng đầu do HIGGS_PAUSE_PREFIX sinh ra, GIỮ LEAD_SILENCE_KEEP_MS.

    An toàn tuyệt đối theo thiết kế: chỉ cắt tối đa (onset − keep), nên không bao
    giờ chạm vào speech; onset < keep (kể cả 0) → trả nguyên bản. Quét bằng cửa sổ
    5ms, mono s16 (định dạng SGLang trả về); format khác → no-op.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
            if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
                return wav_bytes
            sample_rate = reader.getframerate()
            frames = reader.readframes(reader.getnframes())
    except (EOFError, OSError, wave.Error):
        return wav_bytes

    window = max(2, int(sample_rate * 0.005) * 2)
    threshold = 32768.0 * (10.0 ** (LEAD_SILENCE_DBFS / 20.0))
    onset_bytes = None
    for offset in range(0, len(frames) - window, window):
        chunk = frames[offset : offset + window]
        if PCM_STATS_AUDIOOP in ("peak", "all"):
            assert _audioop is not None
            peak = _audioop.max(chunk, 2)
        else:
            samples = array.array("h")
            samples.frombytes(chunk)
            peak = max(max(samples), -min(samples)) if samples else 0
        if peak > threshold:
            onset_bytes = offset
            break
    if onset_bytes is None:
        return wav_bytes

    keep_bytes = int(sample_rate * LEAD_SILENCE_KEEP_MS / 1000) * 2
    cut_bytes = onset_bytes - keep_bytes
    if cut_bytes <= 0:
        return wav_bytes

    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames[cut_bytes:])
    return out.getvalue()


async def _max_volume_dbfs(audio_bytes: bytes) -> Optional[float]:
    """Peak dBFS: PCM WAV native; format khác fallback ffmpeg volumedetect."""
    if not audio_bytes:
        return None
    native = await asyncio.to_thread(_wav_peak_dbfs, audio_bytes)
    if native is not None:
        return native
    if not shutil.which(FFMPEG_BIN):
        return None
    try:
        async with _timed_ffmpeg_operation("peak_fallback") as timing:
            proc = await asyncio.create_subprocess_exec(
                FFMPEG_BIN, "-hide_banner", "-nostats",
                "-threads", str(FFMPEG_THREADS), "-filter_threads", str(FFMPEG_THREADS),
                "-i", "pipe:0", "-af", "volumedetect", "-f", "null", "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate(audio_bytes)
            if timing is not None and proc.returncode != 0:
                timing.failed = True
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
    context: Optional[list[tuple[str, bytes]]] = None, max_new_override: Optional[int] = None,
    force_wav: bool = False,
) -> ChunkResult:
    payload = _sglang_payload(chunk_text, req, ref, context=context)
    if force_wav:  # sub-split cần WAV để nối PCM lossless (chuyển mp3 1 lần sau khi đã nối)
        payload["response_format"] = "wav"
    if seed_override is not None:
        payload["seed"] = seed_override
    if max_new_override is not None:  # reactive KV-fit (retry sau lỗi KV overflow)
        payload["max_new_tokens"] = max_new_override
    client = await _get_sglang_client()
    response = await client.post(f"{SGLANG_BASE_URL}/v1/audio/speech", json=payload)

    if response.status_code >= 400:
        detail = response.text[:500]
        raise RuntimeError(f"SGLang returned HTTP {response.status_code}: {detail}")

    audio_bytes = _unwrap_sglang_audio(response.content)
    if not audio_bytes:
        raise RuntimeError("SGLang returned empty audio.")

    if (
        HIGGS_PAUSE_PREFIX
        and LEAD_SILENCE_KEEP_MS
        and not _OWN_LEAD_PAUSE_RE.match(chunk_text)
        and _is_wav(audio_bytes)
    ):
        audio_bytes = _trim_lead_silence(audio_bytes)

    result = ChunkResult(
        audio_bytes=audio_bytes,
        prompt_tokens=_header_int(response.headers, "x-prompt-tokens"),
        completion_tokens=_header_int(response.headers, "x-completion-tokens"),
        engine_time_s=_header_float(response.headers, "x-engine-time"),
    )

    # Multi-turn (và sub-split force_wav) giữ WAV (re-encode sang req.format khi ghi file / sau khi nối).
    # af_filter (boost-on-bridge) GIỮ WAV ở đây → boost+encode dồn về _generate_one_chunk (quality-check
    # & grounding chạy trên WAV thô như mode wav cũ; tránh encode 2 lần).
    if not force_wav and not req.multi_turn and not req.af_filter and req.format == "mp3" and not _is_mp3(result.audio_bytes) and _is_wav(audio_bytes):
        result.audio_bytes = await _wav_to_mp3(audio_bytes)

    # Audio rỗng/quá nhỏ = không có audio dùng được → hard-fail để retry (xem _generate_one_chunk).
    if CHUNK_MIN_BYTES and len(result.audio_bytes) < CHUNK_MIN_BYTES:
        raise RuntimeError(
            f"SGLang returned undersized audio ({len(result.audio_bytes)} bytes < {CHUNK_MIN_BYTES})."
        )

    # Completeness: KHÔNG raise (trừ undersized ở trên). Chỉ GẮN quality_issue/is_silent rồi
    # _call_sglang_with_retry re-render (đổi seed / nâng max_new) tới khi sạch, hết lượt thì lấy
    # bản best-effort. Worker yêu cầu MỌI chunk có audio hoàn chỉnh → không bao giờ skip/partial.
    result.expected_tokens = _expected_tokens(chunk_text)
    cap = payload.get("max_new_tokens")

    # CHẶN DƯỚI (early-EOS / đọc THIẾU chữ) — tín hiệu CHÍNH cho "mất chữ". Model phát EOS sớm:
    # audio có tiếng + chưa chạm cap → lọt mọi gate cũ → worker nhận đoạn cụt. Bắt bằng
    # completion_tokens thực << kỳ vọng. Chỉ khi wrapper tự quản token (req.max_new_tokens None) và
    # text đủ dài (expected >= ngưỡng) để tránh false-pos ở chunk ngắn/biến thiên token cao.
    if (
        req.max_new_tokens is None
        and result.expected_tokens >= EARLY_EOS_MIN_EXPECTED_TOKENS
        and 0 < result.completion_tokens < result.expected_tokens * EARLY_EOS_RATIO
    ):
        result.quality_issue = "early_eos"

    # CHẶN TRÊN (runaway EOS): không emit EOS → chạy tới đúng cap (audio size hợp lệ nhưng đuôi câm).
    # Chỉ single-turn (context is None): cap động đủ rộng nên chạm cap = runaway thật. Chunk có context
    # cap bị KV giới hạn → chạm cap có thể là chunk dài hợp lệ, KHÔNG flag (silent runaway vẫn bắt dưới).
    if (
        result.quality_issue is None
        and req.max_new_tokens is None
        and context is None
        and isinstance(cap, int)
        and cap < MAX_NEW_TOKENS_CEIL
        and result.completion_tokens >= cap * 0.95
    ):
        result.quality_issue = "runaway"

    # Backstop câm-toàn-phần (đo âm lượng peak). Set is_silent để best-effort KHÔNG ưu tiên bản câm.
    if CHUNK_SILENCE_MAX_DBFS > -90:
        max_db = await _max_volume_dbfs(result.audio_bytes)
        if max_db is not None and max_db < CHUNK_SILENCE_MAX_DBFS:
            result.is_silent = True
            result.quality_issue = "silent"

    return result


async def _call_sglang_with_retry(
    request_id: str,
    chunk_index: int,
    text: str,
    req: TTSRequest,
    ref: ReferenceCacheEntry,
    context: Optional[list[tuple[str, bytes]]] = None,
    force_wav: bool = False,
) -> ChunkResult:
    last_exc: Optional[BaseException] = None
    max_new_override: Optional[int] = None  # set khi KV overflow → thu nhỏ max_new vừa KV cho retry
    best: Optional[ChunkResult] = None      # ứng viên best-effort (chunk có audio nhưng còn issue)
    best_score = -1.0
    for attempt in range(1, CHUNK_RETRY_ATTEMPTS + 1):
        # Attempt đầu dùng seed tự nhiên; retry ĐỔI seed (early-EOS/runaway/silent thường là fluke
        # sampling, cùng seed hay lặp lại) + NÂNG max_new lên trần (chunk dài bị cắt có chỗ đọc trọn).
        seed_override = None if attempt == 1 else random.randint(1, 2_147_483_647)
        if attempt > 1 and max_new_override is None:
            max_new_override = _escalated_max_new_tokens(context)
        try:
            result = await _call_sglang(
                text, req, ref, seed_override=seed_override, context=context,
                max_new_override=max_new_override, force_wav=force_wav,
            )
        except Exception as exc:
            last_exc = exc
            # KV overflow (input_tokens + max_new_tokens > kv_capacity): đổi seed VÔ DỤNG (deterministic)
            # → thu nhỏ max_new = kv_capacity − input − margin rồi retry cho vừa KV.
            m = _KV_OVERFLOW_RE.search(str(exc))
            if m:
                input_tok, kv_cap = int(m.group(1)), int(m.group(2))
                max_new_override = max(MAX_NEW_TOKENS_FLOOR, kv_cap - input_tok - KV_SAFETY_MARGIN)
                logger.warning(
                    "TTS job %s chunk %d KV overflow (input=%d, kv=%d) → refit max_new=%d",
                    request_id, chunk_index, input_tok, kv_cap, max_new_override,
                )
            if attempt < CHUNK_RETRY_ATTEMPTS:
                logger.warning(
                    "TTS job %s chunk %d attempt %d/%d hard-failed: %s; retrying",
                    request_id, chunk_index, attempt, CHUNK_RETRY_ATTEMPTS, exc,
                )
                await asyncio.sleep(CHUNK_RETRY_BASE_DELAY * attempt)
            continue

        if result.quality_issue is None:
            return result  # audio hoàn chỉnh

        # Còn issue → giữ làm ứng viên best-effort. Điểm = số token ĐỌC THẬT (bản câm = 0 để không
        # bao giờ ưu tiên), cap ở kỳ vọng để runaway-đuôi-câm không thắng bản early-EOS đọc sạch.
        score = 0.0 if result.is_silent else float(
            min(result.completion_tokens, result.expected_tokens or result.completion_tokens)
        )
        if score > best_score:
            best_score, best = score, result
        last_exc = RuntimeError(
            f"chunk quality issue '{result.quality_issue}' "
            f"(completion={result.completion_tokens}, expected={result.expected_tokens})"
        )
        if attempt < CHUNK_RETRY_ATTEMPTS:
            logger.warning(
                "TTS job %s chunk %d attempt %d/%d issue=%s (completion=%d/expected=%d); re-render",
                request_id, chunk_index, attempt, CHUNK_RETRY_ATTEMPTS,
                result.quality_issue, result.completion_tokens, result.expected_tokens,
            )
            await asyncio.sleep(CHUNK_RETRY_BASE_DELAY * attempt)

    # Hết lượt mà chưa sạch: trả BEST-EFFORT (worker luôn cần audio cho mọi chunk). Chỉ hard-fail
    # khi KHÔNG có bản nào dùng được (toàn hard-error/undersized) → để worker retry cả job.
    if best is not None:
        logger.warning(
            "TTS job %s chunk %d: best-effort sau %d lần (issue=%s, completion=%d/expected=%d)",
            request_id, chunk_index, CHUNK_RETRY_ATTEMPTS,
            best.quality_issue, best.completion_tokens, best.expected_tokens,
        )
        return best
    assert last_exc is not None
    raise last_exc


async def _render_chunk(
    request_id: str,
    chunk_index: int,
    text: str,
    req: TTSRequest,
    ref: ReferenceCacheEntry,
    context: Optional[list[tuple[str, bytes]]] = None,
) -> ChunkResult:
    """Render 1 chunk → 1 ChunkResult, GIỮ 1:1. Đường thường: gọi thẳng _call_sglang_with_retry.
    LƯỚI AN TOÀN: nếu _expected_tokens(text) > trần token 1 generation (sgl-omni KHÔNG tự chia →
    vượt trần = cắt cụt âm thầm, reseed vô dụng), tách câu → render từng sub-part → NỐI WAV → 1 file."""
    effective_cap = _escalated_max_new_tokens(context)
    if (
        not SUBSPLIT_ENABLE
        or req.max_new_tokens is not None            # client tự quản token → tôn trọng, không tách
        or _expected_tokens(text) <= effective_cap
    ):
        return await _call_sglang_with_retry(request_id, chunk_index, text, req, ref, context=context)

    target = max(MAX_NEW_TOKENS_FLOOR, int(SUBSPLIT_TARGET_RATIO * effective_cap))
    parts = _split_for_subchunk(text, target)
    if len(parts) <= 1 or len(parts) > SUBSPLIT_MAX_PARTS:
        # Không tách được hữu ích (1 phần) hoặc fan-out quá lớn → 1 call best-effort (reseed/runaway lo).
        logger.warning(
            "TTS job %s chunk %d sub-split bỏ qua (parts=%d, exp=%d > cap=%d, max=%d); 1 call best-effort",
            request_id, chunk_index, len(parts), _expected_tokens(text), effective_cap, SUBSPLIT_MAX_PARTS,
        )
        return await _call_sglang_with_retry(request_id, chunk_index, text, req, ref, context=context)

    logger.info(
        "TTS job %s chunk %d vượt trần (exp=%d > cap=%d) → sub-split %d phần",
        request_id, chunk_index, _expected_tokens(text), effective_cap, len(parts),
    )
    wavs: list[bytes] = []
    prompt_tokens = completion_tokens = 0
    engine_time = 0.0
    sub_ctx = context  # multi-turn: chain các sub-part để giữ giọng liền MẠCH trong chunk
    for part in parts:
        # Mỗi sub-part qua _call_sglang_with_retry (thừa hưởng early_eos/runaway/silent + reseed +
        # best-effort). Hard-fail sau hết retry → raise → CẢ chunk fail (KHÔNG ghi file cụt một phần).
        res = await _call_sglang_with_retry(
            request_id, chunk_index, part, req, ref, context=sub_ctx, force_wav=True,
        )
        wavs.append(res.audio_bytes)
        prompt_tokens += res.prompt_tokens
        completion_tokens += res.completion_tokens
        engine_time += res.engine_time_s
        if context is not None:  # chỉ chain khi multi-turn (single-turn giữ sub-part độc lập)
            sub_ctx = [(part, res.audio_bytes)]

    joined = _wav_concat_all(wavs)
    if joined is None:
        raise RuntimeError(f"sub-split chunk {chunk_index}: nối WAV rỗng (sub-part không decode được)")
    # Trả ĐÚNG format như đường thường: single-turn mp3 → mp3 (re-encode 1 lần sau khi nối, tránh
    # artifact biên frame mp3); còn lại giữ WAV (file-write trong _generate_one_chunk lo nốt).
    if not req.multi_turn and not req.af_filter and req.format == "mp3":
        joined = await _wav_to_mp3(joined)
    return ChunkResult(
        audio_bytes=joined,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        engine_time_s=engine_time,
        expected_tokens=_expected_tokens(text),
    )


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
    # Per-job permit bọc trọn vòng đời chunk để giới hạn cả WAV đang chờ hậu kỳ trong RAM.
    # Lane slot chỉ bọc generation (GPU); ffmpeg hậu kỳ có semaphore riêng.
    async with job_semaphore:
        async with _lane_slot(lane):
            await _mark_job_started(request_id)
            result = await _render_chunk(request_id, chunk_index, text, req, ref, context=context)
        # Multi-turn keeps result.audio_bytes as WAV (for the context window);
        # the chunk FILE still needs req.format. Re-encode only for the write.
        file_bytes = result.audio_bytes
        # af_filter (boost-on-bridge): audio giữ WAV tới đây (guard ở _call_sglang/_render_chunk) →
        # mọi nhánh vào _wav_to_mp3 với af → MP3 đã boost. Không af: chỉ multi-turn còn WAV cần encode
        # (single-turn đã mp3 ở _call_sglang) → _is_wav=False, giữ nguyên hành vi cũ.
        if req.format == "mp3" and _is_wav(file_bytes):
            # Shared-gain loudnorm: khoá 1 gain tĩnh cho cả job (hết bậc "to nhỏ" giữa chunk);
            # None (tắt/đo fail/timeout) → đường cũ per-chunk pad-cut dynamic y nguyên.
            shared_af = await _shared_loudnorm_af(request_id, req, file_bytes) if req.af_filter else None
            if shared_af:
                file_bytes = await _wav_to_mp3(file_bytes, af=shared_af, af_final=True)
            else:
                file_bytes = await _wav_to_mp3(file_bytes, af=req.af_filter)
        # KHÔNG BAO GIỜ ghi bytes sai format ra file chunk: payload sgl-omni không unwrap được
        # (không phải WAV/MP3) mà ghi verbatim thành .mp3 thì /audio sẽ phát tán file rác về
        # worker → ffmpeg worker probe nhầm (raw-VVC) → merge chết. Hard-fail để retry/job fail
        # rõ ràng tại nguồn.
        if req.format == "mp3":
            if not _is_mp3(file_bytes):
                raise RuntimeError(
                    f"chunk {chunk_index}: refusing to write non-MP3 bytes as .mp3 "
                    f"(head={file_bytes[:8].hex()})"
                )
        elif not _is_wav(file_bytes):
            raise RuntimeError(
                f"chunk {chunk_index}: refusing to write non-WAV bytes as .wav "
                f"(head={file_bytes[:8].hex()})"
            )
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
            if result.quality_issue is not None:  # về được nhưng là best-effort (vẫn còn issue)
                job.chunks_degraded += 1
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
        async with _lane_slot(lane):
            await _mark_job_started(request_id)
            result = await _call_sglang_with_retry(request_id, -1, text, req, ref, context=None)
    return result.audio_bytes


async def _run_tts_job(request_id: str, req: TTSRequest) -> None:
    completed_outputs = 0
    try:
        # Status giữ nguyên "queued" cho tới khi chunk đầu acquire được lane slot
        # (_mark_job_started trong _generate_one_chunk/_render_anchor). Chuẩn bị reference
        # + xếp hàng lane đều tính là "queued" → worker watchdog không nhầm với treo.

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
            active_same_lane_jobs = _active_same_lane_job_count(lane)
            in_flight_limit = _job_in_flight_limit(active_same_lane_jobs, len(req.chunks))
            if job is not None:
                job.in_flight_limit = in_flight_limit

        job_semaphore = asyncio.Semaphore(in_flight_limit)

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
            # CHỈ neo khi chunk ĐẦU bắt đầu bằng tag LẬT GIỌNG (emotion/style/pitch/expressive):
            # lúc đó c0 chỉ có ref slot (~ngắn) nên dễ lật. Dựng 1 "chunk vô hình" neutral từ
            # REF (render ref_text qua ref, 1 lần) làm context cho RIÊNG c0 → khoá giọng c0; c1+
            # chain chunk-liền-trước bình thường (giọng đã khoá từ c0 → liền mạch, không lật).
            # chunk1 KHÔNG flip → c0 khỏi neo (ref slot lo), chain thuần. Detect flip ở BRIDGE
            # (worker khỏi gửi cờ); neo render fail → c0 về ref slot.
            anchor: Optional[tuple[str, bytes]] = None
            if req.chunks and _starts_with_voice_flip(req.chunks[0]):
                logger.info("TTS job %s: chunk1 bắt đầu tag lật giọng → neo c0 vào ref-anchor", request_id)
                anchor_text = (ref.transcript or "").strip() or _strip_higgs_tags(req.chunks[0])
                if anchor_text:
                    try:
                        anchor = (
                            anchor_text,
                            await _render_anchor(request_id, anchor_text, req, ref, lane, job_semaphore),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "TTS job %s ref-anchor render failed: %s; c0 dùng ref slot",
                            request_id, exc,
                        )
            prev: Optional[tuple[str, bytes]] = None  # (full text, full WAV) of prior chunk
            for index, text in enumerate(req.chunks):
                # c0: neo ref-anchor nếu chunk1 lật giọng, else chỉ ref slot. c1+: chain chunk-trước.
                if index == 0:
                    ctx = [anchor] if anchor is not None else None
                else:
                    ctx = [prev] if prev else None
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
                asyncio.create_task(
                    _generate_one_chunk(
                        request_id, index, text, req, ref, output_paths[index], lane, job_semaphore,
                    )
                )
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
                now = time.time()
                streamed_chunks = job.audio_streamed_chunks or set()
                if len(streamed_chunks) >= len(output_paths):
                    job.audio_stream_completed_at = now
                job.updated_at = now
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
        "chunks_degraded": job.chunks_degraded,
        "input_chars": job.input_chars,
        "lane": job.lane,
        "format": job.format,
    }
    if job.in_flight_limit is not None:
        payload["in_flight_limit"] = job.in_flight_limit
    if job.detail:
        payload["detail"] = job.detail
    # Shared-gain loudnorm: worker đọc từ batch đầu → truyền `ln_measured` cho batch sau của cùng task
    # (cùng 1 gain tĩnh qua mọi batch/box).
    if job.ln_measured is not None:
        payload["ln_measured"] = job.ln_measured
    # Worker watchdog dùng cặp (status, started_at) để phân biệt "xếp hàng" vs "chạy không tiến".
    if job.started_at is not None:
        payload["started_at"] = job.started_at
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
        if not job.chunk_paths:
            payload.pop("audio_url", None)
            payload["audio_expired"] = True
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
        client = await _get_sglang_client()
        response = await client.get(f"{SGLANG_BASE_URL}/health", timeout=5.0)
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
            if job.status not in {"succeeded", "failed"}:
                continue
            if job.active_audio_streams > 0:
                continue

            if job.audio_stream_completed_at is not None:
                expires_at = job.audio_stream_completed_at + max(0, STREAMED_JOB_TTL_SECONDS)
            else:
                expires_at = job.updated_at + max(0, JOB_TTL_SECONDS)
            if now < expires_at:
                continue

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
    if expired:
        logger.info("Cleaned up %d expired TTS jobs.", len(expired))


async def _finish_tts_job_audio_stream(request_id: str, streamed_indices: list[int]) -> None:
    completed_at = time.time()
    async with jobs_lock:
        job = jobs.get(request_id)
        if job is None:
            return
        if job.active_audio_streams > 0:
            job.active_audio_streams -= 1
        if not streamed_indices:
            return

        if job.audio_streamed_chunks is None:
            job.audio_streamed_chunks = set()
        job.audio_streamed_chunks.update(streamed_indices)

        total_chunks = len(job.chunk_paths or [])
        if (
            job.status == "succeeded"
            and total_chunks > 0
            and len(job.audio_streamed_chunks) >= total_chunks
            and job.audio_stream_completed_at is None
        ):
            job.audio_stream_completed_at = completed_at
            job.updated_at = completed_at
            logger.info(
                "TTS job audio fully streamed; eligible for cleanup: request_id=%s ttl_seconds=%d",
                request_id,
                STREAMED_JOB_TTL_SECONDS,
            )


async def _periodic_cleanup() -> None:
    while True:
        await asyncio.sleep(JOB_CLEANUP_INTERVAL_SECONDS)
        await _cleanup_expired_jobs()


@app.on_event("startup")
async def startup() -> None:
    global cleanup_task
    _ensure_dirs()
    await _get_sglang_client()
    cleanup_task = asyncio.create_task(_periodic_cleanup())


@app.on_event("shutdown")
async def shutdown() -> None:
    global sglang_http_client
    if cleanup_task is not None:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    if sglang_http_client is not None and not sglang_http_client.is_closed:
        await sglang_http_client.aclose()
    sglang_http_client = None


@app.get("/health")
async def health() -> dict[str, Any]:
    _ensure_dirs()
    sglang_ready, sglang_status = await _sglang_health()
    counts = await _job_counts()
    soft_reserved_snapshot = (
        await soft_reserved_lane_admission.snapshot()
        if LANE_ADMISSION_MODE == "soft_reserved"
        else None
    )
    async with outstanding_chunks_lock:
        current_outstanding = outstanding_chunks
    payload = {
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
        "lane_admission_mode": LANE_ADMISSION_MODE,
        "short_reserved_chunks": SHORT_RESERVED_CHUNKS,
        "long_concurrent_chunks": (
            MAX_CONCURRENT_CHUNKS
            if LANE_ADMISSION_MODE == "soft_reserved"
            else LONG_CONCURRENT_CHUNKS
        ),
        "short_request_max_chars": SHORT_REQUEST_MAX_CHARS,
        "short_request_max_chunks": SHORT_REQUEST_MAX_CHUNKS,
        "max_in_flight_chunks_per_job": MAX_IN_FLIGHT_CHUNKS_PER_JOB,
        "max_burst_in_flight_chunks_per_job": MAX_BURST_IN_FLIGHT_CHUNKS_PER_JOB,
        "max_burst_active_jobs": MAX_BURST_ACTIVE_JOBS,
        "busy_backlog_chunks": BUSY_BACKLOG_CHUNKS,
        "outstanding_chunks": current_outstanding,
        "lane_inflight": dict(lane_inflight),
        "lane_waiting": dict(lane_waiting),
        "lane_admission_inflight": dict(lane_admission_inflight),
        "soft_reserved_lane_admission": soft_reserved_snapshot,
        "job_ttl_seconds": JOB_TTL_SECONDS,
        "streamed_job_ttl_seconds": STREAMED_JOB_TTL_SECONDS,
    }
    if FFMPEG_TIMING_ENABLED:
        payload["ffmpeg_timing"] = _ffmpeg_timing_snapshot()
    return payload


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
        requested_items = list(enumerate(all_chunk_paths[chunk_from:chunk_to], start=chunk_from))
        media_type = job.chunk_media_type or _media_type_for_format(job.format)
        transcript = job.transcript
        cache_hit = job.audio_cache_hit
        prompt_tokens = job.prompt_tokens
        completion_tokens = job.completion_tokens
        total_tokens = job.total_tokens
        engine_time_s = job.engine_time_s
        job.active_audio_streams += 1

    # Chỉ phục vụ đoạn LIỀN-MẠCH đã ghi xong tính từ `from`; dừng ở chunk đầu tiên chưa có.
    # Worker tự cộng dồn `fetched += parsed.length` rồi xin tiếp from kế tiếp.
    chunk_items: list[tuple[int, Path]] = []
    for index, path in requested_items:
        if path.exists():
            chunk_items.append((index, path))
        else:
            break

    if not chunk_items:
        await _finish_tts_job_audio_stream(request_id, [])
        if job_status == "succeeded":
            # succeeded mà file không còn → đã bị cleanup theo TTL.
            raise HTTPException(status_code=410, detail="TTS job audio expired.")
        # chunk `from` chưa sinh xong → worker poll lại.
        raise HTTPException(status_code=409, detail="TTS chunk not ready yet.")

    async def stream_length_prefixed():
        streamed_indices: list[int] = []
        try:
            yield len(chunk_items).to_bytes(4, "big")
            for index, path in chunk_items:
                yield path.stat().st_size.to_bytes(4, "big")
                with path.open("rb") as handle:
                    while True:
                        block = await asyncio.to_thread(handle.read, STREAM_CHUNK_SIZE_BYTES)
                        if not block:
                            break
                        yield block
                streamed_indices.append(index)
        finally:
            await _finish_tts_job_audio_stream(request_id, streamed_indices)

    headers = {
        "X-Request-Id": request_id,
        "X-Cache-Hit": str(bool(cache_hit)).lower(),
        "X-Transcript": quote(transcript or "", safe=""),
        "X-Transcript-Encoding": "urlencoded-utf8",
        "X-Chunk-From": str(chunk_from),
        "X-Chunks-Returned": str(len(chunk_items)),
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
