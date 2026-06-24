import asyncio
import importlib
import os
import struct
import wave
from io import BytesIO

import httpx
import pytest


def wav_bytes() -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 160)
    return buf.getvalue()


def labeled_wav_bytes(label: str) -> bytes:
    data = wav_bytes()
    return data + label.encode("utf-8")


def tone_wav_bytes() -> bytes:
    """~1s sóng vuông biên độ lớn → max_volume ~0dB (KHÔNG câm)."""
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        sample = struct.pack("<h", 30000) + struct.pack("<h", -30000)
        wav.writeframes(sample * 8000)
    return buf.getvalue()


def parse_framed_audio(body: bytes) -> list[bytes]:
    offset = 0
    count = struct.unpack_from(">I", body, offset)[0]
    offset += 4
    chunks = []
    for _ in range(count):
        size = struct.unpack_from(">I", body, offset)[0]
        offset += 4
        chunks.append(body[offset : offset + size])
        offset += size
    return chunks


def single_audio_frame(audio: bytes) -> bytes:
    return struct.pack(">II", 1, len(audio)) + audio


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "test-token")
    monkeypatch.setenv("TTS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SGLANG_BASE_URL", "http://sglang.test")
    monkeypatch.setenv("BUSY_BACKLOG_CHUNKS", "8")
    monkeypatch.setenv("MAX_CONCURRENT_CHUNKS", "2")
    # Audio test tổng hợp rất nhỏ (~364B); tắt ngưỡng min-bytes để không bị coi là chunk hỏng.
    monkeypatch.setenv("CHUNK_MIN_BYTES", "0")
    monkeypatch.setenv("CHUNK_RETRY_BASE_DELAY", "0")
    # Audio test là silence → tắt silence-check ở suite chung để khỏi false-positive.
    monkeypatch.setenv("CHUNK_SILENCE_MAX_DBFS", "-100")
    module = importlib.import_module("app")
    module = importlib.reload(module)

    async def fake_download(ref_audio_url, target):
        target.write_bytes(b"fake reference audio")

    async def fake_call_sglang(chunk_text, req, ref, seed_override=None, **_kwargs):
        assert ref.audio_path.exists()
        assert ref.transcript == "reference transcript"
        return module.ChunkResult(
            audio_bytes=wav_bytes(),
            prompt_tokens=10,
            completion_tokens=20,
            engine_time_s=0.5,
        )

    monkeypatch.setattr(module, "_download_ref_audio", fake_download)
    monkeypatch.setattr(module, "_call_sglang", fake_call_sglang)
    return module


@pytest.fixture()
async def client(api):
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


@pytest.mark.asyncio
async def test_auth_required(client):
    response = await client.post(
        "/v1/tts",
        json={"chunks": ["hello"], "ref_audio_url": "https://example.com/ref.wav", "ref_text": "reference transcript"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_validate_required_ref_text(client):
    response = await client.post(
        "/v1/tts",
        headers=auth_headers(),
        json={"chunks": ["hello"], "ref_audio_url": "https://example.com/ref.wav", "ref_text": ""},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "ref_text is required."


@pytest.mark.asyncio
async def test_submit_poll_and_download_audio(client):
    response = await client.post(
        "/v1/tts",
        headers=auth_headers(),
        json={
            "chunks": ["hello one", "hello two"],
            "ref_audio_url": "https://example.com/ref.wav",
            "ref_text": "reference transcript",
            "format": "wav",
        },
    )
    assert response.status_code == 202
    payload = response.json()
    assert payload["chunks_total"] == 2

    for _ in range(20):
        status = await client.get(payload["status_url"], headers=auth_headers())
        body = status.json()
        if body["status"] == "succeeded":
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("job did not succeed")

    assert body["chunks_completed"] == 2
    assert body["cache_hit"] is False
    assert body["transcript"] == "reference transcript"
    assert body["usage"]["prompt_tokens"] == 20
    assert body["usage"]["completion_tokens"] == 40
    assert body["usage"]["total_tokens"] == 60

    audio = await client.get(body["audio_url"], headers=auth_headers())
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("application/octet-stream")
    assert audio.headers["x-audio-format"] == "audio/wav"
    assert audio.headers["x-prompt-tokens"] == "20"
    assert audio.headers["x-completion-tokens"] == "40"
    chunks = parse_framed_audio(audio.content)
    assert len(chunks) == 2
    assert chunks[0].startswith(b"RIFF")


@pytest.mark.asyncio
async def test_download_audio_range(client, api, monkeypatch):
    async def fake_call_sglang(chunk_text, req, ref, seed_override=None, **_kwargs):
        return api.ChunkResult(audio_bytes=labeled_wav_bytes(chunk_text))

    monkeypatch.setattr(api, "_call_sglang", fake_call_sglang)
    response = await client.post(
        "/v1/tts",
        headers=auth_headers(),
        json={
            "chunks": ["first", "second", "third"],
            "ref_audio_url": "https://example.com/ranged-ref.wav",
            "ref_text": "reference transcript",
            "format": "wav",
        },
    )
    assert response.status_code == 202
    payload = response.json()

    for _ in range(20):
        status = await client.get(payload["status_url"], headers=auth_headers())
        body = status.json()
        if body["status"] == "succeeded":
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("job did not succeed")

    audio = await client.get(f"{body['audio_url']}?from=1&chunks=1", headers=auth_headers())
    assert audio.status_code == 200
    assert audio.headers["x-chunk-from"] == "1"
    assert audio.headers["x-chunks-returned"] == "1"
    assert audio.headers["x-chunks-total"] == "3"
    chunks = parse_framed_audio(audio.content)
    assert len(chunks) == 1
    assert chunks[0].endswith(b"second")


@pytest.mark.asyncio
async def test_streamed_range_downloads_become_eligible_for_cleanup(client, api, monkeypatch):
    import time

    monkeypatch.setattr(api, "STREAMED_JOB_TTL_SECONDS", 0)
    job_dir = api.JOB_DIR / "stream-cleanup-job"
    job_dir.mkdir(parents=True, exist_ok=True)
    paths = [job_dir / f"chunk_{i:05d}.wav" for i in range(2)]
    paths[0].write_bytes(labeled_wav_bytes("first"))
    paths[1].write_bytes(labeled_wav_bytes("second"))

    now = time.time()
    job = api.TTSJob(
        request_id="stream-cleanup-job",
        status="succeeded",
        created_at=now,
        updated_at=now,
        format="wav",
        chunks_total=2,
        chunks_completed=2,
        chunk_paths=paths,
        chunk_media_type="audio/wav",
        cleanup_paths=[job_dir],
    )
    async with api.jobs_lock:
        api.jobs[job.request_id] = job

    url = "/v1/tts/jobs/stream-cleanup-job/audio"
    first = await client.get(f"{url}?from=0&chunks=1", headers=auth_headers())
    assert first.status_code == 200
    await api._cleanup_expired_jobs()
    async with api.jobs_lock:
        assert "stream-cleanup-job" in api.jobs
    assert job_dir.exists()

    second = await client.get(f"{url}?from=1&chunks=1", headers=auth_headers())
    assert second.status_code == 200
    await api._cleanup_expired_jobs()
    async with api.jobs_lock:
        assert "stream-cleanup-job" not in api.jobs
    assert not job_dir.exists()


@pytest.mark.asyncio
async def test_audio_progressive_contiguous_gating(client, api):
    # Dựng thẳng job state để kiểm tra logic /audio (không phụ thuộc background task/timing).
    # chunk 0 và 2 đã ghi, chunk 1 CHƯA (out-of-order completion); chunks_completed đếm = 2.
    import time

    job_dir = api.JOB_DIR / "progjob"
    job_dir.mkdir(parents=True, exist_ok=True)
    paths = [job_dir / f"chunk_{i:05d}.wav" for i in range(3)]
    paths[0].write_bytes(labeled_wav_bytes("c0"))
    paths[2].write_bytes(labeled_wav_bytes("c2"))

    now = time.time()
    job = api.TTSJob(
        request_id="progjob",
        status="running",
        created_at=now,
        updated_at=now,
        format="wav",
        chunks_total=3,
        chunks_completed=2,
        chunk_paths=paths,
        chunk_media_type="audio/wav",
        cleanup_paths=[job_dir],
    )
    async with api.jobs_lock:
        api.jobs["progjob"] = job

    url = "/v1/tts/jobs/progjob/audio"

    # from=0: chunk0 có nhưng chunk1 chưa → chỉ trả đoạn liền-mạch [0,1), KHÔNG nhảy qua khoảng trống.
    r = await client.get(f"{url}?from=0&chunks=3", headers=auth_headers())
    assert r.status_code == 200
    out = parse_framed_audio(r.content)
    assert len(out) == 1 and out[0].endswith(b"c0")
    assert r.headers["x-chunks-returned"] == "1"
    assert r.headers["x-chunks-total"] == "3"

    # from=1: chunk1 chưa ghi xong → 409 dù job đang running (worker sẽ poll lại).
    r = await client.get(f"{url}?from=1&chunks=2", headers=auth_headers())
    assert r.status_code == 409

    # from=2: chunk2 đã có → phục vụ được NGAY khi job còn running (progressive thật).
    r = await client.get(f"{url}?from=2&chunks=1", headers=auth_headers())
    assert r.status_code == 200
    assert parse_framed_audio(r.content)[0].endswith(b"c2")

    # Job đã succeeded nhưng file chunk0 bị cleanup → 410 (expired), không phải 409.
    async with api.jobs_lock:
        api.jobs["progjob"].status = "succeeded"
    paths[0].unlink()
    r = await client.get(f"{url}?from=0&chunks=3", headers=auth_headers())
    assert r.status_code == 410


@pytest.mark.asyncio
async def test_chunk_retry_recovers_transient_error(client, api, monkeypatch):
    calls = {"n": 0}

    async def flaky_call_sglang(chunk_text, req, ref, seed_override=None, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient sglang error")
        return api.ChunkResult(audio_bytes=labeled_wav_bytes(chunk_text))

    monkeypatch.setattr(api, "_call_sglang", flaky_call_sglang)
    response = await client.post(
        "/v1/tts",
        headers=auth_headers(),
        json={
            "chunks": ["only"],
            "ref_audio_url": "https://example.com/retry-ref.wav",
            "ref_text": "reference transcript",
            "format": "wav",
        },
    )
    payload = response.json()
    for _ in range(50):
        body = (await client.get(payload["status_url"], headers=auth_headers())).json()
        if body["status"] in ("succeeded", "failed"):
            break
        await asyncio.sleep(0.01)
    assert body["status"] == "succeeded"
    assert calls["n"] == 2  # 1 lần lỗi + 1 lần thành công


@pytest.mark.asyncio
async def test_persistent_chunk_error_fails_job_and_audio_409(client, api, monkeypatch):
    monkeypatch.setattr(api, "CHUNK_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(api, "CHUNK_RETRY_BASE_DELAY", 0.0)

    async def always_fail(chunk_text, req, ref, seed_override=None):
        raise RuntimeError("persistent sglang error")

    monkeypatch.setattr(api, "_call_sglang", always_fail)
    response = await client.post(
        "/v1/tts",
        headers=auth_headers(),
        json={
            "chunks": ["x"],
            "ref_audio_url": "https://example.com/small-ref.wav",
            "ref_text": "reference transcript",
            "format": "wav",
        },
    )
    payload = response.json()
    for _ in range(50):
        body = (await client.get(payload["status_url"], headers=auth_headers())).json()
        if body["status"] in ("succeeded", "failed"):
            break
        await asyncio.sleep(0.01)
    assert body["status"] == "failed"
    # /audio cho job failed → 409 kèm detail.
    audio_url = f"/v1/tts/jobs/{payload['request_id']}/audio"
    r = await client.get(f"{audio_url}?from=0&chunks=1", headers=auth_headers())
    assert r.status_code == 409
    assert "failed" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_download_audio_range_rejects_out_of_bounds(client):
    response = await client.post(
        "/v1/tts",
        headers=auth_headers(),
        json={
            "chunks": ["only"],
            "ref_audio_url": "https://example.com/oob-ref.wav",
            "ref_text": "reference transcript",
        },
    )
    assert response.status_code == 202
    payload = response.json()

    for _ in range(20):
        status = await client.get(payload["status_url"], headers=auth_headers())
        body = status.json()
        if body["status"] == "succeeded":
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("job did not succeed")

    audio = await client.get(f"{body['audio_url']}?from=1&chunks=1", headers=auth_headers())
    assert audio.status_code == 416


@pytest.mark.asyncio
async def test_reference_audio_cache_hit_on_second_job(client):
    payload = {
        "chunks": ["hello"],
        "ref_audio_url": "https://example.com/ref.wav",
        "ref_text": "reference transcript",
    }
    first = await client.post("/v1/tts", headers=auth_headers(), json=payload)
    assert first.status_code == 202
    second = await client.post("/v1/tts", headers=auth_headers(), json=payload)
    assert second.status_code == 202

    status = await client.get(second.json()["status_url"], headers=auth_headers())
    assert status.json()["status"] == "succeeded"
    assert status.json()["cache_hit"] is True


def test_cache_path_uses_hash_and_suffix(api):
    path = api._audio_cache_path("https://example.com/audio.wav?token=secret")
    assert path.name.endswith(".wav")
    assert "token" not in path.name
    assert len(path.stem) == 64


def test_unwrap_sglang_single_audio_frame(api):
    audio = b"ID3\x04\x00\x00\x00\x00\x00\x23" + b"\xff\xf3" + b"\x00" * 16
    assert api._unwrap_sglang_audio(single_audio_frame(audio)) == audio


def test_sglang_payload_includes_model_only_when_configured(api, monkeypatch):
    req = api.TTSRequest(
        chunks=["hello"],
        ref_audio_url="https://example.com/ref.wav",
        ref_text="reference transcript",
    )
    ref = api.ReferenceCacheEntry(
        audio_path=api.Path("/tmp/ref.wav"),
        transcript="reference transcript",
        audio_cache_hit=False,
    )

    monkeypatch.setattr(api, "SPEECH_MODEL", "")
    payload = api._sglang_payload("hello", req, ref)
    assert "model" not in payload
    assert payload["response_format"] == "mp3"
    assert "speed" not in payload

    monkeypatch.setattr(api, "SPEECH_MODEL", "bosonai/higgs-audio-v3-tts-4b")
    payload = api._sglang_payload("hello", req, ref)
    assert payload["model"] == "bosonai/higgs-audio-v3-tts-4b"


def test_job_payload_includes_audio_url_only_when_succeeded(api):
    job = api.TTSJob(
        request_id="abc",
        status="queued",
        created_at=1.0,
        updated_at=1.0,
        format="wav",
        chunks_total=1,
    )
    assert "audio_url" not in api._job_payload(job)
    job.status = "succeeded"
    job.chunk_paths = [api.Path("/tmp/chunk.wav")]
    assert api._job_payload(job)["audio_url"] == "/v1/tts/jobs/abc/audio"
    job.chunk_paths = None
    payload = api._job_payload(job)
    assert "audio_url" not in payload
    assert payload["audio_expired"] is True


@pytest.mark.asyncio
async def test_max_volume_detects_silence(api):
    silent = await api._max_volume_dbfs(wav_bytes())       # toàn 0 → câm
    loud = await api._max_volume_dbfs(tone_wav_bytes())    # sóng vuông biên độ lớn
    if silent is None or loud is None:
        pytest.skip("ffmpeg không có trong môi trường test")
    assert silent < -50          # câm
    assert loud > -50            # có tiếng


@pytest.mark.asyncio
async def test_retry_varies_seed_on_silent_chunk(api, monkeypatch):
    from pathlib import Path as _Path
    seeds = []
    calls = {"n": 0}

    async def fake_call(chunk_text, req, ref, seed_override=None, **_kwargs):
        calls["n"] += 1
        seeds.append(seed_override)
        if calls["n"] == 1:
            raise RuntimeError("silent audio output (max_volume -inf dBFS); likely EOS-runaway")
        return api.ChunkResult(audio_bytes=labeled_wav_bytes(chunk_text))

    monkeypatch.setattr(api, "_call_sglang", fake_call)
    req = api.TTSRequest(chunks=["hello"], ref_audio_url="https://x/y.wav", ref_text="t")
    ref = api.ReferenceCacheEntry(audio_path=_Path("/tmp/none"), transcript="t", audio_cache_hit=True)

    result = await api._call_sglang_with_retry("job", 1, "hello", req, ref)
    assert result.audio_bytes.endswith(b"hello")
    assert calls["n"] == 2            # câm lần 1 → retry lần 2 thành công
    assert seeds[0] is None           # lần đầu giữ seed mặc định
    assert isinstance(seeds[1], int)  # retry ĐỔI seed


def test_estimate_max_new_tokens_dynamic(api):
    latin = api._estimate_max_new_tokens("a" * 150)
    dense = api._estimate_max_new_tokens("中" * 150)
    assert api.MAX_NEW_TOKENS_FLOOR <= latin <= api.MAX_NEW_TOKENS_CEIL
    assert latin < dense                                  # CJK tốn token/char hơn latin
    assert dense <= api.MAX_NEW_TOKENS_CEIL               # vẫn clamp ≤ default model
    assert api._estimate_max_new_tokens("") == api.MAX_NEW_TOKENS_FLOOR
    assert api._estimate_max_new_tokens("a" * 100000) == api.MAX_NEW_TOKENS_CEIL
    # latin ~150 ký tự: dư ~2x nhu cầu thật (~250 tok) nhưng << 2048
    assert 400 <= latin <= 900


@pytest.mark.asyncio
async def test_call_sglang_annotates_completeness_bands(tmp_path, monkeypatch):
    # _call_sglang KHÔNG raise quality issue nữa: chỉ GẮN quality_issue (early_eos/runaway/silent)
    # để best-effort retry xử lý. Reload riêng để gọi _call_sglang THẬT (fixture `api` thay nó).
    import importlib
    from pathlib import Path as _P
    monkeypatch.setenv("API_TOKEN", "test-token")
    monkeypatch.setenv("TTS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CHUNK_MIN_BYTES", "0")
    monkeypatch.setenv("CHUNK_SILENCE_MAX_DBFS", "-100")  # tắt volumedetect cho test
    monkeypatch.setenv("CHUNK_RETRY_BASE_DELAY", "0")
    api = importlib.reload(importlib.import_module("app"))
    text = "a" * 150
    cap = api._estimate_max_new_tokens(text)
    expected = api._expected_tokens(text)
    assert cap < api.MAX_NEW_TOKENS_CEIL
    assert expected >= api.EARLY_EOS_MIN_EXPECTED_TOKENS

    class FakeResp:
        def __init__(self, ct):
            self.content = wav_bytes()
            self.headers = {"x-completion-tokens": str(ct)}
            self.status_code = 200
        @property
        def text(self):
            return ""

    def make_client(ct):
        class C:
            async def __aenter__(self_): return self_
            async def __aexit__(self_, *a): return False
            async def post(self_, url, json=None): return FakeResp(ct)
        return C()

    req = api.TTSRequest(chunks=[text], ref_audio_url="https://x/y.wav", ref_text="t", format="wav")
    ref = api.ReferenceCacheEntry(audio_path=_P("/tmp/none"), transcript="t", audio_cache_hit=True)

    # đọc THIẾU (early-EOS): completion << kỳ vọng → quality_issue, KHÔNG raise
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda *a, **k: make_client(int(expected * 0.3)))
    res = await api._call_sglang(text, req, ref)
    assert res.quality_issue == "early_eos"
    assert res.expected_tokens == expected

    # band giữa: audio hoàn chỉnh → không issue
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda *a, **k: make_client(int(expected * 0.9)))
    res = await api._call_sglang(text, req, ref)
    assert res.quality_issue is None

    # chạm cap: runaway (đuôi câm) → quality_issue, KHÔNG raise
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda *a, **k: make_client(cap))
    res = await api._call_sglang(text, req, ref)
    assert res.quality_issue == "runaway"


@pytest.mark.asyncio
async def test_retry_returns_best_effort_after_exhaust(api, monkeypatch):
    # Hết lượt re-render mà chunk vẫn còn issue → trả BEST-EFFORT: bản đọc được NHIỀU nhất, và
    # bản CÂM (is_silent) không bao giờ được ưu tiên (điểm 0). Worker luôn nhận audio.
    from pathlib import Path as _Path
    monkeypatch.setattr(api, "CHUNK_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(api, "CHUNK_RETRY_BASE_DELAY", 0.0)
    # (issue, completion_tokens, is_silent) cho 3 lần thử; câm 250-token KHÔNG được chọn.
    attempts = [("early_eos", 120, False), ("silent", 250, True), ("early_eos", 90, False)]
    calls = {"n": 0}

    async def fake_call(chunk_text, req, ref, seed_override=None, **_kwargs):
        i = calls["n"]; calls["n"] += 1
        issue, ct, silent = attempts[i]
        return api.ChunkResult(
            audio_bytes=labeled_wav_bytes(f"try{i}"),
            completion_tokens=ct,
            expected_tokens=300,
            quality_issue=issue,
            is_silent=silent,
        )

    monkeypatch.setattr(api, "_call_sglang", fake_call)
    req = api.TTSRequest(chunks=["x"], ref_audio_url="https://x/y.wav", ref_text="t")
    ref = api.ReferenceCacheEntry(audio_path=_Path("/tmp/none"), transcript="t", audio_cache_hit=True)

    res = await api._call_sglang_with_retry("job", 1, "x", req, ref)
    assert calls["n"] == 3                      # thử hết 3 lần (không sạch lần nào)
    assert res.quality_issue == "early_eos"     # best-effort: vẫn còn issue nhưng có audio
    assert res.completion_tokens == 120         # chọn bản đọc nhiều nhất, BỎ bản câm 250-token
    assert res.audio_bytes.endswith(b"try0")


def test_sglang_payload_sets_sampling_defaults(api):
    from pathlib import Path as _P
    ref = api.ReferenceCacheEntry(audio_path=_P("/tmp/none"), transcript="t", audio_cache_hit=True)
    req = api.TTSRequest(chunks=["hi"], ref_audio_url="https://x/y.wav", ref_text="t")
    p = api._sglang_payload("hello world there", req, ref)
    assert p["temperature"] == api.HIGGS_TEMPERATURE
    assert p["top_k"] == api.HIGGS_TOP_K
    assert "max_new_tokens" in p
    # top_p MẶC ĐỊNH v3 = unset → KHÔNG inject khi HIGGS_TOP_P để trống; có inject khi env set.
    if api.HIGGS_TOP_P is None:
        assert "top_p" not in p
    else:
        assert p["top_p"] == api.HIGGS_TOP_P
    # client override top_p luôn được tôn trọng dù default là unset
    req_tp = api.TTSRequest(chunks=["hi"], ref_audio_url="https://x/y.wav", ref_text="t", top_p=0.85)
    assert api._sglang_payload("hello", req_tp, ref)["top_p"] == 0.85
    # client override được tôn trọng
    req2 = api.TTSRequest(chunks=["hi"], ref_audio_url="https://x/y.wav", ref_text="t", temperature=0.3)
    assert api._sglang_payload("hello", req2, ref)["temperature"] == 0.3


def test_expected_tokens_per_script(api):
    # Rate theo script (đo thực): Han6.0 Kana4.8 Hangul3.8 Arabic3.0 Thai2.5 Deva2.4 latin2.0
    assert api._expected_tokens("中" * 50) == 300   # Han
    assert api._expected_tokens("あ" * 50) == 240   # kana
    assert api._expected_tokens("가" * 50) == 190   # hangul
    assert api._expected_tokens("ا" * 50) == 150    # arabic
    assert api._expected_tokens("ก" * 50) == 125    # thai
    assert api._expected_tokens("क" * 50) == 120    # devanagari
    assert api._expected_tokens("a" * 50) == 100    # latin/default
    # Trung > Nhật > Hàn > Arabic > Thai > Hindi > latin cho cùng số ký tự
    n = 60
    order = [api._expected_tokens(c * n) for c in "中あ가اกक a".replace(" ", "")]
    assert order == sorted(order, reverse=True)


def test_split_for_subchunk_bounds_and_preserves_text(api):
    target = int(api.SUBSPLIT_TARGET_RATIO * api.MAX_NEW_TOKENS_CEIL)
    # Dày + có dấu câu → nhiều phần, mỗi phần ≤ target, KHÔNG mất chữ
    text = "天气好。" * 120                       # ~2400 token >> cap
    parts = api._split_for_subchunk(text, target)
    assert len(parts) >= 2
    assert all(api._expected_tokens(p) <= target for p in parts)
    assert "".join(parts) == text                # text bảo toàn
    # Câu dài KHÔNG dấu câu → vẫn bị chặn dưới cap bằng cắt cứng
    longrun = "私" * 500
    parts2 = api._split_for_subchunk(longrun, target)
    assert all(api._expected_tokens(p) <= target for p in parts2)
    assert "".join(parts2) == longrun


@pytest.mark.asyncio
async def test_render_chunk_passthrough_under_ceiling(api, monkeypatch):
    from pathlib import Path as _Path
    calls = []

    async def fake_retry(request_id, chunk_index, text, req, ref, context=None, force_wav=False):
        calls.append((text, force_wav))
        return api.ChunkResult(audio_bytes=wav_bytes(), completion_tokens=10,
                               expected_tokens=api._expected_tokens(text))

    monkeypatch.setattr(api, "_call_sglang_with_retry", fake_retry)
    ref = api.ReferenceCacheEntry(audio_path=_Path("/tmp/none"), transcript="t", audio_cache_hit=True)
    req = api.TTSRequest(chunks=["x"], ref_audio_url="https://x/y.wav", ref_text="t", format="wav")
    # Dưới trần → 1 call thẳng, KHÔNG force_wav, KHÔNG sub-split
    await api._render_chunk("job", 0, "a" * 100, req, ref)
    assert len(calls) == 1 and calls[0][1] is False


@pytest.mark.asyncio
async def test_render_chunk_subsplits_over_ceiling_and_concats(api, monkeypatch):
    from pathlib import Path as _Path
    calls = []

    async def fake_retry(request_id, chunk_index, text, req, ref, context=None, force_wav=False):
        calls.append((text, force_wav))
        return api.ChunkResult(audio_bytes=wav_bytes(), completion_tokens=10,
                               expected_tokens=api._expected_tokens(text))

    monkeypatch.setattr(api, "_call_sglang_with_retry", fake_retry)
    ref = api.ReferenceCacheEntry(audio_path=_Path("/tmp/none"), transcript="t", audio_cache_hit=True)
    req = api.TTSRequest(chunks=["x"], ref_audio_url="https://x/y.wav", ref_text="t", format="wav")
    # Vượt trần (dense) → nhiều sub-call force_wav=True, nối thành 1 WAV, token cộng dồn
    res = await api._render_chunk("job", 1, "天" * 400, req, ref)   # ~2400 > 2048
    assert len(calls) >= 2
    assert all(fw is True for _, fw in calls)
    assert api._is_wav(res.audio_bytes)
    assert res.completion_tokens == 10 * len(calls)
