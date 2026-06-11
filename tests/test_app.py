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


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "test-token")
    monkeypatch.setenv("FISH_AUDIO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SGLANG_BASE_URL", "http://sglang.test")
    monkeypatch.setenv("BUSY_BACKLOG_CHUNKS", "8")
    monkeypatch.setenv("MAX_CONCURRENT_CHUNKS", "2")
    module = importlib.import_module("app")
    module = importlib.reload(module)

    async def fake_download(ref_audio_url, target):
        target.write_bytes(b"fake reference audio")

    async def fake_call_sglang(chunk_text, req, ref):
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
    async def fake_call_sglang(chunk_text, req, ref):
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
    assert api._job_payload(job)["audio_url"] == "/v1/tts/jobs/abc/audio"
