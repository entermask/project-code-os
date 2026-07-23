from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

import drain_wrapper


def _request(host: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/health",
            "headers": [],
            "client": (host, 1234),
            "server": ("test", 80),
            "scheme": "http",
            "query_string": b"",
        }
    )


def test_loopback_detection() -> None:
    assert drain_wrapper._is_loopback(_request("127.0.0.1"))
    assert drain_wrapper._is_loopback(_request("::1"))
    assert not drain_wrapper._is_loopback(_request("172.30.55.19"))


@pytest.mark.asyncio
async def test_public_drain_blocks_only_new_tts_and_marks_health_unready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drain_file = tmp_path / ".draining"
    drain_file.touch()
    monkeypatch.setattr(drain_wrapper, "DRAIN_FILE", drain_file)

    test_app = FastAPI()

    @test_app.middleware("http")
    async def drain(request: Request, call_next):
        return await drain_wrapper.production_drain(request, call_next)

    @test_app.get("/health")
    async def health():
        return {"status": "ok", "sglang_ready": True, "active_tts_jobs": 1}

    @test_app.post("/v1/tts")
    async def submit():
        return JSONResponse(status_code=202, content={"request_id": "accepted"})

    @test_app.get("/v1/tts/jobs/{request_id}")
    async def status(request_id: str):
        return {"request_id": request_id, "status": "running"}

    public = ASGITransport(app=test_app, client=("172.30.55.19", 5000))
    async with AsyncClient(transport=public, base_url="http://bridge") as client:
        health_response = await client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["sglang_ready"] is False
        assert health_response.json()["backend_sglang_ready"] is True
        assert health_response.json()["draining"] is True

        submit_response = await client.post("/v1/tts")
        assert submit_response.status_code == 429
        assert submit_response.headers["retry-after"] == "1"

        status_response = await client.get("/v1/tts/jobs/existing")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "running"

    loopback = ASGITransport(app=test_app, client=("127.0.0.1", 5001))
    async with AsyncClient(transport=loopback, base_url="http://bridge") as client:
        assert (await client.get("/health")).json()["sglang_ready"] is True
        assert (await client.post("/v1/tts")).status_code == 202
