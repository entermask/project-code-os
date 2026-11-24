"""Operational drain wrapper for zero-loss rolling GPU deployments.

When ``DRAIN_FILE`` exists, public health checks advertise the bridge as
unready and public TTS submissions receive a retryable 429. Existing status and
audio downloads remain available so in-flight jobs can finish. Loopback
requests bypass the drain, which keeps local smoke tests and benchmarks usable.
"""

from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse

from app import app


DRAIN_FILE = Path(os.getenv("DRAIN_FILE", "/root/autodl-tmp/Fish-Audio/.draining"))


def _is_loopback(request: Request) -> bool:
    if request.client is None:
        return False
    try:
        return ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        return False


@app.middleware("http")
async def production_drain(request: Request, call_next):
    draining = DRAIN_FILE.exists()
    public_request = not _is_loopback(request)

    if draining and public_request and request.method == "POST" and request.url.path == "/v1/tts":
        return JSONResponse(
            status_code=429,
            content={"detail": "Bridge is draining for a rolling deployment."},
            headers={"Retry-After": "1", "X-Bridge-Draining": "1"},
        )

    response = await call_next(request)
    if not (draining and public_request and request.method == "GET" and request.url.path == "/health"):
        return response
    if response.status_code != 200:
        return response

    body = b"".join([chunk async for chunk in response.body_iterator])
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return response

    backend_ready = bool(payload.get("sglang_ready"))
    payload["backend_sglang_ready"] = backend_ready
    payload["sglang_ready"] = False
    payload["draining"] = True

    headers = dict(response.headers)
    headers.pop("content-length", None)
    return JSONResponse(
        status_code=response.status_code,
        content=payload,
        headers=headers,
    )
