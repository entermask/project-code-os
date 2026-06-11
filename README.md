# Fish Audio S2 Pro FastAPI Wrapper

FastAPI wrapper for Fish Audio S2 Pro served by SGLang-Omni. The API process
does not load the model. It manages auth, reference-audio caching, async jobs,
chunk progress, and proxies each TTS chunk to SGLang's OpenAI-compatible
`/v1/audio/speech` endpoint.

## Architecture

Run two processes on the same machine:

1. SGLang-Omni backend:

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000
```

2. FastAPI wrapper:

```bash
cp .env.example .env
source .env
./scripts/run_api.sh
```

The cached reference-audio paths are passed to SGLang as local `audio_path`
values, so `FISH_AUDIO_CACHE_DIR` must be readable by both processes.

## Install

```bash
./scripts/install.sh
source "$HOME/venvs/fish-audio-api/bin/activate"
```

Install SGLang-Omni separately in its own environment following the upstream
Fish Audio S2 Pro docs.

## Production Tuning

On H200, apply the SGLang-Omni preprocessing patch before starting the backend:

```bash
SGLANG_OMNI_DIR=/workspace/sglang-omni ./scripts/patch_sglang_omni.sh
```

The patch caches reference-audio VQ codes and removes avoidable preprocessing
copies/string token lookups. It matters for voice cloning workloads where many
requests reuse the same reference audio.

For long text, split client chunks around 200 characters and keep
`MAX_CONCURRENT_CHUNKS=64` on H200. In the current H200 run this was the best
measured wrapper setting for `5 x 5000 chars`.

## API

For production client scheduling, chunking, retry, and many-request behavior,
see [CLIENT_INTEGRATION.md](CLIENT_INTEGRATION.md).

### `GET /health`

Returns API status, cache counts, job counts, queue pressure, and SGLang health.

### `POST /v1/tts`

Requires:

```http
Authorization: Bearer <API_TOKEN>
```

Request:

```json
{
  "chunks": ["Xin chao.", "Day la doan thu hai."],
  "ref_audio_url": "https://example.com/reference.wav",
  "ref_text": "Transcript of the reference audio.",
  "format": "wav",
  "speed": 1.0,
  "temperature": 0.8,
  "top_p": 0.9,
  "seed": 1234
}
```

Required fields are `chunks`, `ref_audio_url`, and `ref_text`. The client is
responsible for splitting text into chunks. Supported formats are `wav` and
`mp3`.

Response:

```json
{
  "request_id": "uuid",
  "status": "queued",
  "created_at": 1778730000.0,
  "updated_at": 1778730000.0,
  "status_url": "/v1/tts/jobs/uuid",
  "chunks_total": 2,
  "chunks_completed": 0,
  "chunks_failed": 0,
  "input_chars": 28
}
```

### `GET /v1/tts/jobs/{request_id}`

Poll until `status` is `succeeded` or `failed`. Succeeded jobs include
`audio_url`, `format`, `cache_hit`, and `transcript`.

### `GET /v1/tts/jobs/{request_id}/audio`

Returns `application/octet-stream` containing length-prefixed complete audio
files. By default it returns every chunk. To fetch a completed subset, pass
0-based query parameters:

```http
GET /v1/tts/jobs/{request_id}/audio?from=0&chunks=10
```

`from` must be `>= 0`; `chunks` must be `>= 1` when provided. The simple range
mode is available only after the job has `status=succeeded`.

```text
[4 bytes: chunk_count uint32 BE]
[4 bytes: chunk_0_size uint32 BE][chunk_0 bytes]
[4 bytes: chunk_1_size uint32 BE][chunk_1 bytes]
...
```

Headers include `X-Request-Id`, `X-Chunks-Total`, `X-Audio-Format`,
`X-Chunk-From`, `X-Chunks-Returned`, `X-Cache-Hit`, and URL-encoded
`X-Transcript`.

### `POST /v1/cache/clear`

Clears cached reference audio and transcripts.

## Smoke Test

```bash
curl -sS -X POST "http://127.0.0.1:8001/v1/tts" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": ["Get the trust fund to the bank early."],
    "ref_audio_url": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "format": "wav"
  }'
```

Then poll the returned `status_url` and download `audio_url` when the job
succeeds.
