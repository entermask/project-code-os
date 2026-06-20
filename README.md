# Higgs Audio v3 TTS FastAPI Wrapper

FastAPI wrapper for Higgs Audio v3 TTS served by SGLang-Omni. The API process
does not load the model. It manages auth, reference-audio caching, async jobs,
chunk progress, and proxies each TTS chunk to SGLang's OpenAI-compatible
`/v1/audio/speech` endpoint.

Default backend model: `bosonai/higgs-audio-v3-tts-4b`.

Higgs Audio v3 is released under a research and non-commercial license. Review
the upstream model license before hosted, production, or revenue-generating use:
https://huggingface.co/bosonai/higgs-audio-v3-tts-4b

## Architecture

Run two processes on the same machine:

1. SGLang-Omni backend:

```bash
sgl-omni serve \
  --model-path bosonai/higgs-audio-v3-tts-4b \
  --port 8000
```

2. FastAPI wrapper:

```bash
cp .env.example .env
source .env
./scripts/run_api.sh
```

The wrapper listens on port `8080` by default, matching an SSH tunnel such as
`ssh -p 20182 root@108.250.147.21 -L 8080:localhost:8080 -i ~/.ssh/id_ed25519`.
The cached reference-audio paths are passed to SGLang as local `audio_path`
values, so `TTS_CACHE_DIR` must be readable by both processes.

## Install

```bash
./scripts/install.sh
source "$HOME/venvs/sglang-tts-api/bin/activate"
```

Install SGLang-Omni separately in its own environment following the upstream
Higgs Audio v3 model card or SGLang-Omni docs.

## Backend

`scripts/run_sglang.sh` defaults to Higgs Audio v3:

```bash
MODEL_PATH=bosonai/higgs-audio-v3-tts-4b ./scripts/run_sglang.sh
```

For vLLM-Omni compatible endpoints that require a `model` field in speech
requests, set `SPEECH_MODEL=bosonai/higgs-audio-v3-tts-4b` for the API process.

## Production Tuning

For long text, split client chunks around 150-220 characters. The current
measured production-shaped sweet spot for Higgs Audio v3 is:

```text
MAX_CONCURRENT_CHUNKS=16
MAX_IN_FLIGHT_CHUNKS_PER_JOB=10
BUSY_BACKLOG_CHUNKS=2000
SHORT_RESERVED_CHUNKS=0
```

`MAX_CONCURRENT_CHUNKS` controls active backend generation. `BUSY_BACKLOG_CHUNKS`
only controls how much queued/running chunk work the API accepts before `429`;
raising it does not increase GPU throughput.

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
  "format": "mp3",
  "temperature": 0.8,
  "top_p": 0.9,
  "seed": 1234
}
```

Required fields are `chunks`, `ref_audio_url`, and `ref_text`. The client is
responsible for splitting text into chunks. Supported formats are `wav` and
`mp3`; the default is `mp3`.

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
curl -sS -X POST "http://127.0.0.1:8080/v1/tts" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": ["Get the trust fund to the bank early."],
    "ref_audio_url": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "format": "mp3"
  }'
```

Then poll the returned `status_url` and download `audio_url` when the job
succeeds.
