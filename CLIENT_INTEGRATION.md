# Client Integration Guide

This guide describes how a client should submit many Higgs Audio v3 TTS jobs to
the FastAPI wrapper without overfilling the backend chunk scheduler.

The API has one TTS endpoint. Voice cloning is done by passing `ref_audio_url`
and `ref_text` to `POST /v1/tts`.

## Endpoints

Base URL depends on deployment. With an SSH tunnel such as
`-L 8080:localhost:8080`, use:

```text
http://127.0.0.1:8080
```

All endpoints except `/health` require:

```http
Authorization: Bearer <API_TOKEN>
```

Main endpoints:

```http
GET  /health
POST /v1/tts
GET  /v1/tts/jobs/{request_id}
GET  /v1/tts/jobs/{request_id}/audio
GET  /v1/tts/jobs/{request_id}/audio?from=0&chunks=10
```

## Request Shape

```json
{
  "chunks": ["Text chunk 1.", "Text chunk 2."],
  "ref_audio_url": "https://example.com/reference.wav",
  "ref_text": "Transcript of the reference audio.",
  "format": "mp3",
  "temperature": 0.8,
  "top_p": 0.9,
  "seed": 1234
}
```

Required fields:

```text
chunks
ref_audio_url
ref_text
```

Supported output formats:

```text
wav
mp3
```

## Reference Audio Cache

The wrapper caches reference audio on disk by `ref_audio_url`.

For best cache hit rate:

- Use a stable URL for the same voice.
- Avoid signed URLs that change token/query string on every request.
- Keep `ref_text` accurate for the reference audio.
- Reuse the same `ref_audio_url` across all requests for the same voice.

The wrapper passes the cached local `audio_path` plus `ref_text` to
SGLang-Omni for every chunk request. Do not assume that many different voices
are permanently cached in VRAM. If production traffic has many voices, route by
stable `ref_audio_url` or content hash so repeated voices usually land on the
same node.

## Warm-Up

For a voice that will be used heavily, warm it first with a tiny request:

```json
{
  "chunks": ["Hello."],
  "ref_audio_url": "https://example.com/reference.wav",
  "ref_text": "Transcript of the reference audio.",
  "format": "mp3"
}
```

This downloads the reference audio and creates the wrapper cache entry before a
large burst starts.

## Text Chunking

The client is responsible for splitting input text.

Recommended chunk size:

```text
150-220 characters per chunk
```

Rules:

- Prefer splitting on sentence boundaries.
- Avoid very tiny chunks unless the input is short.
- Avoid very long chunks because they increase tail latency.
- Preserve original order with `document_id`, `page_index`, and `chunk_index`.

## Current Production Scheduler

The measured sweet spot for the current Higgs Audio v3 deployment is:

```text
MAX_CONCURRENT_CHUNKS=16
MAX_IN_FLIGHT_CHUNKS_PER_JOB=10
BUSY_BACKLOG_CHUNKS=2000
SHORT_RESERVED_CHUNKS=0
```

What these mean:

- `MAX_CONCURRENT_CHUNKS`: maximum active backend chunk generations.
- `MAX_IN_FLIGHT_CHUNKS_PER_JOB`: maximum active chunks from one submitted job.
- `BUSY_BACKLOG_CHUNKS`: accepted queued/running chunks before the API returns
  `429`.
- `SHORT_RESERVED_CHUNKS`: reserved short-request lane. It is disabled in the
  current throughput-first config.

For the current production pattern of 10 chunks per logical TTS request,
`BUSY_BACKLOG_CHUNKS=2000` accepts about 200 simultaneous jobs before applying
backpressure. Raising backlog does not make the GPU faster; it only allows more
queued work before `429`.

## Request Classes

Short request:

```text
<= 1000 chars or <= 4 chunks
```

Submit immediately unless `/health` shows high backlog pressure.

Medium request:

```text
1,000-10,000 chars
```

Submit as pages of up to 10 chunks.

Long request:

```text
> 10,000 chars
```

Submit pages in round-robin order across documents. Do not submit hundreds of
chunks for one document while other documents are waiting.

## Fair Long-Job Submission

Do not submit a whole 100k-character document as one 500-chunk API job when many
documents are active.

Instead, submit pages in a round-robin schedule.

Example: 10 documents, each around 100k chars and 500 chunks.

With `LONG_PAGE_SIZE=10` and `LONG_OUTSTANDING_LIMIT=160`, submit only sixteen
pages at a time:

```text
doc_0 chunks 0-9
doc_1 chunks 0-9
doc_2 chunks 0-9
...
doc_15 chunks 0-9
```

That uses:

```text
16 * 10 = 160 outstanding long chunks
```

When one page finishes, submit the next page for the next document:

```text
doc_16 chunks 0-9
doc_0 chunks 10-19
doc_17 chunks 0-9
doc_1 chunks 10-19
...
```

This keeps throughput high while preventing one long document from occupying the
server for too long.

## Job Lifecycle

1. Submit a page:

```http
POST /v1/tts
```

2. Store the returned `request_id`.

3. Poll:

```http
GET /v1/tts/jobs/{request_id}
```

4. When `status=succeeded`, download audio:

```http
GET /v1/tts/jobs/{request_id}/audio
```

5. Store page result under:

```text
document_id
page_index
chunk_start
chunks_returned
```

6. Submit the next page only when the scheduler has available outstanding
budget.

## Audio Download Format

`/audio` returns `application/octet-stream`, not a single concatenated WAV file.

The stream format is length-prefixed:

```text
[4 bytes: chunk_count uint32 big-endian]
[4 bytes: chunk_0_size uint32 big-endian][chunk_0 bytes]
[4 bytes: chunk_1_size uint32 big-endian][chunk_1 bytes]
...
```

Each chunk byte payload is a complete audio file in the requested format.

Do not concatenate WAV bytes directly. For WAV output, either:

- play each chunk sequentially, or
- decode each WAV to PCM and concatenate PCM, then write a new WAV, or
- use a proper audio concat pipeline such as ffmpeg concat.

Range download is available after the job succeeds:

```http
GET /v1/tts/jobs/{request_id}/audio?from=0&chunks=10
GET /v1/tts/jobs/{request_id}/audio?from=10&chunks=10
GET /v1/tts/jobs/{request_id}/audio?from=20&chunks=10
```

This helps with large downloads. It does not improve generation fairness,
because generation must already be complete for that job.

Useful response headers:

```text
X-Request-Id
X-Cache-Hit
X-Chunks-Total
X-Chunk-From
X-Chunks-Returned
X-Audio-Format
X-Prompt-Tokens
X-Completion-Tokens
X-Total-Tokens
X-Engine-Time
```

## Retry Behavior

Handle these responses:

```text
202 Accepted: job accepted
400 Bad Request: invalid payload
401 Unauthorized: invalid API token
409 Conflict: job not finished yet when requesting audio
416 Range Not Satisfiable: audio range starts outside available chunks
429 Too Many Requests: server backlog is busy
500/502/503/504: retry with backoff if safe
```

For `429`, respect `Retry-After` if present. Otherwise retry after 1 second with
jitter.

Recommended retry delay:

```text
delay = min(30s, retry_after_or_1s * 2^attempt) + random(0-500ms)
```

Do not retry a large failed page forever. Put it into a dead-letter queue after
the configured maximum attempts.

## Scheduler Pseudocode

```ts
type DocumentJob = {
  documentId: string;
  refAudioUrl: string;
  refText: string;
  chunks: string[];
  nextChunk: number;
  inFlightChunks: number;
};

const LONG_OUTSTANDING_LIMIT = 160;
const LONG_PAGE_SIZE = 10;
const MAX_PER_DOCUMENT = 10;

let globalLongOutstanding = 0;

async function scheduleLongJobs(queue: DocumentJob[]) {
  while (true) {
    for (const job of queue) {
      if (job.nextChunk >= job.chunks.length) continue;
      if (globalLongOutstanding >= LONG_OUTSTANDING_LIMIT) break;
      if (job.inFlightChunks >= MAX_PER_DOCUMENT) continue;

      const pageSize = Math.min(
        LONG_PAGE_SIZE,
        MAX_PER_DOCUMENT - job.inFlightChunks,
        LONG_OUTSTANDING_LIMIT - globalLongOutstanding,
        job.chunks.length - job.nextChunk,
      );

      const chunkStart = job.nextChunk;
      const pageChunks = job.chunks.slice(chunkStart, chunkStart + pageSize);

      job.nextChunk += pageSize;
      job.inFlightChunks += pageSize;
      globalLongOutstanding += pageSize;

      submitPage(job, chunkStart, pageChunks)
        .catch((error) => {
          handlePageFailure(job, chunkStart, pageChunks, error);
        })
        .finally(() => {
          job.inFlightChunks -= pageSize;
          globalLongOutstanding -= pageSize;
        });
    }

    await sleep(100);
  }
}
```

## TypeScript API Example

```ts
async function submitTtsPage(params: {
  baseUrl: string;
  apiToken: string;
  chunks: string[];
  refAudioUrl: string;
  refText: string;
  format?: "wav" | "mp3";
}) {
  const response = await fetch(`${params.baseUrl}/v1/tts`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${params.apiToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      chunks: params.chunks,
      ref_audio_url: params.refAudioUrl,
      ref_text: params.refText,
      format: params.format ?? "mp3",
    }),
  });

  if (response.status === 429) {
    const retryAfter = response.headers.get("Retry-After");
    throw new Error(`busy:${retryAfter ?? "1"}`);
  }

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json() as Promise<{
    request_id: string;
    status_url: string;
    chunks_total: number;
  }>;
}

async function pollUntilSucceeded(baseUrl: string, apiToken: string, requestId: string) {
  while (true) {
    const response = await fetch(`${baseUrl}/v1/tts/jobs/${requestId}`, {
      headers: { "Authorization": `Bearer ${apiToken}` },
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    const job = await response.json();
    if (job.status === "succeeded") return job;
    if (job.status === "failed") throw new Error(job.detail ?? "TTS job failed");

    await sleep(1000);
  }
}
```

## Parsing Audio Chunks

```ts
function parseLengthPrefixedAudio(buffer: ArrayBuffer): Uint8Array[] {
  const view = new DataView(buffer);
  let offset = 0;

  const chunkCount = view.getUint32(offset, false);
  offset += 4;

  const chunks: Uint8Array[] = [];
  for (let i = 0; i < chunkCount; i += 1) {
    const size = view.getUint32(offset, false);
    offset += 4;
    chunks.push(new Uint8Array(buffer, offset, size));
    offset += size;
  }

  return chunks;
}

async function downloadAudioChunks(baseUrl: string, apiToken: string, requestId: string) {
  const response = await fetch(`${baseUrl}/v1/tts/jobs/${requestId}/audio`, {
    headers: { "Authorization": `Bearer ${apiToken}` },
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  const buffer = await response.arrayBuffer();
  return parseLengthPrefixedAudio(buffer);
}
```

## Health-Based Throttling

Call `/health` periodically and slow down if `outstanding_chunks` stays high.

Useful fields:

```text
max_concurrent_chunks
short_reserved_chunks
long_concurrent_chunks
max_in_flight_chunks_per_job
busy_backlog_chunks
outstanding_chunks
active_tts_jobs
cache_audio_count
cache_transcript_count
sglang_ready
```

Simple rule:

```text
if sglang_ready == false: stop submitting new jobs
if outstanding_chunks >= busy_backlog_chunks * 0.8: slow down
if 429 rate increases: reduce LONG_OUTSTANDING_LIMIT by 25%
```

## Recommended Client Defaults

Use these defaults for the current Higgs deployment:

```text
chunk_size_chars=150-220
page_size_chunks=10
global_long_outstanding_chunks=160
max_outstanding_chunks_per_document=10
poll_interval_ms=1000
retry_429_base_ms=1000
retry_max_attempts=5
```

For offline batch-only throughput, raise `global_long_outstanding_chunks` toward
the server backlog limit, but keep each submitted job at about 10 chunks.

## Main Pitfalls

- Do not submit one 500-chunk job per long document when many documents are
  active.
- Do not use changing signed URLs for the same voice unless cache misses are
  acceptable.
- Do not concatenate WAV files by raw bytes.
- Do not use `/audio?from=&chunks=` as a generation scheduler. It only controls
  download after completion.
- Do not assume extra VRAM automatically caches many cloned voices.
