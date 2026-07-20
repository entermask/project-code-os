# SGLang prefill-admission canary results

## Decision

Select **K=16, T=60 ms** for the test bridge. It is the measured sweet spot for
the production-shaped cap96/base10/burst20/two-burst-job configuration and the
69-cue SRT workload below. K12 did not beat K16; K20 regressed because releases
became more timer-dominated.

This is only a K0-versus-K16 comparison inside the candidate stack. Its K0 arm
already contains the new burst-20 bridge and the 1008a SGLang patch, so the
percentages below are **not** uplift versus the actual production runtime. The
production comparison protocol is tracked in `canary/prod-comparison/README.md`.

This result changes only the supplied test server. Production was inspected
read-only and was not restarted or modified.

## Fixed workload and runtime

- Coarse K0/K8/K16/K20 artifact: source commit
  `eafd437303502455102070128c53de5364adb234`; K12 refinement and final smoke
  artifact: source commit
  `d2249e4eb568790590871a4de749accc33d200ad`. The latter only adds the K12 env
  and updates documentation; benchmark and runtime fingerprints are identical.
- Immutable remote artifact roots:
  `deploy-artifacts/eafd437303502455102070128c53de5364adb234/sglang-prefill-gate`
  and
  `deploy-artifacts/d2249e4eb568790590871a4de749accc33d200ad/sglang-prefill-gate`.
- Model: `bosonai/higgs-audio-v3-tts-4b`.
- Task: `7b66bc83-e5d5-48be-af5a-306a72d26bd4`.
- Metadata voice: `fcxDguohxleZaemvsuHB`.
- Input: 69 fixed SRT cues; fixture, SRT, reference audio and fetched reference
  bytes were SHA-256 guarded.
- Bridge: 96 global chunks, 4 short-lane reservations, 92 long-lane slots,
  per-job base 10, burst 20 for at most 2 active jobs, backlog threshold 2000.
- Candidate source: selective final admission-gate semantics from official
  SGLang-Omni PR #1071, with K0 as a runtime kill switch. No vocoder split,
  `torch.compile`, memory-budget change, or built-in `PrefillDelayer` was added.

The matrix used one warmup before each phase. Each single-job phase measured 10
jobs. Each dual-job phase measured 5 synchronized waves, 2 jobs per wave. K0
was measured at both the start and the end and pooled as control. In total the
selection matrix contains **120 measured jobs / 8,280 measured cues**, excluding
warmups and smoke runs.

## Results

Percentages are improvements against pooled K0. Makespan is lower-is-better;
throughput columns are higher-is-better.

### Single 69-cue job

| Gate | Mean makespan | Makespan Δ | Chunks/s | Chunks/s Δ | Audio-s/s | Audio-s/s Δ |
|---|---:|---:|---:|---:|---:|---:|
| K0 | 2.743 s | control | 25.159 | control | 47.381 | control |
| K8 | 2.567 s | +6.42% | 26.890 | +6.88% | 50.593 | +6.78% |
| K12 | 2.547 s | +7.13% | 27.094 | +7.69% | 50.864 | +7.35% |
| **K16** | **2.502 s** | **+8.78%** | **27.580** | **+9.62%** | **51.884** | **+9.50%** |
| K20 | 2.648 s | +3.45% | 26.060 | +3.58% | 49.069 | +3.56% |

For K16, the 50,000-resample bootstrap 95% interval is 7.95–9.50% for
makespan improvement and 8.63–10.50% for chunks/s improvement.

### Two concurrent 69-cue jobs

| Gate | Mean wave makespan | Makespan Δ | Chunks/s | Chunks/s Δ | Audio-s/s | Audio-s/s Δ |
|---|---:|---:|---:|---:|---:|---:|
| K0 | 3.722 s | control | 37.172 | control | 69.989 | control |
| K8 | 3.140 s | +15.64% | 44.103 | +18.64% | 82.819 | +18.33% |
| K12 | 3.227 s | +13.30% | 42.935 | +15.50% | 81.018 | +15.76% |
| **K16** | **3.112 s** | **+16.38%** | **44.441** | **+19.55%** | **83.586** | **+19.43%** |
| K20 | 3.216 s | +13.59% | 43.105 | +15.96% | 81.369 | +16.26% |

For K16, the 50,000-resample bootstrap 95% interval is 11.70–20.43% for
makespan improvement and 13.44–25.25% for chunks/s improvement.

## Latency trade-off

K16 intentionally coalesces small prefill refills while decode is active. Its
queue-wait p50/p95/p99 was 33.0/75.5/78.1 ms for one job and
40.7/77.5/79.9 ms for two jobs. K0 was usually about 2 ms at p50 and 9–16 ms
at p95/p99. T=60 ms is not a hard latency cap because the gate is checked
between scheduler iterations.

That wait reduced total work completion: engine-complete-after-queue p50 fell
from roughly 657–672 ms to 583 ms for one job, and from roughly 907–918 ms to
694 ms for two jobs. K16 therefore improves end-to-end completion despite the
extra admission wait. This matrix covers the exact one/two-large-job target;
short-request fairness and more than two simultaneous large jobs remain a
separate rollout observation, not a claim from these data.

## Safety and verification

- Every matrix phase is `valid=true` with 0 failed/degraded chunks, 0 poll
  errors, 0 retry/error log matches and 0 invalid WAV files.
- K0 control drift was small: single mean 2.731 s at the start versus 2.755 s
  at the end; dual mean 3.728 s versus 3.715 s.
- Runtime checks pinned loopback endpoints, exact bridge/SGLang PIDs and CLI,
  source and diff fingerprints, source mtimes, cache bytes, synchronized dual
  submits, profiler coverage, idle state before each wave and recovery after it.
- Separate session test logs (not embedded in these benchmark reports) recorded
  gate tests 10/10, the combined targeted gate + CUDA/async run 60/60, the
  isolated abort/disconnect contract 5/5 and bridge tests 39/39.
- A final K16 smoke on the deployed `d2249e4` artifact passed 1 warmup plus 1
  measured 69-cue job. Its report SHA-256 is
  `d86ab5eb00d674f02e773699af1b31146dfc058889b949fee5f338ef0a4d44ad`.
- The HTTP bridge PID stayed `128506` throughout the final SGLang-only restart.
  The test server was left healthy, idle and listening on the intended SGLang
  port 8000 with K16/T60.

`summarize.py` reproduces the aggregate table and bootstrap comparisons from
the checked-in raw JSON reports. K0 remains the immediate soft rollback.
