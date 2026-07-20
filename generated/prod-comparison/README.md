# Exact production clone versus K16 candidate

Test date: 2026-07-20. Test host only; live production was read-only and was
not restarted or benchmarked.

Both arms ran on the same test GPU with the same production model snapshot
`a7f70853f163c4cccbdd27ce9a80dd97961fc581`, task
`7b66bc83-e5d5-48be-af5a-306a72d26bd4`, voice
`fcxDguohxleZaemvsuHB`, 69 cues and 16 inference steps.

- A: exact current production bridge/config plus exact production SGLang diff.
- C: candidate bridge (burst 20) plus candidate SGLang K16/T60.
- Single: 10 measured jobs per arm.
- Dual: 5 measured waves / 10 jobs per arm; client submit skew stayed below
  0.75 ms.

| Load | Metric (p50) | Production clone A | Candidate C | C versus A |
|---|---:|---:|---:|---:|
| Single | makespan | 5.155 s | 2.618 s | 1.97x faster |
| Single | audio seconds / second | 25.12 | 49.70 | +97.8% |
| Single | chunks / second | 13.39 | 26.36 | +96.9% |
| Dual | wave makespan | 6.783 s | 3.198 s | 2.12x faster |
| Dual | audio seconds / second | 38.32 | 81.24 | +112.0% |
| Dual | chunks / second | 20.35 | 43.15 | +112.1% |

All 2,760 measured cue executions succeeded: zero failed chunks, degraded
chunks, poll errors or invalid WAVs. Single-job output duration was effectively
equal (A mean 129.876 s; C mean 129.884 s).

The old approximately 175 audio-s/s number is a saturation result with a much
larger offered load. It must not be compared numerically with this exact-SRT
single/dual matrix. This matrix establishes the direct end-to-end software
uplift versus the current production stack for the real SRT task; it does not
claim a saturation uplift.
