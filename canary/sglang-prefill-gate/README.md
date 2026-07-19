# SGLang Higgs prefill-admission canary

This is a selective manual backport of only the final admission-gate semantics
from official SGLang-Omni PR
[#1071](https://github.com/sgl-project/sglang-omni/pull/1071), head
`1d06ade914aecceb85b7043596638ebb3cb98887` as inspected on 2026-07-19.
The PR is still open, unapproved, and dirty; its headline H100 result is not a
performance claim for this Blackwell-class test host.

## Scope and base

- Git base: `df62e91a00d383e6f73ab9604386ffac6c520529`.
- Required preimage: the already-validated 1008a candidate, full diff SHA-256
  `6f8504a3230652a8bc1f6b943fe878900ef99a84251f810fcb3b52bcf09cd5b3`.
- Runtime changes: `omni_scheduler.py` and Higgs `stages.py` only.
- Tests: official ten gate behaviors, strengthened so partial admission mutates
  the real queue and abort calls `OmniScheduler.abort()`.
- Explicitly excluded: vocoder-process split, `torch.compile`, memory-budget
  changes, `engine_builder.py`, and SGLang's built-in `PrefillDelayer`.

The gate holds a small prefill refill while decode is active until either K
requests are waiting or the oldest request has waited T. It bypasses idle and
chunked-prefill paths, stamps missing enqueue times, and force-disables itself
for TP greater than one. K=0 is the kill switch. T=60 ms is not a hard TTFT
bound: release is checked between scheduler iterations and may also include a
decode/resolve step.

## Apply and launch

Reconstruct a new isolated worktree at the exact base, overlay the exact
production snapshot, apply `canary/sglang-1008a/apply.sh`, then apply this layer:

```bash
canary/sglang-prefill-gate/apply.sh \
  /root/autodl-tmp/sglang-omni-prefill-gate-canary
```

Do not patch either `sglang-omni-prod-sim` or the existing 1008a tree in place.
The launcher reads the fixed, non-secret
`/root/autodl-tmp/prod-sim/prefill-gate.env`; missing config defaults safely to
K=0/T=60. The committed K0/K8/K16/K20 variants are exact benchmark inputs.
Only `higgs_sglang` may be restarted; the bridge/API PID must remain unchanged.

## Test and benchmark gates

Run the gate test plus the same 50-test CUDA/async suite used for 1008a. Then
benchmark the exact task `7b66bc83-e5d5-48be-af5a-306a72d26bd4`, voice
`fcxDguohxleZaemvsuHB`, 69 cues, with the same bridge cap96/base10/burst20.

Use K0 on the patched source as control, then K8/K16/K20 at T=60 ms. Run one
warmup plus at least ten measured runs per K. Compare makespan, chunks/s,
generated-audio-seconds/s, completion-token/audio-duration distributions,
queue-enter to prefill-start, TTFT p50/p95/p99, errors/retries/degraded/OOM,
WAV validity, and resource use. Also exercise two concurrent jobs, abort while
held, and K0 again at the end to detect drift before selecting a sweet spot.

## Rollback

Soft rollback: install `prefill-k0.env` and restart only `higgs_sglang`.
Hard rollback: restore the validated 1008a launcher/source and restart only
`higgs_sglang`. Neither rollback requires reloading the bridge/API.
