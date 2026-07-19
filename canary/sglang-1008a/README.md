# SGLang Higgs output-buffer canary (`1008a`)

This is a selective, non-streaming backport from official SGLang-Omni
[#1008](https://github.com/sgl-project/sglang-omni/pull/1008), squash commit
[`0799c95`](https://github.com/sgl-project/sglang-omni/commit/0799c95e391ece0009518e1b41049aefc068f2b4).
It targets the SRT path used by this bridge, which does not send `stream=true`.

## Exact base and scope

- Git base: `df62e91a00d383e6f73ab9604386ffac6c520529`.
- Production six-file overlay: `git diff --no-ext-diff` SHA-256
  `9a3ba2d6f6b8459e631b488b76eb5a9a96432ed32edc5dcab770789dd4ef6ad4`.
- Runtime changes: only `request_builders.py` and `model_runner.py`.
- The custom CUDA-graph/async-decode/RAS implementation is preserved.
- `config.py`, `stages.py`, `vocoder_scheduler.py`, `model.py`, `sampler.py`,
  `payload_types.py`, and `text_tokenizer.py` are not changed by runtime patch
  `0002`.

The backport preallocates one CPU `int64[max_new_tokens, num_codebooks]` buffer
per request, copies each decoded row into it, and lets the result adapter read
the populated slice directly. It removes a Python list of per-row clones and
the final `torch.stack`; the eager prefill collector also reads codebook zero
from the CPU row it just copied instead of causing another GPU-to-CPU sync.
At 2,048 tokens and eight codebooks this is 128 KiB/request, or about 12 MiB
when 96 requests all reach the cap.

Streaming batching/stride propagation from the rest of #1008 is intentionally
excluded because it is not executed by the current SRT bridge. It must be a
separate canary if streaming is enabled later.

## Patch order

1. `0001-test-align-production-fixtures.patch` only updates stale upstream test
   doubles for the six-file production overlay. Before it, the exact baseline
   has four known fixture failures (two `context_turns`, two `recent_codes`).
2. `0002-preallocate-output-code-buffer.patch` contains the two runtime changes
   and targeted result-adapter/eager/CG tests.

`apply.sh` refuses a worktree whose HEAD or pre-patch production diff hash does
not match the values above, then applies and whitespace-checks both patches in
order:

```bash
canary/sglang-1008a/apply.sh /path/to/sglang-omni-1008a-canary
```

Do not apply this to the active `sglang-omni-prod-sim` tree. Reconstruct a new
worktree at `df62e91`, overlay the exact six local files from `prod-current/`,
then run the guarded script.

## Gates on the GPU test host

Use the existing Linux/CUDA venv with the candidate worktree first on
`PYTHONPATH`. Local macOS does not have Torch/SGLang and can only run syntax,
patch, and manifest checks.

```bash
python -m pytest -q \
  tests/unit_test/higgs_tts/test_request_builders.py \
  tests/unit_test/higgs_tts/test_pipeline.py \
  tests/unit_test/higgs_tts/test_async_decode_runner.py \
  tests/unit_test/higgs_tts/test_batched_step.py
```

Then boot the candidate from a separate source tree while preserving cap 96,
CUDA graph 96, Triton attention, the same model/venv/env, and the current bridge
commit. Restart only `higgs_sglang`; do not restart the bridge for this phase.

Run the exact task `7b66bc83-e5d5-48be-af5a-306a72d26bd4`, voice
`fcxDguohxleZaemvsuHB`, 69 cues in one WAV job, one warmup plus at least ten
measured runs for both baseline and candidate. The production sampler currently
uses the global CUDA RNG and does not honor a request seed, so `seed` must not be
used as a reproducibility claim for this comparison. The expected gain is only
low single digits; compare run distributions and normalize throughput by both
chunk count and generated audio duration.

```bash
python /root/autodl-tmp/fixture-task-7b66bc83/run-direct.py \
  --batch-size 69 --poll-seconds 0.05 \
  --output-dir /root/autodl-tmp/canary-runs/sglang-1008a/baseline/run-01
```

Required checks: 69/69, zero failed/degraded, valid PCM16 24 kHz mono WAV,
completion-token/audio-duration distributions, zero quality-retry warnings in
the measured window, no CUDA/OOM errors, GPU/CPU resource sampling, and
API/SGLang health after the run.

## Rollback

The production-sim source tree stays immutable. Roll back by restoring the
launcher `PYTHONPATH` to `/root/autodl-tmp/sglang-omni-prod-sim` and restarting
only `higgs_sglang`. The bridge does not need a reload.
