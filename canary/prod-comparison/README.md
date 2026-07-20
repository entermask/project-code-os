# Production-to-candidate comparison canary

This canary fixes the invalid comparison between the K0/K16 test-only sweep and
the actual production stack. K0 in that sweep already used the new bridge,
burst-20 feed and the 1008a SGLang patch; it was not production.

The comparison uses four named arms:

| Arm | Host | Bridge | SGLang | Purpose |
|---|---|---|---|---|
| P | production | live production | live production | operational baseline only when exclusive |
| A | test | exact `prod-current/app.py`, feed 10 | exact production clone | software baseline on common hardware |
| B | test | candidate, burst 20 | candidate, K0 | pre-gate candidate bundle |
| C | test | candidate, burst 20 | candidate, K16/T60 | final candidate |

`../sglang-prefill-gate/benchmark/benchmark.py` accepts
`PREFILL_BENCH_ARM=candidate` (default) or
`PREFILL_BENCH_ARM=prod-clone`. The latter is pinned to loopback port 6007,
bridge SHA `328851b1...`, SGLang diff `9a3ba2d6...`, feed 10 and a separate
cache under the allowed media root. The candidate remains pinned to port 6006, bridge SHA `00f37e06...`,
SGLang diff `304eb276...` and burst 20. Both arms reject source, CLI, cache,
environment, PID, health, profiler, WAV or recovery drift before producing a
valid report.

Both SGLang launchers pin the same absolute production Hugging Face snapshot
`a7f70853...`, force offline loading, and retain the public served model name.
The test cache reuses the already-identical weight/config/tokenizer blobs and
downloads only the production revision's small README/LICENSE metadata. The
benchmark verifies the complete production symlink manifest
`a06349c4...` plus exact runtime blob links and sizes.

Only A-versus-C is the total software uplift on the same test hardware.
B-versus-C isolates the gate. P-versus-A is a host/driver calibration and must
not run while production contains foreign traffic. Saturation results from an
older N96/N128 workload cannot be mixed with the one/two-job exact-SRT result.

## Safe A to C switching

The prod-clone API has its own supervisor and defaults to stopped. SGLang for
both arms always runs under the main supervisor, so its output stays in the
single log audited by the harness. `switch-sglang-arm.sh` rejects unknown
launcher hashes, stops the managed process, waits until port 8000 is genuinely
bindable without `SO_REUSEADDR`, installs the selected launcher atomically and
waits for health. A failed switch restores the previous launcher.

Run balanced control blocks and always leave the host on candidate K16:

1. Start the separate supervisor once; confirm `prod_clone_api` is stopped.
2. Switch to `prod-clone`, then explicitly start `prod_clone_api` and run A.
3. Stop `prod_clone_api`, switch to `candidate` and run C.
4. Repeat A then C for an A-C-A-C drift check. Reject the matrix when the two A
   controls differ by more than 5%.
5. Confirm the final SGLang CLI is candidate K16/T60 and the candidate bridge
   health is idle.

Never start both bridge benchmarks at once. The profiler exclusivity guard will
also invalidate a phase containing any request outside its journal.
