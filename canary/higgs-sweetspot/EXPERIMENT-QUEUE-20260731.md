# KẾT QUẢ THỰC THI TIER 1 — cập nhật 2026-07-31 chiều

| Block | Trạng thái | Kết quả |
|---|---|---|
| 1. uvloop/httptools | ✅ ĐÓNG (verified-good) | ACTIVE — bằng chứng /proc/maps (uvloop 5 mappings, httptools 10). Không cần A/B. |
| 2. Radix hit-rate | ✅ ĐÓNG (verified-good) | 94.7–97.5% hit trên mọi arm 31/07; ref-block ~180–208/211 tokens reuse; new-token còn lại = text unique. KHÔNG cần canonicalize — prefill headroom ≈ 0. |
| 3. Triton cache persist | ✅ ĐÓNG (NO-OP) | /root/.triton/cache đã persistent sẵn (98 entries từ 30/07, restart không JIT lại). Việc còn giá trị: boot-warmup dummy encode (vá compile-cold lộ ra từ disk-cache test). |
| 4. Sync census | ✅ ĐÃ CHẠY (static) | Async path SẠCH: 1 pinned non-blocking D2H/step, RAS thuần GPU, delay-state GPU, không busy-wait hot path. Offender còn lại: (a) model_runner.py:403 `.sum().item()` + :412 `.nonzero()` — 2 GPU sync/prefill-request có ref codes, chưa bị syncfree cover; (b) stages.py:483 `.tolist()` chạy lại cả khi cache HIT (O(T×8)/request); (c) host collect loop 128 iter/step (CPU-only). → 3 ứng viên patch env-gate tiếp theo. |
| 21. Stage-width 96→128 | ✅ ĐÃ CHẠY (giữ) | Paired n=3 long 20×10 MP3: tok/s +0.43% (noise), p95 −2.55% (âm nhất quán 3/3 waves), 0 fail. Giữ trong arm vì config nhất quán + không có downside. LƯU Ý: tree pin chỉ nhận addressing theo INDEX (`--stages.0/1/3.factory_args...`), theo tên sẽ crash `TypeError: list indices`. |
| 8. Disk-backed ref codes | ✅ LIVE | (chạy buổi sáng) −1.4s cho voice quay lại sau restart; results/20260731/diskcache/SUMMARY.md |

Arm hiện tại trên test: `width128-treat` = cap128 + LRU512 + disk-cache + stage-width-128, tất cả gate sạch.
Controller mới có `TEST_EXTRA_SERVER_ARGS` (charset-validated, argv-verified) — mọi knob experiment sau chỉ cần env.
Kế tiếp theo queue: block 5 (concurrency matrix + gate_prefix.py), block 6 (RAS audit) → block 7 (overlap patch — ứng viên throughput lớn nhất còn lại), và 3 patch từ census.

---

# QUEUE THÍ NGHIỆM — Higgs v3 @ cap-128 (38 đề xuất → 24 block sau gộp, 0 bị giết bởi prior evidence)

**Phương pháp:** rank = (win kỳ vọng trên stack này) × (xác suất qua gate audio) / chi phí; trong mỗi tier win, xếp rẻ-trước. Đã đối chiếu từng `prior_rejection_check` với REPORT-20260730.md: **không item nào có mechanism bị cover đầy đủ bởi rejection cũ** — các rejection (K20@cap96, TRTLLM backend-swap, torch.compile whole-model, FP8/MXFP8 quant, vocoder-CUDA-graph per-call, d957 wholesale, C14/C16 pool) đều là cơ chế khác với đề xuất tương ứng. 14 mục trùng cơ chế được GỘP (chi tiết cuối). Hai fact đã xác minh chéo giữa researchers: (a) `disable_overlap_schedule=True` bị ép cứng trong `stages.py` SAU overrides — bước grep của E1 đã có đáp án, vào thẳng patch; (b) vocoder pin thực tế là `max_batch_size=96 / wait 2ms` (grep SSH), không phải "default 4/2ms" như block papers giả định.

---

## TIER 1 — Win lớn hoặc mở khóa nhiều hướng (rẻ trước)

**1. Xác minh uvloop active + audit httpx pool** *(bridge E7)*
- Cơ chế: `uvicorn.run` không truyền `loop` → auto; nếu wheel uvloop hỏng thì bridge đang chạy stock asyncio với 96-way subprocess+HTTP fan-out — win 5–15% miễn phí nếu inactive.
- Cách chạy: thêm 1 dòng log `type(asyncio.get_running_loop()).__module__` vào startup hook app.py, restart API test, đọc log. Sau đó A/B `loop='asyncio'` vs `loop='uvloop'` với sustained 20×10 + true-short 24×4, n≥3. Log p99 connection-acquire quanh `client.post` trong sustained wave.
- Tín hiệu: nếu inactive → +5–15% wall khi bật; nếu active → arm asyncio hiện regression, tài liệu hóa độ nhạy loop (định giá cho block 22).
- Gate: zero behavior change ngoài loop; clean run mỗi arm; dòng log ship vĩnh viễn (RESEARCH mục 5 yêu cầu).
- Chi phí: **phút**.

**2. Audit radix prefix hit-rate + canonicalize prompt** *(papers)*
- Cơ chế: radix chỉ hit khi prefix byte-identical; nếu bridge lắp prompt không ổn định giữa các chunk cùng job thì N-1 prefill/job đang miss cache — chưa ai đo hit rate thật ở cap-128.
- Cách chạy: parse `cached_tokens/prompt_tokens` per request từ server log/metrics trong 1 wave long 20×10, tính hit ratio theo chunk index (0 vs 1..9). Nếu chunk 1..9 <90% reuse → diff byte 2 payload cùng job trong app.py dispatch, canonicalize (ordering ổn định, system/reference segment identical), đo lại.
- Tín hiệu: phase đo là deliverable; nếu reuse thấp và fix được: long wall −1–5%, TTFT cold-job giảm. Nếu >90% → đóng verified-good giá rẻ.
- Gate: canonicalize không đổi token content → deterministic prefix suite không đổi; pytest xanh.
- Chi phí: **phút** (đo) → giờ (fix nếu cần).

**3. Persist TRITON_CACHE_DIR + warmup shape lúc boot** *(gộp: papers triton-cache + engine-knobs triton-cache-persist)*
- Cơ chế: mỗi restart engine JIT lại toàn bộ kernel Triton → cold restart chậm hàng chục giây, first-job sau restart ăn đủ. Persist cache + warmup 3 shape đại diện chuyển cost này về boot một lần.
- Cách chạy: `export TRITON_CACHE_DIR=/root/autodl-tmp/triton-cache` + `SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE=true` trong launcher; controller thêm bước warmup 3 dummy chunk (short/long/max-context) sau health-check trước khi mark arm live. Đo launch→healthy qua 2 restart liên tiếp + first-job latency, so control, n≥5 restart.
- Tín hiệu: restart 2 nhanh hơn rõ (chục giây); first-wave p95 hội tụ về steady p95. Bonus: mọi A/B tương lai rẻ hơn (arm-switch cost giảm).
- Gate: deterministic prefix exact sau restart có-cache vs không-cache; steady-state tok/s trong noise; version-key cache dir theo Triton/torch version; ghi size cache.
- Chi phí: **phút**.

**4. Census sync/CPU per decode step (nsys + grep), quyết định 4 hướng** *(gộp: fish E2 + papers D2H-merge + papers CPU-budget-profile)*
- Cơ chế: một pass profiling trả lời cùng lúc: (a) còn bao nhiêu D2H/`.item()`/busy-spin per step (cơ chế thắng +46% của vLLM-Omni #1859), (b) resolve path của #590 còn 1 hay 3 sync/step, (c) CPU share của sampling/RAS/delay-bookkeeping ở bs128 — con số mà RESEARCH đã dùng để bác free-threading/Green-Contexts *mà chưa từng đo*.
- Cách chạy: (1) `nsys profile --duration=60` khi steady bs128 long wave; đếm cudaMemcpyDtoH + cudaStreamSynchronize per step, attribute: graph replay / D2H / CPU resolve / RAS / next-step prep. (2) `grep -rnE '\.(cpu\(\)|tolist\(\)|item\(\))'` trong `sglang_omni/models/higgs_tts/` + vocoder_scheduler + resolve path quanh #590. (3) Mỗi offender lớn → 1 patch env-gate riêng (pinned buffer non_blocking / merged staging / Condition wakeup), mỗi patch một arm A/B.
- Tín hiệu: census miễn phí quyết định; mỗi patch +1–5% tok/s; CPU share >10% → spec GPU-resident RAS port làm follow-on; census sạch → đóng cả hướng kernel/CPU bằng data.
- Gate: patch pure-transfer phải bit-exact (PCM hash per chunk 100% = control, prefix 26/26); patch nào đổi PCM → reject ngay; giữ dummy-padding-row guard khi pack state.
- Chi phí: **phút** (census) → giờ (mỗi patch).

**5. Ma trận regression đa-concurrency + pre-screen `gate_prefix.py`** *(fish E6)*
- Cơ chế: stack chưa từng đo c=1/8 hay curve per-request tok/s theo concurrency; ma trận frozen-baseline kiểu Fish bắt cliff giữa CUDA-graph bucket và cho mọi A/B sau một pre-screen <1 phút — infra multiplier cho toàn queue.
- Cách chạy: thêm `--concurrency-matrix 1,8,32,96,128` vào scripts/bench_production_tts.py (giữ c in-flight liên tục 2 phút/mức, ghi per-request decode tok/s + submit→chunk0); freeze JSON baseline vào canary/higgs-sweetspot/; viết `scripts/gate_prefix.py` (1 request greedy seed cố định, so 5 frame delayed-codes đầu với golden); rule CI: mức nào lệch >5% baseline → đỏ.
- Tín hiệu: curve knee/dip actionable ngay; mọi experiment sau dùng chung chuẩn.
- Gate: validation nội tại: token parity 2 lần baseline liên tiếp, CV per-mức <5% trước khi freeze.
- Chi phí: **giờ**.

**6. RAS-window batch-invariance audit (adversarial join/leave)** *(fish E7)*
- Cơ chế: RAS 7/2 history-dependent; nếu window index theo vị trí batch thay vì req_pool_indices thì reorder khi join/leave ở cap-128 trộn history giữa request — đúng lỗi speaker-drift Fish gặp, và là nghi phạm cho long-tail runaway của wave r0 (REPORT ghi "chưa root-cause").
- Cách chạy: (1) audit code buffer RAS trong sampler higgs_tts của pin: remap theo req index ổn định? (2) golden = 1 request deterministic c=1; chạy lại CÙNG request khi 95+ background request join/leave staggered; so delayed-code stream với golden, lặp 26 lần. (3) Mismatch → fix indexing, re-run, re-examine tail r0.
- Tín hiệu: 26/26 exact → tăng độ tin cho mọi experiment batch cao; mismatch → correctness bug thật, ưu tiên trên MỌI tối ưu throughput.
- Gate: chính nó là gate; exact delayed-code hash, không chấp nhận "gần giống".
- Chi phí: **giờ**.

**7. Bật overlap schedule cho Higgs decode (patch có kill switch)** *(fish E1)*
- Cơ chế: đã xác minh `disable_overlap_schedule=True` bị ép cứng trong stages.py → mỗi decode step vẫn có host barrier. Overlap = CPU chuẩn bị batch kế trong khi GPU chạy batch hiện tại — khác hẳn sync-free launch cache (chỉ cache launch buffer, đo neutral). Bản dịch trực tiếp của "pingpong barrier removal" của Fish.
- Cách chạy: patch conditional trong stages.py sau chỗ ép True, gate bằng env `SGLANG_OMNI_HIGGS_OVERLAP=1`, restart arm cap-128; paired A/B long 20×10 + true-short 24×4 MP3+loudnorm, n≥4 waves/arm.
- Tín hiệu: +2–6% completion tok/s, p50/p95 giảm; delta CI-0 → kết luận host barrier không phải bottleneck ở bs128 (đóng có evidence).
- Gate: deterministic greedy-prefix + full-output PCM exact 26/26 (nếu RAS đọc token CPU-side chưa sync, gate này bắt được — chạy SAU block 6 để tách bạch), token parity ≤1.5%, 0 degraded, kill switch.
- Chi phí: **giờ**. Rủi ro gate cao hơn trung bình (multimodal path bị disable có thể có lý do) — nhưng chính vì thế mới cần chạy sau block 6.

**8. Persist reference codes: endpoint tối thiểu + bridge store** *(gộp: fish E9 + papers reference-codes)*
- Cơ chế: cold first-use 1.57–1.81s vs warm 22ms là headroom latency lớn nhất còn lại (REPORT xếp "next highest-value"; MEMORY: "headroom còn lại là latency"). Chiều GỬI `reference_codes` [T,8] undelayed đã có sẵn trong pin; chỉ thiếu chiều LẤY.
- Cách chạy: (1) engine test tree: `POST /v1/audio/reference_codes` ~40 dòng tái dùng `load_audio_to_24k` + `encode_reference`, trả [T,8] undelayed + model/codec version — không đụng AR path. (2) app.py env-gate `HIGGS_REFERENCE_CODES=1`: tại voice upload/first-use fetch + persist JSON cạnh REF_AUDIO_DIR, key = content-hash + TTS_BACKEND_NAME + codec version; dispatch gửi codes + reference_text, silent fallback về raw path khi mismatch. (3) Bench: clean restart, đo submit→first-prefill raw-cold vs codes cho bucket reference 1/5/10/30s, n≥10 cold start/bucket.
- Tín hiệu: cold first-use 1.6–1.8s → ~50–100ms; warm không đổi.
- Gate: codes persist phải cho output identical với engine-encoded dưới deterministic seed (exact delayed-code hash); cap T≤7500 enforced; version key invalidate; ASR + nghe tay; kill switch + fallback; pytest + MP3 smoke.
- Chi phí: **ngày** — nhưng win chắc chắn nhất tier này.

---

## TIER 2 — Win vừa (1–5% hoặc tail), rẻ trước

**9. LAME `-compression_level` sweep @ CBR 128k** *(bridge E2)*
- Cơ chế: app.py:1283 không truyền compression_level → LAME default q3 (tier chậm); q7 nhanh 1.5–2.5× cùng stream params (vẫn CBR 128k/44.1k — không đụng contract concat của worker, không vi phạm ràng buộc MP3-always). Encode service 33.7s/wave là phần ffmpeg lớn nhất của bridge.
- Cách chạy: bước 1 (phút, offline): microbench 50× một WAV ~15s với `-compression_level {unset,2,5,7,9}` full production chain. Bước 2 (chỉ nếu delta ≥20%): env `FFMPEG_MP3_COMPRESSION_LEVEL` (rỗng = behavior cũ) trong `_wav_to_mp3`; A/B long 10×10 + true-short 24×4, FFMPEG_TIMING_ENABLED=1, n≥5.
- Tín hiệu: encode service total −10–30%; true-short tok/s + vài % (encode chiếm tỷ trọng lớn ở job ngắn).
- Gate: blind AB listening ≥10 cặp không thua; ffprobe stream params identical; `_is_mp3` pass; concat smoke worker-side; loudness delta <0.2 LU. Arm VBR chỉ xét nếu qua gate concat/duration riêng.
- Chi phí: **phút → giờ**.

**10. `SGLANG_ENABLE_TORCH_INFERENCE_MODE=true`** *(engine-knobs)*
- Cơ chế: inference_mode bỏ autograd version-counter/view-tracking trên host — đúng đường CPU-launch-bound đã chứng minh của stack (win PCM native, sync-free family). Numerics không đổi.
- Cách chạy: thêm env vào launcher arm test, restart, long 20×10 n=4 + true-short n=3 paired. Kill switch = bỏ env.
- Tín hiệu: +0–2% tok/s; hoặc fail-fast RuntimeError ngay wave đầu (không âm thầm).
- Gate: zero error toàn wave; deterministic prefix exact (kỳ vọng bit-exact); bất kỳ RuntimeError inference-mode → reject ngay.
- Chi phí: **giờ** (1 restart).

**11. `--num-continuous-decode-steps` sweep {1,2,4}** *(fish E5)*
- Cơ chế: 2–4 decode step mỗi scheduler iteration giảm CPU overhead per token (member CPU-side của barrier-removal), đổi lấy admission prefill trễ hơn — tương tác với K16 + short lane phải đo.
- Cách chạy: xác nhận flag trong pin, sweep {1(control),2,4} trên arm cap-128; paired long 20×10 + BẮT BUỘC mixed soft_reserved wave đo short queue p95.
- Tín hiệu: long +1–3% nếu scheduler CPU là overhead thật; short p95 tăng >5% → reject dù long thắng (mixed fairness là sweet spot đã chọn).
- Gate: deterministic prefix exact (chỉ đổi nhịp scheduler); token parity; 0 degraded; theo dõi p95 hai lane.
- Chi phí: **giờ**.

**12. `async_decode_min_batch_size` sweep {1, 2(control), 8}** *(engine-knobs)*
- Cơ chế: async decode chỉ kích hoạt khi batch ≥2 (omni_scheduler.py:1587) → đuôi wave/short-lane batch-1 rơi về đường sync. Hạ 1 kéo async tới singleton; nâng 8 đo fixed cost async ở batch nhỏ.
- Cách chạy: `--stages.2.factory_args.async_decode_min_batch_size {1|8}`; true-short 24×4 n=4 (tail nhạy nhất) + long 20×10 n=3 paired.
- Tín hiệu: arm 1: short p95/max −1–5%, long neutral; arm 8 tốt hơn control cũng là thông tin.
- Gate: zero errors; deterministic prefix; token parity ≤1.5%.
- Chi phí: **giờ**.

**13. Triton decode kv-splits sweep** *(gộp: fish E4 + engine-knobs triton-decode-kv-splits)*
- Cơ chế: decode Triton dùng max_kv_splits=8 + kernel phụ tính num_kv_splits động mỗi step; context Higgs ngắn (~vài trăm token) + bs128 đã bão hòa CTA → splits thấp/static bỏ kernel phụ + giảm reduce split-k. Tune trong chính backend Triton BF16 đang pass gate — KHÔNG phải backend swap như TRTLLM (rejection đó là đổi backend/numerics gây early-termination, không cover tune nội bộ).
- Cách chạy: arms (a) `triton_attention_num_kv_splits 4`; (b) `2`; (c) `SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS=true`; (d) optional `SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE=512`. Mỗi arm long 20×10 n=4 + true-short n=3 paired.
- Tín hiệu: +0–4% tok/s long; splits quá thấp mất parallelism sẽ hiện rõ trong sweep.
- Gate: NẶNG — split đổi thứ tự cộng FP → không bit-exact: deterministic prefix/full-output cohort seeded + phân phối completion-token/chunk (~340) bắt early-termination + quality corpus 13 file + token parity ≤1.5%. Xác suất qua gate trung bình (stack trajectory-sensitive) — vì thế xếp sau các knob numerics-neutral.
- Chi phí: **giờ** (4 restart).

**14. Prefill re-sweep @ cap-128: K×wait + token budget** *(gộp: papers K-resweep + engine-knobs prefill-token-budget)*
- Cơ chế: K16-optimal tìm ở cap 96 đã stale — mỗi prefill interruption giờ stall 128 stream thay vì 96 (tỷ số amortization đổi). Đồng thời budget token (`chunked_prefill_size` 8192 / `max_prefill_tokens` 16384, batch-wide) chưa từng sweep: K16 thả 16 prompt × 1–2k token có thể vượt budget → bẻ nhiều forward, mỗi forward stall decode (overlap đang bị ép off).
- Cách chạy: phần 1: K {16(control),24,32} × wait {60,120} (ưu tiên K24/60, K32/60 trước). Phần 2: `chunked_prefill_size 16384 + max_prefill_tokens 32768` (một pass) vs `chunked_prefill_size 4096` (stall ngắn, xen decode) vs control. Mỗi arm: long 20×10 n=4 + mixed soft_reserved n=4, quy trình run_round2_mixed_arm.sh.
- Tín hiệu: long +1–4% và/hoặc p95 giảm; guard metric = short queue p95 (K lớn trì hoãn short prefill).
- Gate: token parity ≤1.5%; 0 degraded; short p95 regression <5%; deterministic prefix; reversible thuần env.
- Chi phí: **giờ**. *Prior check: rejection "K20" là tại cap 96 (REPORT dòng "K16→K20 under FCFS/1.0 ... retain K16") — không cover operating point 128; token budget chưa từng có arm.*

**15. Một pass WAV parse cho cả peak + RMS** *(bridge E4)*
- Cơ chế: mỗi chunk hiện mở/copy PCM 2–3 lần (peak backstop, RMS screen, anchor) + 2–3 executor hop; gộp một `readframes` pass và cache rms trên ChunkResult — mở rộng đúng winner audioop (family được ACCEPT, không phải reject).
- Cách chạy: patch ~40 dòng `_wav_peak_rms_dbfs` trả (peak, rms) trong 1 to_thread; `_shared_or_rescue_af` nhận rms precomputed, fallback parse khi absent. Verify bit-identical trên corpus 13 file rồi A/B true-short 24×4 + long 10×10, n≥5.
- Tín hiệu: khiêm tốn thật thà: +1–3% true-short, có thể một phần trong noise — nhưng patch bé, vĩnh viễn, strictly bớt việc.
- Gate: peak/rms bit-identical corpus 13 file (0/169 flip như methodology REPORT); pytest xanh; báo CI-backed delta kể cả khi nằm trong noise.
- Chi phí: **giờ**.

**16. Persist loudnorm stats per-voice (sidecar seed anchor)** *(bridge E1)*
- Cơ chế: app.py:1192-1198 đã short-circuit toàn bộ anchor measure khi `req.ln_measured` hợp lệ (đường worker batch-2+ đã production-proven); persist sidecar generalize nó cho batch 1 — bỏ 1 ffmpeg measure/job VÀ bỏ chỗ N-1 sibling chờ anchor future (~0.3s trên lane p95 ~3.5s). Outlier-rescue screen giữ nguyên làm lưới an toàn chống stats cũ.
- Cách chạy: Phase A (không code): 20 job cùng voice khác text, scrape ln_measured từ status payload, tính std/range của i/tp/rms. Phase B (nếu qua): sidecar ~30 dòng `REF_AUDIO_DIR/<sha256>.ln.json` key hash(af)+backend, kill switch `HIGGS_LN_PERSIST` (default off); A/B n≥5 long 10×10 + true-short 24×4, FFMPEG_TIMING_ENABLED=1.
- Tín hiệu: measure-kind calls/wave → ~0; true-short p95 −2–4%.
- Gate: Phase A: std(i) ≤1 LU, rms range trong LN_OUTLIER_DB/2; final: ebur128 loudness sampled MP3 ±1 LU vs control; rescue flip rate không đổi (0/169-style); clean run zero degraded; pytest xanh.
- Chi phí: **giờ**.

**17. Duration-aware max_new_tokens cap + runaway guard (bridge-only)** *(papers)*
- Cơ chế: budget max_new_tokens dự đoán từ độ dài text (ceil(1.6×pred)+75) hard-stop runaway RAS — cắt p99 tail + token lãng phí, không đụng chunk khỏe. REPORT: r0 có tail giống runaway chưa root-cause, usage headers che retry. *Prior check: rejection "không hạ schedule_conservativeness" là knob ENGINE cho admission headroom (REPORT mục riêng) — không cover thu hẹp declared budget để chặn tail.*
- Cách chạy: env `HIGGS_MAXTOK_FROM_TEXT=1` trong chunk-dispatch app.py; fit a,b bằng regress completion_tokens ~ char count từ raw logs results/20260730 (không cần data mới); shadow 1 ngày bench (cap OFF, log prediction); rồi A/B ON/OFF long 20×10 + mixed n≥5 + 1 adversarial wave dùng text stress r0.
- Tín hiệu: shadow: P(actual>cap)<0.1% cho chunk khỏe; A/B: p99 −10%+ trên wave stress-shaped, retry rate <0.5%, tok/s flat-to-up.
- Gate: ZERO truncated audio — chunk bị cap bắt buộc retry với budget uncapped (cap không bao giờ ship truncation); ASR/duration spot-check 20 chunk capped-then-retried; kill switch; pytest xanh. Chạy SAU block 6 (nếu tail r0 là bug RAS-window thì fix bug trước, cap chỉ là guard).
- Chi phí: **giờ**.

**18. CPU affinity partition: engine vs API+FFmpeg** *(gộp 4: fish E8 + papers CPU-isolation + engine-knobs cpu-affinity + bridge E8)*
- Cơ chế: nice19 evidence (short 3,879 vs 3,492 tok/s nice-off, n=1) chứng minh contention CPU nhưng nice không ngăn migration/cache-thrash/SMT-sibling; box 208 core đủ để partition cứng. Khác hẳn Green-Contexts bị loại (đó là GPU SM partitioning).
- Cách chạy: launcher: `taskset -c 0-103` cho sgl-omni serve; `taskset -c 104-207` cho uvicorn (ffmpeg con kế thừa affinity), hoặc extend `_FFMPEG_NICE_PREFIX` (app.py:174-176) thành `['taskset','-c',FFMPEG_CPUSET,'nice','-n','19']` qua env. Arms: (a) nice19-only control, (b) affinity-only, (c) affinity+nice19 — tách hai cơ chế đúng yêu cầu REPORT ("chưa đủ claim quan hệ nhân quả"). True-short 24×4 n≥5 (CV là metric chính, baseline 2.33%) + long 20×10 n=3 + 1 mixed wave.
- Tín hiệu: short CV <2%, mean +1–3%; long không giảm; flat → nice19 đã đủ, đóng hướng.
- Gate: zero errors; long regression <1%; FFmpeg service time không tăng >5%; encode queue so baseline C10 (~9.4s); kill switch env.
- Chi phí: **giờ**.

**19. Cherry-pick #1031: WAV fast-path serialization** *(papers)*
- Cơ chế: pin serialize WAV response ~6.3ms/request vs 0.2ms với #1031 — pure serialization bit-identical, nằm trên cùng CPU threads với AR resolve loop nên giảm jitter đúng chỗ block 4 đo. (MP3-always là ràng buộc bridge output; WAV engine→bridge nội bộ không bị ràng buộc.)
- Cách chạy: check pin có #1031 chưa (`git log`); nếu chưa: cherry-pick 1 commit, bench long 20×10 n≥3 paired, log serialization time per chunk + histogram fetch latency bridge-side.
- Tín hiệu: 6.3→0.2ms/chunk; wave p95 −0.5–2%; null trên mean tok/s vẫn chấp nhận (CPU pressure giảm miễn phí).
- Gate: sha256 WAV payload identical 20 chunk sampled; 0 errors full wave; ffprobe mọi file sampled.
- Chi phí: **giờ**.

**20. Cherry-pick #574 (length-bucketed vocoder batching) rồi #1071 (out-of-process vocoder + compile_decode)** *(gộp: fish E3 + papers cherry-pick-#1071)*
- Cơ chế: giảm HOST-side contention giữa vocoder và AR loop (GIL/process isolation + cross-request batching) — khác cơ chế per-call kernel-launch mà #1209/#798 đo flat; torch.compile rejection là whole-model AR backbone, không cover compile submodule codec non-AR static-shape. d957-wholesale rejection không áp cho diff 2 commit chọn lọc.
- Cách chạy: diff đúng commit #574 và #1071 vs pin df62e91 (đụng RAS/sampler → dừng); apply trong worktree riêng, 2 patch độc lập 2 kill switch (mirror workflow syncfree/). Bench: chỉ-#574 trước (bit-exact, rẻ hơn), rồi +#1071; long 20×10 + mixed soft_reserved, n≥4; đo riêng first-job latency (warmup compile phải <~10s).
- Tín hiệu: #574: +3–8% req/s nếu vocoder serialization đang chặn; #1071: thêm vài % + p95 long giảm, variance decode-step giảm trong nsys. Cả hai flat → đóng vĩnh viễn nhánh vocoder, khớp #1209.
- Gate: #574 claim bit-exact → PCM hash per chunk 100%; #1071: prefix 26/26 + duration + ffprobe + 0 early-termination (~340 tok/chunk); 392/392 sạch; rollback env+restart.
- Chi phí: **ngày** — CHỈ chạy nếu block 4 (census) hoặc block 24 (vocoder knobs) cho tín hiệu vocoder/host-contention dương; nếu không, EV thấp.

---

## TIER 3 — Kỳ vọng thấp/likely-flat nhưng rẻ để đóng có evidence

**21. Vocoder/encoder stage knobs: width 96→128 + wait {0,2,5}** *(gộp 3: engine-knobs stage-width-128 + engine-knobs micro-batch-wait + papers batch-window-sweep)*
- Cơ chế: cap engine 128 nhưng stage 0/1/3 vẫn DEFAULT_MAX_CONCURRENCY=96 (stages.py:86, grep-verified — sửa premise "default 4" của block papers) → khi >96 stream, mỗi tick vocoder tách 96+32 hai đợt. Align width + sweep wait window. Demote vì #1209/#798 gợi ý vocoder không nằm trên critical path non-streaming — nhưng cơ chế queueing-burst chưa bị cover nên không giết.
- Cách chạy: `--stages.preprocessing.factory_args.max_concurrency 128 --stages.audio_encoder.factory_args.max_batch_size 128 --stages.vocoder.factory_args.max_batch_size 128`; arm riêng `max_batch_wait_ms {0,5}` cho vocoder+encoder (chạy tách block để không confound). Sustained 20×10 + true-short n≥3-4 paired.
- Tín hiệu: +0–3% chỉ khi #running >96; wait 0: short p95 giảm chục–trăm ms; flat cả loạt → đóng vĩnh viễn knob vocoder pin.
- Gate: 800/800 sạch/arm; quality corpus + prefix cohort seeded; per-item waveform equivalence giữa arms; VRAM peak <92GB.
- Chi phí: **giờ**.

**22. FFmpeg input qua tmpfs thay stdin pipe** *(bridge E5)*
- Cơ chế: 4 call site dùng `communicate(wav_bytes)` → event loop bơm 0.5–1.5MB qua pipe 64KB mỗi invocation; file /dev/shm viết trong worker thread giải phóng loop. Giá trị phụ thuộc kết quả block 1 (nếu uvloop inactive→active đã ăn phần lớn win này).
- Cách chạy: helper env-gate `HIGGS_FFMPEG_INPUT=file|pipe` cho 2 call site nóng; A/B long 10×10 + true-short + sustained 20×10 n≥5; instrument loop-lag (overshoot của sleep(0.05)).
- Tín hiệu: loop-lag p99 giảm; wall 0–3%, mạnh nhất ở sustained; loop-lag không đổi → reject rẻ, giữ pipe.
- Gate: MP3 byte-identical cùng input (hash 10 mẫu); /dev/shm không leak; fallback pipe khi lỗi ghi; zero failures.
- Chi phí: **giờ**.

**23. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** *(engine-knobs)*
- Cơ chế: vocoder/encoder eager với batch shape đổi liên tục → fragmentation, thi thoảng cudaMalloc/Free đồng bộ stall cả process. PyTorch khuyến nghị đúng cho "frequently changing batch sizes".
- Cách chạy: env launcher, restart; XÁC NHẬN graph capture 128-bucket thành công lúc boot (fail → reject ngay); long 20×10 n=4 + true-short n=3 + ghi peak VRAM; 3 wave sustained kiểm leak.
- Tín hiệu: +0–2% tok/s, CV p95 giảm; flat nhưng VRAM −≥1GB vẫn đáng giữ.
- Gate: capture OK; zero errors; deterministic prefix exact; VRAM ổn định qua sustained.
- Chi phí: **giờ**.

**24. `CUDA_DEVICE_MAX_CONNECTIONS=32`** *(engine-knobs)*
- Cơ chế: ≥3 stream GPU active (AR graph replay, vocoder, encoder, async-copy) trên default 8 hardware queue → false dependency có thể serialize ngầm. Khác Green-Contexts (reserve SM) và MPS-DP (multi-process) — không bị cover.
- Cách chạy: env launcher, restart; long 20×10 n=4 + mixed n=2 paired. Flat → bỏ (knob có cost nhẹ per-connection).
- Tín hiệu: +0–2%, p95 mixed giảm nhẹ nếu false-dependency thật.
- Gate: zero errors; deterministic prefix exact; VRAM không tăng đáng kể.
- Chi phí: **giờ**.

**25. Chunk text-splitting balance policy (bench-only, forward cho worker team)** *(bridge E6)*
- Cơ chế: job wall = max(chunk completion); DEFAULT_LONG_CHUNKS lệch ~2.5×; re-pack đều theo `_expected_tokens` giảm max-chunk → p95 giảm mà không đổi serving.
- Cách chạy: không đổi app: arm A = natural chunks, arm B = cùng text re-pack 10 chunk gần đều token (dùng `_subsplit_pack` offline); paired 10×10 n≥5 cùng seed/reference; verify token parity <2%.
- Tín hiệu: p95 arm B −5–15% theo tỷ lệ giảm max-chunk; >3% với token parity → forward policy cho worker team.
- Gate: token totals ±2%; packing chỉ cắt tại sentence boundary; nghe spot-check boundary.
- Chi phí: **giờ**.

**26. Fused measure+encode một invocation cho anchor chunk** *(bridge E3 — conditional)*
- Cơ chế: gộp 2 ffmpeg run tuần tự của anchor (measure rồi encode) thành 1 bằng asplit + 2 output; bỏ 1 spawn + 1 decode + rút ngắn critical path mà mọi sibling chờ. CHỈ còn giá trị cho cold-voice sau khi block 16 land (sidecar warm đã bỏ measure) — chạy cuối, chỉ nếu cold-voice traffic đáng kể.
- Cách chạy: `_measure_and_encode_anchor()` env-gate `HIGGS_LN_FUSED=1`; validate tay 3 WAV (JSON parse + MP3 hợp lệ + loudness) rồi A/B n≥5 với FFMPEG_TIMING_ENABLED=1.
- Tín hiệu: −1 invocation/job, anchor path −0.3–0.5s; true-short p95 giảm.
- Gate: anchor MP3 ±1 LU vs siblings và vs control; measured JSON ±0.1 LU vs standalone trên corpus 13 file; rescue flip 0; 392-chunk sạch; nghe boundary anchor/sibling (anchor dùng dynamic single-pass ~0.5 LU kém precision — chỉ trên 1 chunk/job).
- Chi phí: **giờ**.

---

## ĐÃ LOẠI / GỘP

**Bị giết bởi prior evidence: 0/38.** Đối chiếu trung thực với REPORT-20260730.md: mọi rejection cũ đều là cơ chế khác với đề xuất tương ứng — K20-rejection đo tại cap 96 (bảng "GPU/scheduler candidate matrix", dòng K16→K20) không cover K24/32@cap128; TRTLLM-rejection (early termination −91%, dòng "Triton prefill + TRTLLM MHA decode") là backend swap, không cover tune nội bộ Triton; torch.compile-rejection (long −3.51%, first job 43s) là whole-model AR, không cover compile_decode codec; #1209/#798 (mục "Next highest-value work": "non-streaming end-to-end gần như flat") là per-call graph capture, không cover batch-window/process-isolation — dùng để DEMOTE block 20/21 xuống conditional/Tier 3, không đủ để giết; C14/C16-rejection là pool concurrency, không cover encoder settings/affinity/transport.

**14 mục gộp vì trùng cơ chế:**
- fish E2 + papers "D2H sync-point audit" + papers "Decode-step CPU budget profile" → **block 4** (một pass profiling chung).
- fish E4 + engine-knobs "triton-decode-kv-splits" → **block 13**.
- papers "Triton cache persistence" + engine-knobs "triton-cache-persist" → **block 3**.
- fish E8 + papers "CPU core isolation" + engine-knobs "cpu-affinity-partition" + bridge E8 → **block 18**.
- fish E9 + papers "Reference-codes persistence" → **block 8**.
- fish E3 + papers "Cherry-pick #1071" → **block 20**.
- papers "Vocoder batch-window/bucket sweep" + engine-knobs "vocoder/encoder micro-batch wait sweep" + engine-knobs "stage-width-alignment-128" → **block 21** (kèm sửa premise sai: pin thực tế 96/2ms, không phải 4/2ms).
- papers "Prefill-coalescing re-sweep" + engine-knobs "prefill-token-budget-sweep" → **block 14** (cùng harness, hai họ knob).

**2 ghi chú thứ tự bắt buộc:** block 6 (RAS audit) chạy trước block 7 (overlap) và block 17 (maxtok cap) — nếu r0 tail là bug RAS-window thì fix correctness trước khi đo throughput/đắp guard; block 20 (#574/#1071, cost ngày) chỉ chạy khi block 4/21 cho tín hiệu host-contention dương.

## CẬP NHẬT ĐỢT 2 (chiều muộn 31/07) — blocks 6, 7, E2 + phát hiện determinism

| Block | Trạng thái | Kết quả |
|---|---|---|
| 6. RAS batch-invariance | ✅ ĐÓNG (code-level PASS) | Census xác nhận RAS buffers pool-indexed, thuần GPU → batch-invariant by construction. PHÁT HIỆN QUAN TRỌNG: không thể gate byte-exact trên arm production — 3 probe idle (seed+top_k=1) ra 3 hash khác nhau: tree pin KHÔNG honor per-request seed (khớp ghi chú REPORT về d957) + RAS resample phi định đoán. Gate byte-exact cần arm riêng TEST_RAS_WIN_LEN. Probe: /root/autodl-tmp/det_probe.py. |
| 7. Overlap schedule | ❌ REJECT (có bằng chứng) | Patch env-gate `SGLANG_OMNI_HIGGS_OVERLAP` (stages.py:573, default off, giữ trong tree — diff hash 207767b8). A/B paired n=3 vs width128: tok/s −0.22% (noise), p50 +3.82%, p95 +3.29% TỆ ĐI nhất quán 3/3. Kết luận: async-decode #590 đã ăn trọn win "pingpong" của Fish; chồng overlap sglang chỉ thêm queue latency. Đóng vĩnh viễn. |
| E2. LAME compression_level | 📊 MICROBENCH XONG | Per-encode (sample ~10s, threads 1): default 57ms, cl=5 53ms, cl=7 45ms (−21%). Đáng làm A/B + gate NGHE trước khi đổi (cl cao = quality thấp hơn một bậc psychoacoustic). |

Arm cuối trên test: `final-cap128-lru512-disk-width128` (overlap off). Diff hash pin hiện tại: 207767b8.
Chưa chạy (còn trong queue): census-patch prefill-sync (model_runner.py:403/412), tolist-on-hit, ln-persist per-voice, maxtok guard, CPU affinity, cherry-pick #574/#1071 (conditional), concurrency matrix + gate_prefix.py, boot-warmup encoder, arm deterministic RAS-off.

## CẬP NHẬT ĐỢT 3: LRU 512 → 2048

Theo yêu cầu operator (churn prod ~1k voice mới/ngày → cửa sổ ~2 ngày):
`stages.py:80-83` giờ là codes 2048 items/2GB, waveform 2048 items/4GB. Diff
hash pin: `b1c134de…`. Arm: `final-cap128-lru2048-disk-width128`. Smoke warm:
5,699.7 tok/s, p95 12.932s, 200/200 sạch. RAM test box: 1TB (937GB available)
— trần 6GB của cache là không đáng kể; cap không preallocate, chỉ lớn khi đủ
voice thật chảy qua. Lưu ý: sau khi có disk-cache, miss RAM-LRU chỉ còn tốn
~chục ms (đọc đĩa + waveform load) thay vì 1.6–1.8s re-encode — LRU size giờ
là tối ưu bậc hai, không phải đòn bẩy chính nữa.
