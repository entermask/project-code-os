# Production runtime snapshot — 2026-11-24

Snapshot đồng bộ từ bản production `a8h3` đang chạy (chỉ đọc; không sửa tiến trình production).

## Source thực đang chạy

- HTTP Wrapper: `Fish-Audio` (`/root/autodl-tmp/Fish-Audio/{app.py,drain_wrapper.py}`, `scripts/run_api.sh` dùng `UVICORN_LOOP=asyncio`).
- SGLang-Omni Engine: commit gốc `df62e91a00d383e6f73ab9604386ffac6c520529` + patch FP8 canary + prefill coalesce + `fish_scales_ops` (`75285f0` + CuTeDSL/MXFP8 SM120). Repo git đồng bộ tại `/Users/thinhj/Documents/sglang-omni-fp8-canary` (nhánh `prod/fp8-canary`).
- Model: `bosonai/higgs-audio-v3-tts-4b` (pinned `model.safetensors` OID `2f7965264c360b38180885006944aa16bd1de20f4e6cff79f6473bfcf8ae3d5a`, `9,309,834,930` bytes).
- GPU production fleet: 3 bridge `RTX PRO 6000D` (`a8h3`, `8cw5`, `eaxk`), `DEFAULT_MAX_CONCURRENCY=128`, `HIGGS_FSO_MXFP8=all`, `HIGGS_DECODE_NOSYNC=1`, `HIGGS_SAMPLER_TAIL=1`, `HIGGS_PAUSE_PREFIX=1`, `LEAD_SILENCE_KEEP_MS=200`.

## Hash file (`prod-current/`, SHA-256)

| File | SHA-256 |
|---|---|
| `app.py` | `dbc232739127c569284b3bb913ff84942a4ae23a8d5bf07c91dffe339c3caace` |
| `drain_wrapper.py` | `8f40fd2b1b919125132573a8a180b9a40bf51ea16044660b0b26be3dbb6a4ceb` |
| `config.py` | `cbaf236433efc7251daebe0b975dfc1da54257cfed847ad62925388b34e96686` |
| `fso_mxfp8.py` | `dcdbfd0484f25bb961676e7242b69cd81d9d07fd99eb0ea1825f940b148ff1fd` |
| `model.py` | `7e31bb1bf081ec1513845c0fcdaeb9251e01945edacc88a2b16963e3b2381ff3` |
| `model_runner.py` | `0273813dbbe5f8ec7509604261b262a170039c8268910974d47a2d7223611873` |
| `omni_scheduler.py` | `ce0707c75ef193184b65af705192160ee25c694c13122bfd7e7733e5c026d6f3` |
| `payload_types.py` | `56015bc7a5559b23dddcae09039ff51fca9baa95d8a0269b96f4d1e1bb14b8f0` |
| `request_builders.py` | `f94cf858385466a400c66b937413fc8a10011b266df5ab9e5f6f0692eeb68ca4` |
| `sampler.py` | `54ecd0f94c045efd8b290b5e1074453839587fde9f4d9e9b44cefd81b970b61a` |
| `stages.py` | `10e01414c392c740f1565b23ccafdfce8323921d46a926d545818f00d66a52a1` |
| `text_tokenizer.py` | `6f44c52ac2056bcf6967ad89fa36331f5d28f06f76155adad139f02cd9b8679d` |
| `vocoder_scheduler.py` | `f1020d031dd91b1d7161c125b49683fed67f210bfd089c9868faad7724ddda87` |
| `runtime/ffmpeg-opt.sh` | `1997905dbe5f30a8885a3c23c8f7df1266107b52b3137f2da8cea7967b5c4924` |
| `runtime/higgs-api.sh` | `6f1a323ea794278973d5de3a5cea834dd6e029dea5bcdff82e37b31a36fa3d61` |
| `runtime/higgs-keepalive.sh` | `842d0075191d8457f3a682ca55344a6123e36c53ad4b4d92d05bce2bbc9f6503` |
| `runtime/higgs-restart.sh` | `969564ae5ee5c89b9b3f9f339c45dd8138e306f881d396fe5a91260a6fd9b703` |
| `runtime/higgs-sglang.sh` | `680e665841a8d5ee3e5d0f4a736c54e3b80d46847ce5a2c25a21c5a9af7f5576` |
| `runtime/higgs-up.sh` | `ffe12cc8c33e57d945db9e3c49d6642b681785151f143469d9bbde38e1d38b63` |
| `runtime/supervisord.conf` | `69052a503777f3c976f79bf38430acb7940192d71391ab1309bd3c01d8bfe32d` |

## Ghi chú thư mục đồng bộ thêm

- `canary/fp8-canary-bundle/`: chứa `01-fso-mxfp8-hook.patch`, `02-s3_decode_nosync.patch`, `03-g22_sampler_tail.patch`, `tune_table.txt`, `SHA256SUMS` và các script rollout/rollback `fp8_canary_*.sh`.
- `prod-current/fish_scales_ops/`: mã nguồn Python/CuTeDSL sạch của `fish_scales_ops` (bản nhị phân `.tgz` lưu cục bộ ngoài Git tại `.local-bundles/fish_scales_ops-75285f0-local.tgz`).
- Không sync: raw `.env`/token, venv Linux/CUDA, model weights, Hugging Face cache, audio cache, logs, dump canary và file `.bak.*`.
