# Production runtime snapshot — 2026-07-19

Snapshot này là baseline trước khi thử tăng sweet spot cao hơn 96. Production chỉ được đọc; không có process hay file nào trên production bị sửa.

## Source thực đang chạy

- Bridge: `/root/autodl-tmp/Fish-Audio/app.py`
- SGLang-Omni: detached commit `df62e91a00d383e6f73ab9604386ffac6c520529`
- Dirty patch production: 6 file, aggregate `git diff` SHA-256 `9a3ba2d6f6b8459e631b488b76eb5a9a96432ed32edc5dcab770789dd4ef6ad4`
- Model: `bosonai/higgs-audio-v3-tts-4b`
- GPU production: RTX PRO 6000 Blackwell Server Edition, 97,887 MiB

Resolved launch argv:

```text
/root/autodl-tmp/Fish-Audio/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 6006
/root/autodl-tmp/sglang-omni/.venv/bin/sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --host 127.0.0.1 --port 8000 --allowed-local-media-path /root/autodl-tmp/tts-cache --stages.2.factory_args.server_args_overrides.attention_backend triton
```

Production chạy concurrency 96 thật trong SGLang, không chỉ ở HTTP wrapper:

- `DEFAULT_MAX_CONCURRENCY=96`
- AR `max_running_requests=96`
- AR `cuda_graph_max_bs=96`
- preprocessing/audio encoder/vocoder batch 96
- sampler pool 128
- attention backend override `triton`
- log production đã quan sát `#running-req: 96`

## Hash file

| File | SHA-256 |
|---|---|
| `app.py` | `328851b1d77dacf921963376fe15c16255335e68fca75863b981fb11d45cf033` |
| `model.py` | `855cca0d1ab15bce98b34a516234f150cbc882f93bfaa8f4f9efc3845a6b4fb9` |
| `model_runner.py` | `3ae11e91c779b2439a8e407e6480920cf5a7a01b929bcff9451d4ec92ab0e8f4` |
| `payload_types.py` | `56015bc7a5559b23dddcae09039ff51fca9baa95d8a0269b96f4d1e1bb14b8f0` |
| `sampler.py` | `ffbe920740e2223532b4ee9636c27f6d1abe4bea33c83640f2a81799e057bf9a` |
| `stages.py` | `20a0bbbc9d328f42262f5040d0c52509b2ceef25bffcdff7a4caa3c650da6e3f` |
| `text_tokenizer.py` | `6f44c52ac2056bcf6967ad89fa36331f5d28f06f76155adad139f02cd9b8679d` |
| `config.py` | `cbaf236433efc7251daebe0b975dfc1da54257cfed847ad62925388b34e96686` |
| `vocoder_scheduler.py` | `f1020d031dd91b1d7161c125b49683fed67f210bfd089c9868faad7724ddda87` |
| `runtime/higgs-api.sh` | `08b1395f851cd3b8af9ead8b7b4f4c371730dbf4548bffe6f4bff5841f291baa` |
| `runtime/higgs-sglang.sh` | `cbd4e4ca970e0b3d86110d7238f13db9a7008ce969176e3b1b5032f3a5667665` |
| `runtime/higgs-keepalive.sh` | `1531bcd97bea54b915e70da574562c576411f120e1c6cb6d05840f73c42e429a` |
| `runtime/supervisord.conf` | `69052a503777f3c976f79bf38430acb7940192d71391ab1309bd3c01d8bfe32d` |

`runtime/runtime.env.example` là snapshot non-secret của env hiệu lực. Raw `.env` production và `API_TOKEN` không được copy về local.

Mapping snapshot:

| Production | Local |
|---|---|
| `/root/autodl-tmp/Fish-Audio/app.py` | `prod-current/app.py` |
| `/root/autodl-tmp/sglang-omni/sglang_omni/models/higgs_tts/*.py` | `prod-current/*.py` theo cùng basename |
| `/root/autodl-tmp/bin/higgs-{api,sglang,keepalive}.sh` | `prod-current/runtime/` |
| `/root/autodl-tmp/supervisor/supervisord.conf` | `prod-current/runtime/supervisord.conf` |

Trong đó `config.py` và `vocoder_scheduler.py` là support baseline sạch tại commit production, không thuộc 6 file dirty patch. Chúng được lưu để đối chiếu các thử nghiệm scheduler/vocoder sau này.

Không sync: raw `.env`/token, venv Linux/CUDA, model weights, Hugging Face cache, audio cache, logs, dump canary và file `.bak.*` trên production.

## Phiên bản runtime production

- Python 3.12.13
- `sglang-omni 0.1.0` editable
- `sglang 0.5.12.post1`
- `torch 2.11.0+cu130`, `torchaudio 2.11.0`, `triton 3.6.0`
- `flashinfer-python 0.6.11.post1`, `transformers 5.6.0`
- FFmpeg 4.4.2

Local macOS chỉ dùng để lưu source/config và test bridge/FFmpeg. Benchmark CUDA/SGLang phải chạy trên máy test GPU.

## Máy test production-sim

- Worktree source: `/root/autodl-tmp/sglang-omni-prod-sim`
- Runtime overlay: `/root/autodl-tmp/prod-sim/runtime.env`
- Backup canary trước sync: `/root/autodl-tmp/backups/pre-prod-sync-20260719-2230`
- Repo canary cũ `/root/autodl-tmp/sglang-omni` được giữ nguyên để rollback/đối chiếu.

Máy test vẫn khác production ở host layer: driver 595.71.05 thay vì 580.95.05 và Python 3.12.3 thay vì 3.12.13. Source SGLang, cap/graph, attention backend, bridge config và supervisor policy mới là các phần được đồng bộ để benchmark.
