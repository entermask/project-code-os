#!/usr/bin/env bash
set -euo pipefail

SGLANG_OMNI_DIR="${SGLANG_OMNI_DIR:-$HOME/sglang-omni}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - "$SGLANG_OMNI_DIR" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])

stages = root / "sglang_omni/models/fishaudio_s2_pro/stages.py"
tokenizer = root / "sglang_omni/models/fishaudio_s2_pro/tokenizer.py"
content = root / "sglang_omni/models/fishaudio_s2_pro/fish_speech/content_sequence.py"

for path in (stages, tokenizer, content):
    if not path.exists():
        raise SystemExit(f"Missing expected SGLang-Omni file: {path}")

text = stages.read_text()
old = '''    codec = _load_codec(checkpoint_dir, "cpu")

    def _encode_reference_audio(audio_path: str) -> torch.Tensor:
        import torchaudio

        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, codec.sample_rate)
        audios = audio.squeeze(0).unsqueeze(0)
        audio_lengths = torch.tensor([audios.shape[1]], dtype=torch.long)
        with torch.no_grad():
            indices, _ = codec.encode(audios, audio_lengths)
            if indices.ndim == 3:
                indices = indices[0]
        return indices.cpu()
'''
new = '''    codec = _load_codec(checkpoint_dir, "cpu")

    from collections import OrderedDict
    import threading

    ref_vq_cache_size = max(0, int(os.environ.get("S2PRO_REF_VQ_CACHE_SIZE", "128")))
    ref_vq_cache: OrderedDict[tuple[str, int, int], torch.Tensor] = OrderedDict()
    ref_vq_cache_lock = threading.Lock()

    def _reference_cache_key(audio_path: str) -> tuple[str, int, int] | None:
        try:
            stat = os.stat(audio_path)
        except OSError:
            return None
        return (os.path.abspath(audio_path), stat.st_mtime_ns, stat.st_size)

    def _encode_reference_audio(audio_path: str) -> torch.Tensor:
        cache_key = _reference_cache_key(audio_path) if ref_vq_cache_size else None
        if cache_key is not None:
            with ref_vq_cache_lock:
                cached = ref_vq_cache.get(cache_key)
                if cached is not None:
                    ref_vq_cache.move_to_end(cache_key)
                    return cached

        import torchaudio

        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, codec.sample_rate)
        audios = audio.squeeze(0).unsqueeze(0)
        audio_lengths = torch.tensor([audios.shape[1]], dtype=torch.long)
        with torch.no_grad():
            indices, _ = codec.encode(audios, audio_lengths)
            if indices.ndim == 3:
                indices = indices[0]
        indices = indices.cpu()

        if cache_key is not None:
            with ref_vq_cache_lock:
                ref_vq_cache[cache_key] = indices
                ref_vq_cache.move_to_end(cache_key)
                while len(ref_vq_cache) > ref_vq_cache_size:
                    ref_vq_cache.popitem(last=False)
        return indices
'''
if "ref_vq_cache_size = max(0, int(os.environ.get(\"S2PRO_REF_VQ_CACHE_SIZE\"" not in text:
    if old not in text:
        raise SystemExit("Could not patch stages.py; expected reference encode block was not found.")
    stages.write_text(text.replace(old, new))

text = tokenizer.read_text()
old = '''            if all_codes:
                combined = torch.cat(all_codes, dim=1)
                system_parts.append(VQPart(codes=combined, cal_loss=False))
'''
new = '''            if all_codes:
                combined = all_codes[0] if len(all_codes) == 1 else torch.cat(all_codes, dim=1)
                system_parts.append(VQPart(codes=combined, cal_loss=False))
'''
if "combined = all_codes[0] if len(all_codes) == 1 else torch.cat" not in text:
    if old not in text:
        raise SystemExit("Could not patch tokenizer.py; expected all_codes block was not found.")
    tokenizer.write_text(text.replace(old, new))

text = content.read_text()
old = '''                curr_codes = part.codes.clone().to(torch.int)
                tokens = torch.tensor(
                    tokenizer.convert_tokens_to_ids(
                        [
                            SEMANTIC_TOKEN_TEMPLATE.format(i=i)
                            for i in curr_codes[0].int()
                        ]
                    ),
                    dtype=torch.int,
                )
                vq_parts.append(curr_codes)
'''
new = '''                curr_codes = part.codes.clone().to(torch.int)
                semantic_base_id = tokenizer.convert_tokens_to_ids(
                    SEMANTIC_TOKEN_TEMPLATE.format(i=0)
                )
                tokens = (curr_codes[0].int() + semantic_base_id).to(torch.int)
                vq_parts.append(curr_codes)
'''
if "tokens = (curr_codes[0].int() + semantic_base_id).to(torch.int)" not in text:
    if old not in text:
        raise SystemExit("Could not patch content_sequence.py; expected VQ token block was not found.")
    content.write_text(text.replace(old, new))

print("Patched SGLang-Omni Fish S2 Pro preprocessing optimizations.")
PY
