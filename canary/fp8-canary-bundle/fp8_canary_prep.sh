#!/bin/bash
# fp8_canary_prep.sh — PROD CANARY box 22405, 28/09/2026, owner-approved. PREPARES ONLY, restarts nothing:
# /root/autodl-tmp/sglang-omni-fp8-canary = copy of sglang-omni-k16-e161d2f + engine patches 01/02/03 + fish_scales_ops.
set -euo pipefail
B=/root/autodl-tmp/fp8-canary-bundle
C=/root/autodl-tmp/sglang-omni-fp8-canary
SRC=/root/autodl-tmp/sglang-omni-k16-e161d2f
PY=/root/autodl-tmp/sglang-omni/.venv/bin/python
[ -e $C ] && { echo "$C exists already, refusing"; exit 1; }
cd $B && sha256sum -c SHA256SUMS
cp -a $SRC $C
cd $C
for p in 01-fso-mxfp8-hook 02-s3_decode_nosync 03-g22_sampler_tail; do patch -p1 --no-backup-if-mismatch < $B/$p.patch > /dev/null && echo "applied $p"; done
tar xzf $B/fish_scales_ops-75285f0-local.tgz -C $C
md5sum sglang_omni/models/higgs_tts/{fso_mxfp8.py,model.py,model_runner.py,sampler.py}
sha256sum fish_scales_ops/_C*.so
$PY -m py_compile sglang_omni/models/higgs_tts/{fso_mxfp8.py,model.py,model_runner.py,sampler.py} && echo PY_COMPILE_OK
cd /tmp && PYTHONPATH=$C nice -n 10 $PY -c "
import torch, fish_scales_ops as f
x = torch.randn(16, 2560, device='cuda', dtype=torch.bfloat16); w = (torch.randn(6144, 2560, device='cuda') * 0.02).to(torch.bfloat16)
xq, sx = f.gemm.quantize_1x32_fp8(x); wq, sw = f.gemm.quantize_1x32_fp8(w)
y = f.gemm.linear_mxfp8(xq, wq, sx, sw).float(); r = x.float() @ w.float().T
print('fso import', f.__file__, '| sm120 MXFP8 GEMM rel err vs bf16', round(((y - r).norm() / r.norm()).item(), 4))
" 2>&1 | grep -v Warning | tail -2
echo PREP_DONE
