"""TEST-ONLY: swap the Higgs Qwen3 backbone linears to fish-scales-ops MXFP8 (sm_120a) right after weights load.

HIGGS_FSO_MXFP8 = "" (off, default) | "mlp" (gate_up + down) | "all" (mlp + qkv + o).
Weights are quantized once (MXFP8 1x32, UE8M0 scales) before CUDA-graph capture and KV-pool sizing; the bf16
copies of patched weights are released. Everything else (embeddings, audio heads, norms, attention math) stays bf16.

Hybrid / diagnostic knobs (28/09 naturalness follow-up; unset = the behaviour above):
  HIGGS_FSO_PROJ="qkv,o,gate_up,down"  subset of projections to run in MXFP8 (default from the mode)
  HIGGS_FSO_SKIP_LAYERS="0,35"         decoder layers kept fully bf16
  HIGGS_FSO_FAKE="w"|"a"               diagnostic: bf16 GEMMs, MXFP8 quantize-dequantize of weights only / inputs only
"""
from __future__ import annotations

import logging
import os
import types

import torch

logger = logging.getLogger(__name__)


def _quantize(weight: torch.Tensor):
    import fish_scales_ops as fso

    return fso.gemm.quantize_1x32_fp8(weight.detach().contiguous())


# HIGGS_FSO_DEBUG: "" | "cmp" (keep bf16 weights, log cos fp8-vs-bf16, return fp8) | "ref" (same, but return bf16)
_DEBUG = os.environ.get("HIGGS_FSO_DEBUG", "").strip().lower()
_DEBUG_CALLS: dict[int, int] = {}
_FUSED_LAYERS = {int(x) for x in os.environ.get("HIGGS_FSO_FUSED_LAYERS", "").split(",") if x.strip()}
_ALL_PROJ = ("qkv", "o", "gate_up", "down")
_PROJ_ENV = {x.strip() for x in os.environ.get("HIGGS_FSO_PROJ", "").split(",") if x.strip()}
_SKIP_LAYERS = {int(x) for x in os.environ.get("HIGGS_FSO_SKIP_LAYERS", "").split(",") if x.strip()}
_FAKE = os.environ.get("HIGGS_FSO_FAKE", "").strip().lower()


def _fake_mx(t: torch.Tensor) -> torch.Tensor:
    """MXFP8 1x32 quantize-dequantize along the last dim with fso's rule (RCEIL UE8M0 scale, saturate to +-448)."""
    x = t.float().reshape(*t.shape[:-1], t.shape[-1] // 32, 32)
    s = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0)))
    return ((x / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s).reshape(t.shape).to(t.dtype)


def _fake_patch(lin: torch.nn.Module) -> None:
    if _FAKE == "w":
        lin.weight.data.copy_(_fake_mx(lin.weight.data))
        return
    orig = lin.forward
    lin.forward = lambda x, *args, **kwargs: orig(_fake_mx(x), *args, **kwargs)


def _release(param: torch.nn.Parameter) -> None:
    if _DEBUG and _DEBUG != "nan":
        return
    param.data = torch.empty(0, dtype=param.dtype, device=param.device)


def _nan_probe(module: torch.nn.Module, x: torch.Tensor, gu: torch.Tensor, y: torch.Tensor) -> None:
    if torch.cuda.is_current_stream_capturing():
        return
    n = _DEBUG_CALLS.get(id(module), 0)
    _DEBUG_CALLS[id(module)] = n + 1
    torch.cuda.synchronize()
    bad = lambda t: bool((~torch.isfinite(t.float())).any().item())
    if n < 4 or bad(y) or bad(x):
        logger.warning("FSO_NAN layer=%s call=%d M=%d stride=%s x_bad=%s x_absmax=%.3g gu_bad=%s y_bad=%s y_absmax=%.3g",
                       module._fso_layer, n, x.shape[0], tuple(x.stride()), bad(x), x.float().nan_to_num().abs().max().item(),
                       bad(gu), bad(y), y.float().nan_to_num().abs().max().item())


def _debug(module: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    n = _DEBUG_CALLS.get(id(module), 0)
    _DEBUG_CALLS[id(module)] = n + 1
    if n == 0 and module._fso_layer in (1, 2) and os.environ.get("HIGGS_FSO_DUMP"):
        torch.save(x.detach().cpu(), f"{os.environ['HIGGS_FSO_DUMP']}/x_{type(module).__name__}_L{module._fso_layer}.pt")
    if n < 2 or (module._fso_layer in (0, 1, 18, 35) and n % 500 == 0):
        cos = torch.nn.functional.cosine_similarity(y.float().flatten(), ref.float().flatten(), dim=0).item()
        logger.warning("FSO_DEBUG layer=%s %s call=%d M=%d x_absmax=%.3g x_stride=%s y_absmax=%.3g ref_absmax=%.3g cos=%.5f",
                       module._fso_layer, type(module).__name__, n, x.shape[0], x.float().abs().max().item(), tuple(x.stride()),
                       y.float().abs().max().item(), ref.float().abs().max().item(), cos)
    return ref if _DEBUG == "ref" else y


def _patch_mlp(mlp: torch.nn.Module, proj: set[str]) -> None:
    import fish_scales_ops as fso

    gu_q = gu_s = d_q = d_s = None
    if "gate_up" in proj:
        gu_q, gu_s = _quantize(mlp.gate_up_proj.weight)
    if "down" in proj:
        d_q, d_s = _quantize(mlp.down_proj.weight)
    hidden = mlp.down_proj.weight.shape[0]
    mlp._fso = (gu_q, gu_s, d_q, d_s, hidden)
    if gu_q is not None:
        _release(mlp.gate_up_proj.weight)
    if d_q is not None:
        _release(mlp.down_proj.weight)

    def forward(self, x, forward_batch=None):
        gu_q, gu_s, d_q, d_s, hidden = self._fso
        if x.shape[0] == 0:
            return x.new_empty((0, hidden))
        if gu_q is None:
            gu, _ = self.gate_up_proj(x)
        else:
            xq, sx = fso.gemm.quantize_1x32_fp8(x.contiguous())
            gu = fso.gemm.linear_mxfp8(xq, gu_q, sx, gu_s)
        if d_q is None:  # hybrid: down projection stays bf16
            y, _ = self.down_proj(self.act_fn(gu))
            return y
        # NOT fso.gemm.silu_chunk_mul_quantize_1x32_fp8: at fso 75285f0 the fused kernel picks wrong UE8M0 scales on
        # real Qwen3 activations (layers 1-4 MLP cos 0.72-0.94 vs bf16). bf16 SiluAndMul + plain quantize costs one
        # extra M x 9728 bf16 round-trip, negligible next to the 150 MB of weights per layer at M <= 128.
        # HIGGS_FSO_FUSED_LAYERS re-enables the buggy fused kernel on chosen layers: positive control for screens only.
        if self._fso_layer in _FUSED_LAYERS:
            hq, sh = fso.gemm.silu_chunk_mul_quantize_1x32_fp8(gu)
        else:
            hq, sh = fso.gemm.quantize_1x32_fp8(self.act_fn(gu))
        y = fso.gemm.linear_mxfp8(hq, d_q, sh, d_s)
        if _DEBUG == "nan":
            _nan_probe(self, x, gu, y)
        elif _DEBUG:
            return _debug(self, x, y, type(self).forward(self, x, forward_batch=forward_batch))
        return y

    mlp.forward = types.MethodType(forward, mlp)


def _patch_linear(lin: torch.nn.Module) -> None:
    import fish_scales_ops as fso

    w_q, w_s = _quantize(lin.weight)
    out_features = lin.weight.shape[0]
    lin._fso = (w_q, w_s, out_features)
    _release(lin.weight)

    def forward(self, x, *args, **kwargs):
        w_q, w_s, out_features = self._fso
        if x.shape[0] == 0:
            return x.new_empty((0, out_features)), None
        xq, sx = fso.gemm.quantize_1x32_fp8(x.contiguous())
        y = fso.gemm.linear_mxfp8(xq, w_q, sx, w_s)
        if _DEBUG:
            return _debug(self, x, y, type(self).forward(self, x, *args, **kwargs)[0]), None
        return y, None

    lin.forward = types.MethodType(forward, lin)


def maybe_enable_fso_mxfp8(backbone: torch.nn.Module) -> None:
    mode = os.environ.get("HIGGS_FSO_MXFP8", "").strip().lower()
    if mode not in ("mlp", "all"):
        return
    proj = _PROJ_ENV or ({"gate_up", "down"} if mode == "mlp" else set(_ALL_PROJ))
    assert proj <= set(_ALL_PROJ) and _FAKE in ("", "w", "a"), (proj, _FAKE)
    layers = backbone.model.layers
    for idx, layer in enumerate(layers):
        if idx in _SKIP_LAYERS:
            continue
        mods = {"qkv": layer.self_attn.qkv_proj, "o": layer.self_attn.o_proj,
                "gate_up": layer.mlp.gate_up_proj, "down": layer.mlp.down_proj}
        if _FAKE:
            for p in proj:
                _fake_patch(mods[p])
            continue
        if proj & {"gate_up", "down"}:
            layer.mlp._fso_layer = idx
            _patch_mlp(layer.mlp, proj)
        for p in ("qkv", "o"):
            if p in proj:
                mods[p]._fso_layer = idx
                _patch_linear(mods[p])
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    logger.warning("HIGGS_FSO_MXFP8=%s: patched %d decoder layers to fish-scales-ops MXFP8 (fused-swiglu control layers: %s) proj=%s skip=%s fake=%s",
                   mode, len(layers) - len(_SKIP_LAYERS & set(range(len(layers)))), sorted(_FUSED_LAYERS) or "none",
                   ",".join(p for p in _ALL_PROJ if p in proj), sorted(_SKIP_LAYERS) or "none", _FAKE or "none")
