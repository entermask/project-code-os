# SPDX-License-Identifier: Apache-2.0
"""Test-only Fish Scales Ops MXFP8 adapter for the Higgs Qwen3 MLP."""

from __future__ import annotations

import gc
import logging
import os
from typing import Any

import torch
from torch import nn


logger = logging.getLogger(__name__)

_EXPECTED_LAYERS = 36
_EXPECTED_HIDDEN = 2560
_EXPECTED_INTERMEDIATE = 9728


def fso_mxfp8_requested() -> bool:
    """Parse the canary flag strictly so malformed values fail startup."""
    raw = os.environ.get("HIGGS_FSO_MXFP8", "0")
    if raw not in {"0", "1"}:
        raise ValueError(
            "HIGGS_FSO_MXFP8 must be exactly '0' or '1'; "
            f"got {raw!r}"
        )
    return raw == "1"


class HiggsFsoQwen3MLP(nn.Module):
    """Qwen3 MLP with an FSO MXFP8 gate-up and native BF16 SiLU/down.

    The native projection children retain their original names so SGLang's
    post-load traversal still sees the same module structure. FSO's packed
    scales are opaque architecture-native tensors and must not be cloned,
    reshaped, made contiguous, or serialized.
    """

    def __init__(
        self,
        native_mlp: nn.Module,
        *,
        gate_up_weight_fp8: torch.Tensor,
        gate_up_scale: torch.Tensor,
        ops: Any,
    ) -> None:
        super().__init__()
        self.gate_up_proj = native_mlp.gate_up_proj
        self.down_proj = native_mlp.down_proj
        self.act_fn = native_mlp.act_fn
        self._fso_ops = ops

        self.register_buffer(
            "_fso_gate_up_weight_fp8",
            gate_up_weight_fp8,
            persistent=False,
        )
        self.register_buffer(
            "_fso_gate_up_scale",
            gate_up_scale,
            persistent=False,
        )
    def forward(
        self,
        x: torch.Tensor,
        forward_batch: Any = None,
    ) -> torch.Tensor:
        if (
            x.dtype != torch.bfloat16
            or x.ndim < 2
            or x.shape[-1] != _EXPECTED_HIDDEN
            or not x.is_cuda
        ):
            raise RuntimeError(
                "FSO Higgs MLP requires a CUDA BF16 [..., 2560] activation; "
                f"got shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"
            )
        if (
            self._fso_gate_up_weight_fp8.dtype != torch.float8_e4m3fn
            or self._fso_gate_up_scale.dtype != torch.int32
        ):
            raise RuntimeError("FSO Higgs MLP buffers changed dtype after install")

        output_shape = x.shape
        x_2d = x.reshape(-1, _EXPECTED_HIDDEN)
        xq, sx = self._fso_ops.quantize_1x32_fp8(x_2d)
        gate_up = self._fso_ops.linear_mxfp8(
            xq,
            self._fso_gate_up_weight_fp8,
            sx,
            self._fso_gate_up_scale,
        )
        hidden = self.act_fn(gate_up)
        output, _ = self.down_proj(
            hidden,
            forward_batch=forward_batch,
        )
        return output.reshape(output_shape)

    def forward_bf16(
        self,
        x: torch.Tensor,
        forward_batch: Any = None,
    ) -> torch.Tensor:
        """Reference path used only by isolated correctness probes."""
        gate_up, _ = self.gate_up_proj(x)
        hidden = self.act_fn(gate_up)
        output, _ = self.down_proj(hidden, forward_batch=forward_batch)
        return output


def _require_native_weight(
    layer_index: int,
    projection: nn.Module,
    expected_shape: tuple[int, int],
    label: str,
) -> torch.Tensor:
    weight = getattr(projection, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError(
            f"Layer {layer_index} {label} has no native weight tensor"
        )
    if tuple(weight.shape) != expected_shape:
        raise RuntimeError(
            f"Layer {layer_index} {label} shape mismatch: "
            f"expected {expected_shape}, got {tuple(weight.shape)}"
        )
    if (
        weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or weight.layout != torch.strided
        or not weight.is_contiguous()
    ):
        raise RuntimeError(
            f"Layer {layer_index} {label} must be contiguous CUDA BF16; "
            f"got dtype={weight.dtype}, device={weight.device}, "
            f"layout={weight.layout}, contiguous={weight.is_contiguous()}"
        )
    if getattr(projection, "bias", None) is not None:
        raise RuntimeError(f"Layer {layer_index} {label} unexpectedly has bias")
    return weight


def _validate_quantized(
    label: str,
    source: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    rows, cols = source.shape
    expected_scale_shape = (rows, cols // 128)
    expected_scale_stride = (1, rows)
    if (
        tuple(qweight.shape) != tuple(source.shape)
        or qweight.dtype != torch.float8_e4m3fn
        or qweight.device != source.device
        or not qweight.is_contiguous()
    ):
        raise RuntimeError(
            f"{label} FSO qweight layout mismatch: "
            f"shape={tuple(qweight.shape)}, dtype={qweight.dtype}, "
            f"device={qweight.device}, stride={qweight.stride()}"
        )
    if (
        tuple(scale.shape) != expected_scale_shape
        or scale.dtype != torch.int32
        or scale.device != source.device
        or tuple(scale.stride()) != expected_scale_stride
        or scale.is_contiguous()
        or scale.storage_offset() != 0
    ):
        raise RuntimeError(
            f"{label} FSO scale layout mismatch: "
            f"shape={tuple(scale.shape)}, dtype={scale.dtype}, "
            f"device={scale.device}, stride={scale.stride()}, "
            f"contiguous={scale.is_contiguous()}, "
            f"storage_offset={scale.storage_offset()}"
        )


def install_fso_mxfp8_mlp(model: nn.Module, quant_config: Any) -> dict[str, int]:
    """Validate the exact Higgs topology, then install FSO adapters.

    Validation is a separate first pass: no decoder layer is mutated until
    every topology, shape, dtype, parallelism, and device guard has passed.
    Any conversion failure is fatal to this process; there is no mixed
    BF16/FSO fallback.
    """
    if quant_config is not None:
        raise RuntimeError("Custom FSO MXFP8 cannot be combined with quant_config")

    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.server_args import get_global_server_args

    tp_group = get_tp_group()
    if int(tp_group.world_size) != 1:
        raise RuntimeError(
            f"Custom FSO MXFP8 canary requires TP=1, got {tp_group.world_size}"
        )
    pp_group = model.backbone.pp_group
    if int(pp_group.world_size) != 1:
        raise RuntimeError(
            f"Custom FSO MXFP8 canary requires PP=1, got {pp_group.world_size}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Custom FSO MXFP8 canary requires CUDA")
    capability = torch.cuda.get_device_capability()
    if tuple(capability) != (12, 0):
        raise RuntimeError(
            f"Custom FSO MXFP8 canary requires SM120, got {capability}"
        )
    server_args = get_global_server_args()
    if bool(getattr(server_args, "enable_torch_compile", False)):
        raise RuntimeError(
            "Custom FSO MXFP8 canary requires torch.compile to be disabled"
        )
    if bool(getattr(server_args, "enable_lora", False)) or bool(
        getattr(server_args, "lora_paths", None)
    ):
        raise RuntimeError("Custom FSO MXFP8 canary does not support LoRA")
    if getattr(server_args, "rl_on_policy_target", None) is not None:
        raise RuntimeError(
            "Custom FSO MXFP8 canary does not support RL on-policy mode"
        )

    text_config = model.config.get_text_config()
    config_contract = {
        "num_hidden_layers": _EXPECTED_LAYERS,
        "hidden_size": _EXPECTED_HIDDEN,
        "intermediate_size": _EXPECTED_INTERMEDIATE,
        "hidden_act": "silu",
    }
    for name, expected in config_contract.items():
        actual = getattr(text_config, name, None)
        if actual != expected:
            raise RuntimeError(
                f"FSO Higgs config mismatch for {name}: "
                f"expected {expected!r}, got {actual!r}"
            )

    layers = model.backbone.model.layers
    if len(layers) != _EXPECTED_LAYERS:
        raise RuntimeError(
            f"FSO Higgs layer count mismatch: expected {_EXPECTED_LAYERS}, "
            f"got {len(layers)}"
        )

    for index, layer in enumerate(layers):
        native_mlp = getattr(layer, "mlp", None)
        if isinstance(native_mlp, HiggsFsoQwen3MLP):
            raise RuntimeError(f"Layer {index} already has an FSO MLP adapter")
        if native_mlp is None:
            raise RuntimeError(f"Layer {index} has no native MLP")
        gate_up = getattr(native_mlp, "gate_up_proj", None)
        down = getattr(native_mlp, "down_proj", None)
        if gate_up is None or down is None:
            raise RuntimeError(
                f"Layer {index} does not expose native Qwen3 projections"
            )
        _require_native_weight(
            index,
            gate_up,
            (2 * _EXPECTED_INTERMEDIATE, _EXPECTED_HIDDEN),
            "gate_up_proj",
        )
        _require_native_weight(
            index,
            down,
            (_EXPECTED_HIDDEN, _EXPECTED_INTERMEDIATE),
            "down_proj",
        )

    try:
        import fish_scales_ops as fso
    except Exception as exc:
        raise RuntimeError("Unable to import pinned fish_scales_ops") from exc

    required_ops = (
        "quantize_1x32_fp8",
        "linear_mxfp8",
    )
    for name in required_ops:
        if not callable(getattr(fso.gemm, name, None)):
            raise RuntimeError(f"fish_scales_ops.gemm.{name} is unavailable")

    allocated_before = torch.cuda.memory_allocated()
    probe = torch.linspace(
        -0.125,
        0.125,
        _EXPECTED_HIDDEN,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(1, _EXPECTED_HIDDEN)
    minimum_probe_cosine = 1.0
    for index, layer in enumerate(layers):
        native_mlp = layer.mlp
        gate_up_weight = native_mlp.gate_up_proj.weight.detach()
        down_weight = native_mlp.down_proj.weight.detach()

        gate_up_q, gate_up_scale = fso.gemm.quantize_1x32_fp8(
            gate_up_weight
        )
        torch.cuda.current_stream(gate_up_weight.device).synchronize()

        _validate_quantized(
            f"layer {index} gate_up",
            gate_up_weight,
            gate_up_q,
            gate_up_scale,
        )
        adapter = HiggsFsoQwen3MLP(
            native_mlp,
            gate_up_weight_fp8=gate_up_q,
            gate_up_scale=gate_up_scale,
            ops=fso.gemm,
        )
        with torch.no_grad():
            reference = adapter.forward_bf16(probe)
            candidate = adapter(probe)
            finite = bool(torch.isfinite(candidate).all().item())
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    candidate.float().flatten(),
                    reference.float().flatten(),
                    dim=0,
                ).item()
            )
        if not finite or cosine < 0.99:
            raise RuntimeError(
                f"Layer {index} FSO startup probe failed: "
                f"finite={finite}, cosine={cosine}"
            )
        minimum_probe_cosine = min(minimum_probe_cosine, cosine)
        layer.mlp = adapter
        del gate_up_weight, down_weight, native_mlp

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated()
    stats = {
        "layers": len(layers),
        "allocated_before_bytes": allocated_before,
        "allocated_after_bytes": allocated_after,
        "fso_bytes": allocated_after - allocated_before,
        "minimum_probe_cosine_ppm": round(minimum_probe_cosine * 1_000_000),
    }
    logger.warning(
        "Installed test-only FSO MXFP8 Higgs gate-up adapters: "
        "layers=%d, allocated_before=%d, allocated_after=%d, "
        "fso_bytes=%d, minimum_probe_cosine=%.8f",
        stats["layers"],
        stats["allocated_before_bytes"],
        stats["allocated_after_bytes"],
        stats["fso_bytes"],
        minimum_probe_cosine,
    )
    return stats
