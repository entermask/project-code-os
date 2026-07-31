#!/usr/bin/env python3
"""Isolated correctness and CUDA-graph smoke test for the Higgs FSO adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


HIDDEN = 2560
INTERMEDIATE = 9728


class _Projection(nn.Module):
    def __init__(self, rows: int, cols: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(
                rows,
                cols,
                dtype=torch.bfloat16,
                device="cuda",
            )
            / (cols**0.5),
            requires_grad=False,
        )
        self.bias = None

    def forward(self, x: torch.Tensor, forward_batch=None):
        del forward_batch
        return F.linear(x, self.weight), None


class _SiluAndMul(nn.Module):
    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate) * up


class _NativeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = _Projection(2 * INTERMEDIATE, HIDDEN)
        self.down_proj = _Projection(HIDDEN, INTERMEDIATE)
        self.act_fn = _SiluAndMul()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.adapter_dir.resolve()))

    import fish_scales_ops as fso
    from fso_mlp import HiggsFsoQwen3MLP

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("adapter_smoke requires SM120")

    torch.manual_seed(20260730)
    native = _NativeMLP()
    wgu_q, sgu = fso.gemm.quantize_1x32_fp8(
        native.gate_up_proj.weight.detach()
    )
    adapter = HiggsFsoQwen3MLP(
        native,
        gate_up_weight_fp8=wgu_q,
        gate_up_scale=sgu,
        ops=fso.gemm,
    )

    x = torch.randn(
        2,
        4,
        HIDDEN,
        dtype=torch.bfloat16,
        device="cuda",
    )
    reference = adapter.forward_bf16(x)
    actual = adapter(x)
    cosine = float(
        F.cosine_similarity(
            actual.float().flatten(),
            reference.float().flatten(),
            dim=0,
        ).item()
    )
    if cosine < 0.997:
        raise RuntimeError(f"adapter cosine gate failed: {cosine}")
    if actual.shape != x.shape:
        raise RuntimeError(
            f"adapter shape gate failed: {actual.shape} != {x.shape}"
        )
    if any(name.startswith("_fso_") for name in adapter.state_dict()):
        raise RuntimeError("FSO opaque buffers leaked into state_dict")

    graph_x = torch.randn(
        16,
        HIDDEN,
        dtype=torch.bfloat16,
        device="cuda",
    )
    graph_reference = adapter(graph_x).clone()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            adapter(graph_x)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_y = adapter(graph_x)
    graph_y.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    replay_bit_exact = bool(torch.equal(graph_reference, graph_y))
    replay_max_abs = float(
        (graph_reference.float() - graph_y.float()).abs().max().item()
    )
    replay_cosine = float(
        F.cosine_similarity(
            graph_reference.float().flatten(),
            graph_y.float().flatten(),
            dim=0,
        ).item()
    )
    if not bool(torch.isfinite(graph_y).all().item()) or replay_cosine < 0.9999:
        raise RuntimeError(
            "adapter CUDA-graph replay stability gate failed: "
            f"cosine={replay_cosine}, max_abs={replay_max_abs}"
        )

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "sm": 120,
                "cosine": cosine,
                "shape": list(actual.shape),
                "cuda_graph_bit_exact": replay_bit_exact,
                "cuda_graph_replay_cosine": replay_cosine,
                "cuda_graph_replay_max_abs": replay_max_abs,
                "opaque_buffers_persistent": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
