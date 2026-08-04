# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""MindSpeed-Ops arch32 GDN chunk kernels adapted for FLA triton-ascend backends."""

from __future__ import annotations

import torch

from fla.utils import input_guard


@input_guard
def chunk_scaled_dot_kkt_fwd_npu(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    from fla.ops.common.backends.triton_ascend.mindspeed_ops.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )

    del chunk_indices
    _, _, H, _ = k.shape
    if beta is not None and beta.shape[2] != H:
        raise ValueError(
            "MindSpeed-Ops arch32 chunk_scaled_dot_kkt does not support GVA (HV != H); "
            "set FLA_TRITON_ASCEND_IMPL=fla for GVA workloads."
        )
    return chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=output_dtype,
    )


@input_guard
def chunk_gated_delta_rule_fwd_h_npu(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    from fla.ops.common.backends.triton_ascend.mindspeed_ops.chunk_gated_delta_rule_fwd_h import (
        chunk_gated_delta_rule_fwd_h,
    )

    del state_v_first, cu_seqlens_cpu, chunk_indices
    return chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens,
    )


@input_guard
def chunk_fwd_o_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    from fla.ops.common.backends.triton_ascend.mindspeed_ops.chunk_fwd_o import chunk_fwd_o

    del state_v_first, chunk_indices
    return chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )


@input_guard
def chunk_bwd_dv_local_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    A: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    from fla.ops.common.backends.triton_ascend.mindspeed_ops.chunk_bwd_dv_local import (
        chunk_bwd_dv_local,
    )

    del A, chunk_indices
    return chunk_bwd_dv_local(
        q=q,
        k=k,
        do=do,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )


@input_guard
def chunk_bwd_dqkwg_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    from fla.ops.common.backends.triton_ascend.mindspeed_ops.chunk_bwd_dqkwg import (
        chunk_bwd_dqkwg,
    )

    del state_v_first, chunk_indices
    return chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v,
        do=do,
        h=h,
        dh=dh,
        g=g,
        g_gamma=g_gamma,
        dv=dv,
        w=w,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        scale=1.0 if scale is None else scale,
    )


@input_guard
def chunk_gated_delta_rule_bwd_dhu_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from fla.ops.common.backends.triton_ascend.mindspeed_ops.chunk_gated_delta_rule_bwd_dhu import (
        chunk_gated_delta_rule_bwd_dhu,
    )

    del state_v_first
    return chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
        h0=h0,
        dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
