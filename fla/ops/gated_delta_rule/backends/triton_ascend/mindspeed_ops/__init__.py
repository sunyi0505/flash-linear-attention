# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""MindSpeed-Ops arch32 GDN WY kernels adapted for FLA triton-ascend backends."""

from __future__ import annotations

import torch

from fla.utils import input_guard


@input_guard
def recompute_w_u_fwd_npu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from fla.ops.gated_delta_rule.backends.triton_ascend.mindspeed_ops.recompute_w_u_fwd import (
        recompute_w_u_fwd,
    )

    del chunk_indices
    return recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        gk=None,
        cu_seqlens=cu_seqlens,
    )


@input_guard
def prepare_wy_repr_bwd_npu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    from fla.ops.gated_delta_rule.backends.triton_ascend.mindspeed_ops.wy_fast import (
        prepare_wy_repr_bwd,
    )

    del chunk_indices
    if g is None:
        raise ValueError("MindSpeed-Ops arch32 prepare_wy_repr_bwd requires g")
    return prepare_wy_repr_bwd(
        k=k,
        v=v,
        g=g,
        beta=beta,
        A=A,
        dw=dw,
        du=du,
        cu_seqlens=cu_seqlens,
        chunk_size=A.shape[-1],
    )
