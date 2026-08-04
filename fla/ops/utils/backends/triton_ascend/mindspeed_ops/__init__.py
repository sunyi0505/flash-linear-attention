# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""MindSpeed-Ops arch32 cumsum kernels adapted for FLA triton-ascend backends."""

from __future__ import annotations

import torch

from fla.utils import input_guard


@input_guard
def chunk_global_cumsum_npu(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    from fla.ops.utils.backends.triton_ascend.mindspeed_ops.cumsum import chunk_global_cumsum

    return chunk_global_cumsum(
        s=s,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        scale=scale,
        head_first=head_first,
        output_dtype=output_dtype,
    )


@input_guard
def chunk_local_cumsum_npu(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    from fla.ops.utils.backends.triton_ascend.mindspeed_ops.cumsum import chunk_local_cumsum

    return chunk_local_cumsum(
        g=g,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        output_dtype=output_dtype,
        chunk_indices=chunk_indices,
        **kwargs,
    )
