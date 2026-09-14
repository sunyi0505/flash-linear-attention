# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CANN 32-size DMA padding for Triton-Ascend workspace tensors.

NPU block_ptr leftover of a tile is rounded to 32. If the last-dim stride is
not a multiple of ``max(tile, 32)``, that leftover aliases the next row.

A partial last tile is worse: leftover DMA starts at logical ``n`` and is 32
wide, so the row must also satisfy ``n + 32 <= padded`` (e.g. D=60, tile=64
needs 128, not 64).
"""

from __future__ import annotations

import torch
import triton

NPU_ALIGN = 32


def npu_pad(n: int, tile: int = NPU_ALIGN) -> int:
    """Pad ``n`` so leftover DMA of ``tile`` cannot alias the next row."""
    n = max(int(n), 1)
    align = max(int(tile), NPU_ALIGN)
    padded = triton.cdiv(n, align) * align
    if n % NPU_ALIGN != 0 and n + NPU_ALIGN > padded:
        padded += align
    return padded


def npu_leftover_mask(
    *,
    T: int | None = None,
    BT: int | None = None,
    K: int | None = None,
    BK: int | None = None,
    V: int | None = None,
    BV: int | None = None,
    varlen: bool = False,
) -> bool:
    """True if a kernel must zero leftover lanes (partial last tile / varlen)."""
    if varlen:
        return True
    return any(
        n is not None and b is not None and int(n) % int(b) != 0
        for n, b in ((T, BT), (K, BK), (V, BV))
    )


def npu_unpad(t: torch.Tensor | None, *valid: int) -> torch.Tensor | None:
    """Slice trailing dims back to logical sizes after a padded store."""
    if t is None:
        return None
    if all(t.shape[-i] == d for i, d in enumerate(reversed(valid), 1)):
        return t
    return t[(...,) + tuple(slice(0, d) for d in valid)].contiguous()
