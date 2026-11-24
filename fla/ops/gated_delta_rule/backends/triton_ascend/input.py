# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused input preparation for short, equally headed GDN sequences."""

import torch
import triton
import triton.language as tl

from fla.utils.ascend_ub_manager import ASCEND_MAX_GRID_DIM


@triton.jit(do_not_specialize=['T', 'BLOCK_OFFSET'])
def gdn_input_fwd_kernel(
    q, k, v, g, beta, qo, ko, vo, go, bo, qr, kr, T, BLOCK_OFFSET,
    H: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr, NORMALIZE: tl.constexpr,
    BR: tl.constexpr,
):
    rows = (tl.program_id(0).to(tl.int64) + BLOCK_OFFSET) * BR + tl.arange(0, BR)
    cols = tl.arange(0, D)
    offsets = rows[:, None] * D + cols[None, :]
    valid = rows < T * H
    qv = tl.load(q + offsets, valid[:, None], other=0).to(tl.float32)
    kv = tl.load(k + offsets, valid[:, None], other=0).to(tl.float32)
    if NORMALIZE:
        qs = 1 / tl.sqrt(tl.sum(qv * qv, 1) + 1e-6)
        ks = 1 / tl.sqrt(tl.sum(kv * kv, 1) + 1e-6)
        qv *= qs[:, None]
        kv *= ks[:, None]
        tl.store(qr + rows, qs, rows < ROWS)
        tl.store(kr + rows, ks, rows < ROWS)
    tl.store(qo + offsets, qv, (rows < ROWS)[:, None])
    tl.store(ko + offsets, kv, (rows < ROWS)[:, None])
    vv = tl.load(v + offsets, valid[:, None], other=0)
    tl.store(vo + offsets, vv, (rows < ROWS)[:, None])
    gv = tl.load(g + rows, valid, other=0)
    bv = tl.load(beta + rows, valid, other=0)
    tl.store(go + rows, gv, rows < ROWS)
    tl.store(bo + rows, bv, rows < ROWS)


def prepare_padded_inputs(q, k, v, g, beta, padded_t, normalize):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1 or q.shape[-1] != 128:
        raise ValueError('fused GDN input preparation requires matching [1, T, H, 128] Q/K/V')
    if g.shape != q.shape[:3] or beta.shape != g.shape or padded_t < q.shape[1]:
        raise ValueError('fused GDN input preparation requires matching gates and nonnegative padding')
    if any(not x.is_contiguous() or x.device != q.device for x in (q, k, v, g, beta)):
        raise ValueError('fused GDN input preparation requires contiguous inputs on the same device')
    _, t, h, d = q.shape
    shape = (1, padded_t, h, d)
    qo, ko, vo = (torch.empty(shape, device=x.device, dtype=x.dtype) for x in (q, k, v))
    go, bo = (torch.empty(shape[:3], device=x.device, dtype=x.dtype) for x in (g, beta))
    qr, kr = (torch.empty(shape[:3], device=q.device, dtype=torch.float32) for _ in range(2)) if normalize else (None, None)
    blocks = triton.cdiv(padded_t * h, 32)
    for offset in range(0, blocks, ASCEND_MAX_GRID_DIM):
        gdn_input_fwd_kernel[(min(blocks - offset, ASCEND_MAX_GRID_DIM),)](
            q, k, v, g, beta, qo, ko, vo, go, bo, qr, kr, t, offset,
            H=h, D=d, ROWS=padded_t * h, NORMALIZE=normalize, BR=32,
        )
    return qo, ko, vo, go, bo, qr, kr
