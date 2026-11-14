# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend chunk GKA solve.

The stock kernel keeps a dense `[BK, BK]` state next to `[BT, BK]` iterates and does not fit in
UB once `BK > 64`. Frobenius still streams `H` in `[BS, BS]` blocks, in its own launch: the
Chebyshev loop keeps a resident `[BK, BK]` state and full `[BT, BK]` iterates, and the two live
sets together overflow UB.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.cann.extension import extract_slice, insert_slice

    if not hasattr(tl, 'extract_slice'):
        tl.extract_slice = extract_slice
    if not hasattr(tl, 'insert_slice'):
        tl.insert_slice = insert_slice
except ImportError:
    pass

from fla.backends import TritonAscendBackend, register
from fla.ops.common.backends.triton_ascend.chunk_h import (
    _packed_varlen_split_segments,
    _slice_seq,
    chunk_fwd_h_npu,
    chunk_fwd_h_verifier,
)
from fla.ops.common.chunk_h import chunk_fwd_h
from fla.ops.gka.chunk_solve_fwd import chunk_gka_solve_fwd
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import ascend_compile_kwargs, get_multiprocessor_count
from fla.utils.ascend_ub_manager import ASCEND_MAX_GRID_DIM

_BK = 128
_BS = 64
_MAX_CHUNK = 64


@triton.jit
def _qk_panel(
    ptr, origin, head, stride, n_tokens, K, panel,
    BT: tl.constexpr, BS: tl.constexpr,
):
    rows = tl.arange(0, BT)
    cols = panel * BS + tl.arange(0, BS)
    base = tl.cast(origin, tl.int64) * stride + head
    off = base + rows[:, None] * stride + cols[None, :]
    mask = (rows[:, None] < n_tokens) & (cols[None, :] < K)
    return tl.load(ptr + off, mask=mask, other=0)


@triton.jit
def _store_panel(
    ptr, origin, head, stride, n_tokens, K, panel, value,
    BT: tl.constexpr, BS: tl.constexpr,
):
    rows = tl.arange(0, BT)
    cols = panel * BS + tl.arange(0, BS)
    base = tl.cast(origin, tl.int64) * stride + head
    off = base + rows[:, None] * stride + cols[None, :]
    mask = (rows[:, None] < n_tokens) & (cols[None, :] < K)
    tl.store(ptr + off, value, mask=mask)


@triton.jit
def _h_sub(h_ptr, row_panel, col_panel, K, BS: tl.constexpr):
    rows = row_panel * BS + tl.arange(0, BS)
    cols = col_panel * BS + tl.arange(0, BS)
    mask = (rows[:, None] < K) & (cols[None, :] < K)
    return tl.load(h_ptr + rows[:, None] * K + cols[None, :], mask=mask, other=0)


@triton.jit
def _as_lhs(tile):
    # tl.dot reuses the left operand's UB storage. A copy keeps the source panel readable.
    return tile + 0


@triton.jit
def _qk_full(
    ptr, origin, head, stride, n_tokens, K,
    BT: tl.constexpr, BK: tl.constexpr,
):
    rows = tl.arange(0, BT)
    cols = tl.arange(0, BK)
    base = tl.cast(origin, tl.int64) * stride + head
    off = base + rows[:, None] * stride + cols[None, :]
    mask = (rows[:, None] < n_tokens) & (cols[None, :] < K)
    return tl.load(ptr + off, mask=mask, other=0)


@triton.jit
def _store_full(
    ptr, origin, head, stride, n_tokens, K, value,
    BT: tl.constexpr, BK: tl.constexpr,
):
    rows = tl.arange(0, BT)
    cols = tl.arange(0, BK)
    base = tl.cast(origin, tl.int64) * stride + head
    off = base + rows[:, None] * stride + cols[None, :]
    mask = (rows[:, None] < n_tokens) & (cols[None, :] < K)
    tl.store(ptr + off, value, mask=mask)


@triton.jit
def _h_cols(h_ptr, col_panel, K, BK: tl.constexpr, BS: tl.constexpr):
    rows = tl.arange(0, BK)
    cols = col_panel * BS + tl.arange(0, BS)
    mask = (rows[:, None] < K) & (cols[None, :] < K)
    return tl.load(h_ptr + rows[:, None] * K + cols[None, :], mask=mask, other=0)


@triton.heuristics({
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'task_offset', 'BH', 'NT'])
def _gka_fro_kernel(
    k, h, gk, fro, cu_seqlens, chunk_indices,
    T, task_offset, BH, NT,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    BK: tl.constexpr, BS: tl.constexpr,
    USE_GK: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_task = tl.cast(tl.program_id(0), tl.int64) + task_offset
    i_outer = i_task // BH
    i_inner = i_task % BH
    i_h = i_inner % H
    if IS_VARLEN:
        i_tg = i_outer
        i_n = tl.load(chunk_indices + i_tg * 2).to(tl.int64)
        i_t = tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = eos - bos
    else:
        i_b = i_inner // H
        i_t = i_outer
        i_tg = tl.cast(i_b, tl.int64) * NT + i_t
        bos = tl.cast(i_b, tl.int64) * T
        seq_len = T

    n_tokens = tl.minimum(seq_len - i_t * BT, BT)
    origin = bos + tl.cast(i_t, tl.int64) * BT
    stride = tl.cast(H, tl.int64) * K
    head = tl.cast(i_h, tl.int64) * K
    rows = tl.arange(0, BT)
    row_mask = rows < n_tokens
    h_ptr = h + (tl.cast(i_tg, tl.int64) * H + i_h) * K * K
    if USE_GK:
        if IS_VARLEN:
            g_off = tl.cast(i_h, tl.int64) * T + origin + rows
        else:
            g_off = (tl.cast(i_b, tl.int64) * H + i_h) * T + tl.cast(i_t, tl.int64) * BT + rows
        b_g = tl.load(gk + g_off, mask=row_mask, other=0).to(tl.float32)
        b_m = tl.where(row_mask[:, None] & row_mask[None, :] & (
            rows[:, None] >= rows[None, :]), exp2(b_g[:, None] - b_g[None, :]), 0)
        b_gh = tl.where(row_mask, exp2(b_g), 0)
    else:
        b_m = tl.where(row_mask[:, None] & row_mask[None, :] & (rows[:, None] >= rows[None, :]), 1.0, 0.0)
        b_gh = tl.where(row_mask, 1.0, 0.0)

    b_kk = tl.zeros([BT, BT], dtype=tl.float32)
    for panel in tl.static_range(BK // BS):
        b_kp = _qk_panel(k, origin, head, stride, n_tokens, K, panel, BT, BS).to(tl.float32)
        b_kk += tl.dot(_as_lhs(b_kp), tl.trans(b_kp))
    b_kk2 = b_kk * b_kk
    b_fro = tl.sum(tl.dot(_as_lhs(b_m), b_kk2) * b_m, 1)
    h_sq = 0.0
    for row_panel in tl.static_range(BK // BS):
        for col_panel in tl.static_range(BK // BS):
            sub = _h_sub(h_ptr, row_panel, col_panel, K, BS).to(tl.float32)
            h_sq += tl.sum(sub * sub)
    b_fro += b_gh * (b_gh * h_sq)
    b_khk = tl.zeros([BT], dtype=tl.float32)
    for col_panel in tl.static_range(BK // BS):
        b_part = tl.zeros([BT, BS], dtype=tl.float32)
        for row_panel in tl.static_range(BK // BS):
            b_kr = _qk_panel(k, origin, head, stride, n_tokens, K, row_panel, BT, BS).to(tl.float32)
            sub = _h_sub(h_ptr, row_panel, col_panel, K, BS).to(tl.float32)
            b_part += tl.dot(_as_lhs(b_kr), sub)
        b_kc = _qk_panel(k, origin, head, stride, n_tokens, K, col_panel, BT, BS).to(tl.float32)
        b_khk += tl.sum(b_kc * b_part, 1)
    b_fro = tl.sqrt(b_fro + 2.0 * b_gh * tl.sum(b_m * b_khk[None, :], 1))
    tl.store(fro + (origin + rows) * H + i_h, tl.where(row_mask, b_fro, 1.0), mask=row_mask)


@triton.jit
def _chebyshev_step(
    x_f, p_f, b_w, b_k, rhs, b_m, b_gh, b_lamb, b_step, b_rho, q_f,
    BT: tl.constexpr, BK: tl.constexpr,
):
    # scores = x @ K^T and the state term x @ H share the left operand once gh is applied
    # to the state block afterwards. One [BK, BT+BK] product replaces those two dots.
    b_w = 1.0 / (1.0 - b_rho * b_w)
    prod = tl.dot(x_f.to(b_k.dtype), rhs)
    scores = tl.extract_slice(prod, [0, 0], [BT, BT], [1, 1]) * b_m
    state = tl.extract_slice(prod, [0, BT], [BT, BK], [1, 1]) * b_gh[:, None]
    intra = tl.dot(scores.to(b_k.dtype), b_k)
    residual = intra + state + b_lamb[:, None] * x_f - q_f
    delta = b_step[:, None] * b_w[:, None] * residual + (b_w[:, None] - 1.0) * p_f
    x_next = b_w[:, None] * x_f - delta
    return x_next, x_f, b_w


@triton.heuristics({
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'FIXED_ITERS': lambda args: args['num_iter'] == 30,
})
@triton.jit(do_not_specialize=['T', 'task_offset', 'num_iter', 'BH', 'NT'])
def chunk_gka_solve_fwd_kernel_npu(
    q, k, h, gk, x, fro, cu_seqlens, chunk_indices,
    ridge_ratio, T, task_offset, num_iter, BH, NT,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    BK: tl.constexpr, BS: tl.constexpr,
    USE_GK: tl.constexpr, IS_VARLEN: tl.constexpr, FIXED_ITERS: tl.constexpr,
):
    i_task = tl.cast(tl.program_id(0), tl.int64) + task_offset
    i_outer = i_task // BH
    i_inner = i_task % BH
    i_h = i_inner % H
    if IS_VARLEN:
        i_tg = i_outer
        i_n = tl.load(chunk_indices + i_tg * 2).to(tl.int64)
        i_t = tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = eos - bos
    else:
        i_b = i_inner // H
        i_t = i_outer
        i_tg = tl.cast(i_b, tl.int64) * NT + i_t
        bos = tl.cast(i_b, tl.int64) * T
        seq_len = T

    n_tokens = tl.minimum(seq_len - i_t * BT, BT)
    origin = bos + tl.cast(i_t, tl.int64) * BT
    stride = tl.cast(H, tl.int64) * K
    head = tl.cast(i_h, tl.int64) * K
    rows = tl.arange(0, BT)
    row_mask = rows < n_tokens
    h_ptr = h + (tl.cast(i_tg, tl.int64) * H + i_h) * K * K

    if USE_GK:
        if IS_VARLEN:
            g_off = tl.cast(i_h, tl.int64) * T + origin + rows
        else:
            g_off = (tl.cast(i_b, tl.int64) * H + i_h) * T + tl.cast(i_t, tl.int64) * BT + rows
        b_g = tl.load(gk + g_off, mask=row_mask, other=0).to(tl.float32)
        b_m = tl.where(row_mask[:, None] & row_mask[None, :] & (
            rows[:, None] >= rows[None, :]), exp2(b_g[:, None] - b_g[None, :]), 0)
        b_gh = tl.where(row_mask, exp2(b_g), 0)
    else:
        b_m = tl.where(row_mask[:, None] & row_mask[None, :] & (rows[:, None] >= rows[None, :]), 1.0, 0.0)
        b_gh = tl.where(row_mask, 1.0, 0.0)

    b_fro = tl.load(fro + (origin + rows) * H + i_h, mask=row_mask, other=1.0).to(tl.float32)
    b_lamb = ridge_ratio * b_fro
    b_step = 2.0 / (2.0 * b_lamb + b_fro)
    b_rho = b_fro / (2.0 * b_lamb + b_fro) / 2.0
    b_rho = b_rho * b_rho

    # Full-width iterate: one dot each for scores, state, and the intra term.
    b_q = _qk_full(q, origin, head, stride, n_tokens, K, BT, BK)
    b_k = _qk_full(k, origin, head, stride, n_tokens, K, BT, BK)
    b_h = _h_cols(h_ptr, 0, K, BK, BK)
    rhs = tl.zeros([BK, BT + BK], dtype=b_k.dtype)
    rhs = tl.insert_slice(rhs, tl.trans(b_k), [0, 0], [BK, BT], [1, 1])
    rhs = tl.insert_slice(rhs, b_h.to(b_k.dtype), [0, BT], [BK, BK], [1, 1])
    q_f = b_q.to(tl.float32)
    x_f = b_step[:, None] * q_f
    p_f = tl.zeros([BT, BK], dtype=tl.float32)
    b_w = tl.full([BT], 2.0, dtype=tl.float32)
    if FIXED_ITERS:
        for _ in tl.range(30, loop_unroll_factor=1):
            x_f, p_f, b_w = _chebyshev_step(
                x_f, p_f, b_w, b_k, rhs, b_m, b_gh, b_lamb, b_step, b_rho, q_f, BT, BK,
            )
    else:
        for _ in range(num_iter):
            x_f, p_f, b_w = _chebyshev_step(
                x_f, p_f, b_w, b_k, rhs, b_m, b_gh, b_lamb, b_step, b_rho, q_f, BT, BK,
            )

    _store_full(
        x, origin, head, stride, n_tokens, K,
        x_f.to(x.dtype.element_ty, fp_downcast_rounding='rtne'), BT, BK,
    )


def _run_chunk_solve(
    q: torch.Tensor,
    k: torch.Tensor,
    h: torch.Tensor,
    gk: torch.Tensor | None = None,
    fro: torch.Tensor | None = None,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = q.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    x = torch.empty_like(q)
    bh = H if cu_seqlens is not None else B * H
    # [B, H, T] so a chunk of gates is contiguous. H == 1 does not copy.
    gk_c = None if gk is None else gk.transpose(1, 2)
    if gk_c is not None and not gk_c.is_contiguous():
        gk_c = gk_c.contiguous()
    task_num = NT * bh
    if fro is None and task_num == 0:
        fro = q.new_empty(B, T, H, dtype=torch.float)
    if task_num > 0:
        compile_kwargs = ascend_compile_kwargs(workspace_multibuffer=0)
        get_multiprocessor_count(use_aicore=True)
        if fro is None:
            fro = q.new_empty(B, T, H, dtype=torch.float)
            for offset in range(0, task_num, ASCEND_MAX_GRID_DIM):
                count = min(ASCEND_MAX_GRID_DIM, task_num - offset)
                _gka_fro_kernel[(count,)](
                    k, h, gk_c, fro, cu_seqlens, chunk_indices,
                    T, offset, bh, NT, H, K, BT, _BK, _BS,
                    **compile_kwargs,
                )
        kwargs = dict(
            q=q, k=k, h=h, gk=gk_c, x=x, fro=fro,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            ridge_ratio=ridge_ratio, T=T, num_iter=num_iter,
            H=H, K=K, BT=BT, BH=bh, BK=_BK, BS=_BS, NT=NT,
        )
        for offset in range(0, task_num, ASCEND_MAX_GRID_DIM):
            count = min(ASCEND_MAX_GRID_DIM, task_num - offset)
            chunk_gka_solve_fwd_kernel_npu[(count,)](task_offset=offset, **kwargs, **compile_kwargs)
    return x, fro


def chunk_gka_solve_fwd_verifier(
    q: torch.Tensor,
    k: torch.Tensor,
    h: torch.Tensor,
    gk: torch.Tensor | None = None,
    fro: torch.Tensor | None = None,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[bool, str | None]:
    del h, gk, fro, ridge_ratio, num_iter, cu_seqlens, chunk_indices
    if q.device.type != 'npu':
        return False, 'GKA Ascend chunk solve requires NPU tensors'
    # The stock [BK, BK] state compiles on this NPU only while the padded head dim is <= 64.
    if max(triton.next_power_of_2(q.shape[-1]), 16) <= 64:
        return False, 'stock chunk GKA solve fits in UB'
    if chunk_size > _MAX_CHUNK:
        return False, f'GKA Ascend chunk solve requires chunk_size <= {_MAX_CHUNK}, got {chunk_size}'
    if q.shape[-1] > _BK:
        return False, f'GKA Ascend chunk solve supports K <= {_BK}, got {q.shape[-1]}'
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32) or k.dtype not in (
        torch.float16, torch.bfloat16, torch.float32,
    ):
        return False, 'GKA Ascend chunk solve received an unsupported dtype'
    return True, None


@register(chunk_gka_solve_fwd, backend=TritonAscendBackend, verifier=chunk_gka_solve_fwd_verifier)
def chunk_gka_solve_fwd_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    h: torch.Tensor,
    gk: torch.Tensor | None = None,
    fro: torch.Tensor | None = None,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_chunk_solve(
        q, k, h, gk=gk, fro=fro, ridge_ratio=ridge_ratio, num_iter=num_iter,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices,
    )


# Host windows for the state scan that feeds this solve.
# `chunk_h_fwd_kernel` unrolls NT with `tl.static_range`. NT=256 (T=16384, chunk 64)
# compiles to a 47344-byte AIV frame and is rejected above 32768. NT=128 fits that
# stack when the kernel does not also carry a boundary state. A window sets
# STORE_FINAL_STATE / USE_INITIAL_STATE; that specialization at NT=128 is rejected
# with `out of jump/jumpc imm range`. NT=64 with the carry still compiles.
# Sequences that fit in one NT=128 launch stay on `chunk_fwd_h_npu`. Longer ones
# are split into NT=64 windows that chain the state. The backward scan uses the
# same limits.
_MAX_DIRECT_NT = 128
_MAX_WINDOW_NT = 64


def _aligned_tokens(chunk_size: int, split_size: int | None, max_nt: int) -> int:
    bt = chunk_size
    bs = bt if split_size is None else split_size
    nts = bs // bt
    chunks = max_nt - (max_nt % nts)
    if chunks < nts:
        chunks = nts
    return chunks * bt


def scan_limits(chunk_size: int, split_size: int | None) -> tuple[int, int]:
    direct = _aligned_tokens(chunk_size, split_size, _MAX_DIRECT_NT)
    window = _aligned_tokens(chunk_size, split_size, _MAX_WINDOW_NT)
    return direct, window


def sequence_exceeds(T: int, cu_seqlens: torch.Tensor | None, window: int) -> bool:
    if cu_seqlens is None:
        return window < T
    cu = cu_seqlens if cu_seqlens.device.type == 'cpu' else cu_seqlens.detach().cpu()
    return bool((cu[1:] - cu[:-1]).max().item() > window)


def _fwd_equal(
    k, v, g, g_gamma, gk, gv, h0, output_final_state, state_v_first,
    chunk_size, split_size, states_in_fp32, window: int,
):
    T = k.shape[1]
    if window >= T:
        return chunk_fwd_h_npu(
            k=k, v=v, g=g, g_gamma=g_gamma, gk=gk, gv=gv, h0=h0,
            output_final_state=output_final_state, state_v_first=state_v_first,
            cu_seqlens=None, chunk_size=chunk_size, split_size=split_size,
            states_in_fp32=states_in_fp32,
        )
    parts = []
    carry = h0
    ht = None
    for start in range(0, T, window):
        end = min(start + window, T)
        last = end == T
        h_i, ht_i = chunk_fwd_h_npu(
            k=_slice_seq(k, start, end),
            v=_slice_seq(v, start, end),
            g=_slice_seq(g, start, end),
            g_gamma=g_gamma,
            gk=_slice_seq(gk, start, end),
            gv=_slice_seq(gv, start, end),
            h0=carry,
            output_final_state=output_final_state or not last,
            state_v_first=state_v_first,
            cu_seqlens=None,
            chunk_size=chunk_size,
            split_size=split_size,
            states_in_fp32=states_in_fp32,
        )
        parts.append(h_i)
        if last:
            ht = ht_i if output_final_state else None
        else:
            carry = ht_i
    return torch.cat(parts, dim=1), ht


@register(chunk_fwd_h, backend=TritonAscendBackend, verifier=chunk_fwd_h_verifier)
def chunk_fwd_h_windowed(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    direct, window = scan_limits(chunk_size, split_size)
    if not sequence_exceeds(k.shape[1], cu_seqlens, direct):
        return chunk_fwd_h_npu(
            k=k, v=v, g=g, g_gamma=g_gamma, gk=gk, gv=gv, h0=h0,
            output_final_state=output_final_state, state_v_first=state_v_first,
            cu_seqlens=cu_seqlens, chunk_size=chunk_size, split_size=split_size,
            states_in_fp32=states_in_fp32,
        )
    if cu_seqlens is None:
        return _fwd_equal(
            k, v, g, g_gamma, gk, gv, h0, output_final_state, state_v_first,
            chunk_size, split_size, states_in_fp32, window,
        )

    bs = chunk_size if split_size is None else split_size
    B, _, H, K = k.shape
    V = v.shape[-1]
    _, ns, segments = _packed_varlen_split_segments(cu_seqlens, bs)
    state_shape = (V, K) if state_v_first else (K, V)
    h = k.new_zeros(B, ns, H, *state_shape, dtype=torch.float if states_in_fp32 else k.dtype)
    ht = k.new_zeros(len(segments), H, *state_shape, dtype=torch.float) if output_final_state else None
    for i_n, (bos, eos, boh, n_i) in enumerate(segments):
        if eos <= bos or n_i <= 0:
            if ht is not None and h0 is not None:
                ht[i_n].copy_(h0[i_n])
            continue
        h_i, ht_i = _fwd_equal(
            _slice_seq(k, bos, eos), _slice_seq(v, bos, eos),
            _slice_seq(g, bos, eos), g_gamma, _slice_seq(gk, bos, eos), _slice_seq(gv, bos, eos),
            None if h0 is None else h0[i_n:i_n + 1],
            output_final_state, state_v_first, chunk_size, split_size, states_in_fp32, window,
        )
        h[:, boh:boh + n_i] = h_i
        if ht is not None:
            ht[i_n].copy_(ht_i[0])
    return h, ht
