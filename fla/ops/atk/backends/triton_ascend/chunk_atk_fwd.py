# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Chunked ATK forward adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl

from fla.utils.ascend_ub_manager import ASCEND_LAUNCH_BLOCK_BUDGET

_NPU_CHUNK_LEN = 16


def atk_forward_pass_chunks_torch(
    a: torch.Tensor,
    sa: torch.Tensor,
    ac: torch.Tensor,
    h0: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_len: int,
):
    """Prefix-scan ATK chunk summaries. ``tl.range`` over chunks is unreliable on NPU."""
    n, n_chunks, h, d = a.shape
    if cu_seqlens is None:
        seq_chunks = [n_chunks] * n
    else:
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        seq_chunks = [int(math.ceil(int(s) / chunk_len)) for s in seqlens]

    for i in range(n):
        nc = seq_chunks[i]
        if nc <= 0:
            continue
        if h0 is None:
            carry = torch.zeros(h, d, device=a.device, dtype=torch.float32)
        else:
            carry = h0[i].float()
        for c in range(nc):
            carry = torch.exp(sa[i, c, :, None]) * carry + a[i, c]
            ac[i, c] = carry


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T', 'chunk_offset'])
def _forward_chunk_summary_npu(
    k,
    beta,
    log_g,
    a,
    sa,
    cu_seqlens,
    chunk_indices,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    k_stride_b, k_stride_t, k_stride_h, k_stride_d,
    beta_stride_b, beta_stride_t, beta_stride_h,
    log_g_stride_b, log_g_stride_t, log_g_stride_h,
    a_stride_b, a_stride_c, a_stride_h, a_stride_d,
    sa_stride_b, sa_stride_c, sa_stride_h,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    chunk_offset,
):
    i_t = tl.program_id(0)
    h = tl.program_id(1)
    chunk_id = tl.program_id(2).to(tl.int64) + chunk_offset

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        chunk_id = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        b = i_n.to(tl.int64)
    else:
        b = tl.program_id(0).to(tl.int64)
        bos = 0
        eos = T

    if h >= H:
        return

    if chunk_id * CHUNK_LEN >= T:
        return

    T_range = chunk_id * CHUNK_LEN + tl.arange(0, CHUNK_LEN)
    mask_T = T_range < T

    if IS_VARLEN:
        beta_ptr = beta + (bos + T_range) * beta_stride_t + h * beta_stride_h
        log_g_ptr = log_g + (bos + T_range) * log_g_stride_t + h * log_g_stride_h
    else:
        beta_ptr = beta + b * beta_stride_b + T_range * beta_stride_t + h * beta_stride_h
        log_g_ptr = log_g + b * log_g_stride_b + T_range * log_g_stride_t + h * log_g_stride_h

    beta_val = tl.load(beta_ptr, mask=mask_T, other=0.0).to(tl.float32)
    log_g_val = tl.load(log_g_ptr, mask=mask_T, other=0.0).to(tl.float32)

    sa_val = tl.sum(log_g_val)
    decays = tl.exp(sa_val - tl.cumsum(log_g_val))

    for i_k in range(tl.cdiv(D, BK)):
        d_offset = i_k * BK
        D_range = d_offset + tl.arange(0, BK)
        mask_D = D_range < D

        if IS_VARLEN:
            k_ptr = k + (bos + T_range)[:, None] * k_stride_t + h * k_stride_h + D_range[None, :] * k_stride_d
        else:
            k_ptr = k + b * k_stride_b + T_range[:, None] * k_stride_t + h * k_stride_h + D_range[None, :] * k_stride_d

        k_val = tl.load(k_ptr, mask=mask_T[:, None] * mask_D[None, :], other=0.0).to(tl.float32)

        k_sq = k_val * k_val
        U = beta_val[:, None] * k_sq
        a_val = tl.sum(decays[:, None] * U, 0)

        a_ptr = a + b * a_stride_b + h * a_stride_h + chunk_id * a_stride_c + D_range * a_stride_d
        tl.store(a_ptr, a_val, mask=mask_D)

    sa_ptr = sa + b * sa_stride_b + h * sa_stride_h + chunk_id * sa_stride_c
    tl.store(sa_ptr, sa_val)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'chunk_offset'])
def _forward_chunk_out_npu(
    k,
    beta,
    log_g,
    ac,
    h0,
    k_precond,
    log_atk_scale,
    logx,
    eps,
    cu_seqlens,
    chunk_indices,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    k_stride_b, k_stride_t, k_stride_h, k_stride_d,
    beta_stride_b, beta_stride_t, beta_stride_h,
    log_g_stride_b, log_g_stride_t, log_g_stride_h,
    ac_stride_b, ac_stride_c, ac_stride_h, ac_stride_d,
    kp_stride_b, kp_stride_t, kp_stride_h, kp_stride_d,
    BK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    chunk_offset,
):
    i_t = tl.program_id(0)
    h = tl.program_id(1)
    chunk_id = tl.program_id(2).to(tl.int64) + chunk_offset

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        chunk_id = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        b = i_n.to(tl.int64)
    else:
        b = tl.program_id(0).to(tl.int64)
        bos = 0
        eos = T

    if h >= H:
        return

    if chunk_id * CHUNK_LEN >= T:
        return

    T_range = chunk_id * CHUNK_LEN + tl.arange(0, CHUNK_LEN)
    mask_T = T_range < T
    C_range = tl.arange(0, CHUNK_LEN)

    if IS_VARLEN:
        beta_ptr = beta + (bos + T_range) * beta_stride_t + h * beta_stride_h
        log_g_ptr = log_g + (bos + T_range) * log_g_stride_t + h * log_g_stride_h
    else:
        beta_ptr = beta + b * beta_stride_b + T_range * beta_stride_t + h * beta_stride_h
        log_g_ptr = log_g + b * log_g_stride_b + h * log_g_stride_h + (log_g_stride_t * T_range)

    beta_val = tl.load(beta_ptr, mask=mask_T, other=0.0).to(tl.float32)
    log_g_val = tl.load(log_g_ptr, mask=mask_T, other=0.0).to(tl.float32)
    g_val = tl.exp(log_g_val)

    la_cumsum = tl.cumsum(log_g_val)

    roll_mat = (C_range[:, None] == (C_range[None, :] + 1)).to(tl.float32)
    la_cumsum_roll = tl.sum(roll_mat[:, :] * la_cumsum[None, :], 1)
    M = tl.exp(la_cumsum_roll[:, None] - la_cumsum[None, :])
    M = tl.where(C_range[:, None] > C_range[None, :], M, 0.0)

    base_decays = tl.exp(la_cumsum_roll * (C_range > 0).to(tl.float32))

    center = tl.load(log_atk_scale + h).to(tl.float32)

    for i_k in range(tl.cdiv(D, BK)):
        d_offset = i_k * BK
        D_range = d_offset + tl.arange(0, BK)
        mask_D = D_range < D

        if IS_VARLEN:
            k_ptr = k + (bos + T_range)[:, None] * k_stride_t + h * k_stride_h + D_range[None, :] * k_stride_d
            kp_ptr = k_precond + (bos + T_range)[:, None] * kp_stride_t + h * kp_stride_h + D_range[None, :] * kp_stride_d
        else:
            k_ptr = k + b * k_stride_b + h * k_stride_h + (k_stride_t * T_range)[:, None] + (k_stride_d) * D_range[None, :]
            kp_ptr = k_precond + b * kp_stride_b + h * kp_stride_h + \
                (kp_stride_t * T_range)[:, None] + (kp_stride_d) * D_range[None, :]

        k_val = tl.load(k_ptr, mask=mask_T[:, None] * mask_D[None, :], other=0.0).to(tl.float32)

        k_sq = k_val * k_val
        U = beta_val[:, None] * k_sq

        ac_ptr = ac + b * ac_stride_b + h * ac_stride_h + (chunk_id - 1) * ac_stride_c + D_range * ac_stride_d

        if chunk_id == 0:
            if USE_INITIAL_STATE:
                ac_val = tl.load(h0 + (b * H + h) * D + D_range, mask=mask_D, other=0).to(tl.float32)
            else:
                ac_val = tl.zeros([BK], dtype=tl.float32)
        else:
            ac_val = tl.load(ac_ptr, mask=mask_D)

        raw_state = base_decays[:, None] * ac_val[None, :] + tl.dot(M, U)

        A_t = g_val[:, None] * raw_state + U

        ell = tl.log(A_t + eps)
        r = ell - center
        s = r / (1.0 + tl.abs(r))
        M_precond = tl.exp(-logx * s)
        k_precond_val = k_val * M_precond

        tl.store(kp_ptr, k_precond_val, mask=mask_T[:, None] * mask_D[None, :])


def _atk_fwd_stages_npu_packed(
    k: torch.Tensor,
    beta: torch.Tensor,
    log_g: torch.Tensor,
    chunk_size: int,
    initial_A_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    x: float,
    eps: float,
    log_atk_scale: torch.Tensor | None,
):
    """Packed varlen ATK on NPU: run the working batched kernels per sequence.

    Triton-Ascend IS_VARLEN kernels for chunk_summary/chunk_out disagree with
    the batched path (and with naive ATK) even for a single packed sequence.
    """
    B, T, H, D = k.shape
    assert B == 1, "packed varlen ATK expects batch size 1"
    CHUNK_LEN = _NPU_CHUNK_LEN
    N = len(cu_seqlens) - 1
    seq_lens = [(int(cu_seqlens[i + 1]) - int(cu_seqlens[i])) for i in range(N)]
    max_chunks = max((math.ceil(s / CHUNK_LEN) if s > 0 else 0) for s in seq_lens)
    max_chunks = max(max_chunks, 1)

    a = torch.zeros(N, max_chunks, H, D, dtype=torch.float32, device=k.device)
    sa = torch.zeros(N, max_chunks, H, dtype=torch.float32, device=k.device)
    ac = torch.zeros(N, max_chunks, H, D, dtype=torch.float32, device=k.device)
    k_precond = torch.zeros_like(k)
    at_rows = []

    for i in range(N):
        bos, eos = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        if eos <= bos:
            if output_final_state:
                at_rows.append(ac[i, 0])
            continue
        h0_i = None if initial_A_state is None else initial_A_state[i:i + 1]
        kp_i, ac_i, a_i, sa_i, at_i = _atk_fwd_stages_npu_batched(
            k[:, bos:eos],
            beta[:, bos:eos],
            log_g[:, bos:eos],
            chunk_size,
            h0_i,
            output_final_state,
            x,
            eps,
            log_atk_scale,
        )
        k_precond[:, bos:eos] = kp_i
        nc = a_i.shape[1]
        a[i, :nc] = a_i[0]
        sa[i, :nc] = sa_i[0]
        ac[i, :nc] = ac_i[0]
        if output_final_state:
            at_rows.append(at_i[0] if at_i is not None else ac[i, nc - 1])

    at = torch.stack(at_rows, dim=0).to(k.dtype) if output_final_state else None
    return k_precond.to(k.dtype), ac, a, sa, at


def _atk_fwd_stages_npu_batched(
    k: torch.Tensor,
    beta: torch.Tensor,
    log_g: torch.Tensor,
    chunk_size: int,
    initial_A_state: torch.Tensor | None,
    output_final_state: bool,
    x: float,
    eps: float,
    log_atk_scale: torch.Tensor | None,
):
    del chunk_size
    B, T, H, D = k.shape
    CHUNK_LEN = _NPU_CHUNK_LEN
    BK = D if os.environ.get('ATK_NO_KTILE') else 32
    num_chunks = math.ceil(T / CHUNK_LEN)

    a = torch.zeros(B, num_chunks, H, D, dtype=torch.float32, device=k.device)
    sa = torch.zeros(B, num_chunks, H, dtype=torch.float32, device=k.device)
    ac = torch.zeros(B, num_chunks, H, D, dtype=torch.float32, device=k.device)
    # NPU tests poison torch.empty* with NaN. Triton-Ascend masked stores can
    # RMW destination lanes, so leftover NaNs leak into k_precond.
    k_precond = torch.zeros_like(k)

    k = k.contiguous()
    beta = beta.contiguous()
    log_g = log_g.contiguous()

    if log_atk_scale is None:
        log_atk_scale = torch.full((H,), -0.2, dtype=torch.float32, device=k.device)
    else:
        log_atk_scale = log_atk_scale.contiguous()

    logx = math.log(x) if x > 0 else 0.0

    def _launch_chunk_grid(kernel, args):
        max_c = max(1, ASCEND_LAUNCH_BLOCK_BUDGET // max(B * H, 1))
        for off in range(0, num_chunks, max_c):
            n = min(max_c, num_chunks - off)
            kernel[(B, H, n)](*args, chunk_offset=off)

    _launch_chunk_grid(
        _forward_chunk_summary_npu,
        (
            k, beta, log_g, a, sa,
            None, None,
            B, T, H, D, CHUNK_LEN,
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            log_g.stride(0), log_g.stride(1), log_g.stride(2),
            a.stride(0), a.stride(1), a.stride(2), a.stride(3),
            sa.stride(0), sa.stride(1), sa.stride(2),
            BK,
        ),
    )

    atk_forward_pass_chunks_torch(a, sa, ac, initial_A_state, None, CHUNK_LEN)

    _launch_chunk_grid(
        _forward_chunk_out_npu,
        (
            k, beta, log_g, ac, initial_A_state, k_precond,
            log_atk_scale, logx, eps,
            None, None,
            B, T, H, D, CHUNK_LEN,
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            log_g.stride(0), log_g.stride(1), log_g.stride(2),
            ac.stride(0), ac.stride(1), ac.stride(2), ac.stride(3),
            k_precond.stride(0), k_precond.stride(1), k_precond.stride(2), k_precond.stride(3),
            BK,
        ),
    )

    at = ac[:, -1, :, :].contiguous().to(k.dtype) if output_final_state else None
    return k_precond.to(k.dtype), ac, a, sa, at


def _atk_fwd_stages_npu(
    k: torch.Tensor,
    beta: torch.Tensor,
    log_g: torch.Tensor,
    chunk_size: int,
    initial_A_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    x: float,
    eps: float,
    log_atk_scale: torch.Tensor | None,
):
    if cu_seqlens is not None:
        return _atk_fwd_stages_npu_packed(
            k, beta, log_g, chunk_size, initial_A_state, output_final_state,
            cu_seqlens, x, eps, log_atk_scale,
        )
    return _atk_fwd_stages_npu_batched(
        k, beta, log_g, chunk_size, initial_A_state, output_final_state,
        x, eps, log_atk_scale,
    )


def chunk_atk_fwd_npu(
    k: torch.Tensor,
    beta: torch.Tensor,
    log_g: torch.Tensor = None,
    chunk_size: int = 64,
    initial_A_state: torch.Tensor = None,
    output_final_state: bool = True,
    cu_seqlens: torch.Tensor = None,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
):
    k_precond, _, _, _, at = _atk_fwd_stages_npu(
        k, beta, log_g, chunk_size,
        initial_A_state, output_final_state, cu_seqlens,
        x, eps, log_atk_scale,
    )
    return k_precond, at


def recompute_atk_fwd_npu(
    k: torch.Tensor,
    beta: torch.Tensor,
    log_g: torch.Tensor,
    chunk_size: int = 64,
    initial_A_state: torch.Tensor = None,
    cu_seqlens: torch.Tensor = None,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
):
    k_precond, ac, a, sa, _ = _atk_fwd_stages_npu(
        k, beta, log_g, chunk_size,
        initial_A_state, False, cu_seqlens,
        x, eps, log_atk_scale,
    )
    return k_precond, ac, a, sa
