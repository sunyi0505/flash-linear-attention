# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused AttnRes kernels for triton-ascend on Ascend NPU.

GPU passes a padded tuple of per-source pointers (``res[i]``). Triton-Ascend
cannot bind tuple arguments, so sources are stacked in L-chunks as
``[L_chunk, N, D]`` (capped at 512 MiB per chunk) instead of one ``[L, N, D]``
buffer. Cross-chunk softmax stats are merged on the host; the weighted sum
accumulates in a single fp32 ``o_mix`` buffer to match GPU register precision.

Per-N forward/backward Triton kernels mirror the CUDA structure: when
``BD >= D`` each L tile loads ``v`` once and uses ``tl.dot`` for D reductions.
Cross-chunk softmax stats and ``o_mix`` are merged on the host.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    iter_axis_launch_chunks,
)

_FWD_MEM_MULT = 4.0
_BWD_DV_MEM_MULT = 6.0
_SAFETY_MARGIN = 0.85
_FALLBACK_BD = 256
_MAX_L_CHUNK_BYTES = 512 << 20


def _l_chunk_size(L: int, N: int, D: int, elem_size: int) -> int:
    layer_bytes = N * D * elem_size
    if layer_bytes == 0:
        return L
    return max(1, min(L, _MAX_L_CHUNK_BYTES // layer_bytes))


def _flat_residual(r: torch.Tensor, D: int) -> torch.Tensor:
    """View ``[..., D]`` as ``[N, D]`` for the autograd path."""
    return r.reshape(-1, D).contiguous()


def _stack_sources(sources: Sequence[torch.Tensor], l0: int, l1: int) -> torch.Tensor:
    if l1 - l0 == 1:
        return sources[l0].unsqueeze(0)
    return torch.stack(sources[l0:l1], dim=0)


def _iter_l_chunks(
    sources: Sequence[torch.Tensor],
) -> Iterator[tuple[int, int, torch.Tensor, int, int, int]]:
    L, N, D = len(sources), *sources[0].shape
    stride_ln, l_chunk = N * D, _l_chunk_size(L, N, D, sources[0].element_size())
    for l0 in range(0, L, l_chunk):
        l1 = min(l0 + l_chunk, L)
        yield l0, l1 - l0, _stack_sources(sources, l0, l1), stride_ln, N, D


def _merge_online_softmax_o(
    m: torch.Tensor,
    acc: torch.Tensor,
    o: torch.Tensor,
    m_chunk: torch.Tensor,
    acc_chunk: torch.Tensor,
    o_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m_new = torch.maximum(m, m_chunk)
    scale_old = torch.exp(m - m_new)
    scale_chunk = torch.exp(m_chunk - m_new)
    acc_new = acc * scale_old + acc_chunk * scale_chunk
    o_new = o * scale_old.unsqueeze(-1) + o_chunk * scale_chunk.unsqueeze(-1)
    return m_new, acc_new, o_new


def _get_bl(L: int) -> int:
    return min(8, max(1, triton.next_power_of_2(L)))


def _get_bd(row_dim: int, col_dim: int, *, memory_multiplier: float) -> int:
    return compute_row_tile_block_size(
        row_dim,
        col_dim,
        memory_multiplier,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BD,
        min_block=64,
        max_block=2048,
    )


def _get_bl_bd(ll: int, D: int, *, mem_mult: float) -> tuple[int, int, bool]:
    bl = _get_bl(ll)
    bd = _get_bd(bl, D, memory_multiplier=mem_mult)
    bd = min(bd, triton.next_power_of_2(D))
    return bl, bd, bd >= D


def _launch_n(
    kernel: Callable,
    N: int,
    /,
    **kwargs,
) -> None:
    for n_off, n_len in iter_axis_launch_chunks(N, 1, max_grid=ASCEND_MAX_GRID_DIM):
        kernel[(n_len,)](N_OFFSET=n_off, **kwargs)


@triton.jit(do_not_specialize=['L', 'N', 'D', 'L_OFFSET'])
def attnres_fwd_chunk_kernel(
    qw,
    res,
    rstd,
    logit,
    chunk_m,
    chunk_acc,
    chunk_o,
    L,
    N,
    D,
    stride_ln,
    eps,
    scale,
    BL: tl.constexpr,
    BD: tl.constexpr,
    SINGLE_D_TILE: tl.constexpr,
    L_OFFSET,
    N_OFFSET: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64) + N_OFFSET
    b_m = tl.full([], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([], dtype=tl.float32)
    if SINGLE_D_TILE:
        o_d = tl.arange(0, BD).to(tl.int64)
        m_d = o_d < D
        b_qw = tl.load(qw + o_d, mask=m_d, other=0.).to(tl.float32)
        for i_l in range(tl.cdiv(L, BL)):
            o_l = (i_l * BL + tl.arange(0, BL)).to(tl.int64)
            m_l = o_l < L
            g_l = L_OFFSET + o_l
            m_v = m_l[:, None] & m_d[None, :]
            b_v = tl.load(
                res + o_l[:, None] * stride_ln + i_n * D + o_d[None, :],
                mask=m_v,
                other=0.0,
            ).to(tl.float32)
            b_rstd = tl.rsqrt(tl.sum(b_v * b_v, axis=1) / D + eps)
            b_logit = tl.sum(b_v * b_qw[None, :], axis=1) * b_rstd
            b_s = tl.where(m_l, b_logit * scale, float('-inf'))
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, axis=0)), b_m
            b_r = exp(b_mp - b_m)
            b_p = exp(b_s - b_m)
            b_acc = b_acc * b_r + tl.sum(b_p, axis=0)
            b_o_slice = tl.sum(b_p[:, None] * b_v, axis=0)
            p_o = chunk_o + i_n * D + o_d
            b_o = tl.load(p_o, mask=m_d, other=0.).to(tl.float32) * b_r + b_o_slice
            tl.store(p_o, b_o, mask=m_d)
            tl.store(rstd + g_l * N + i_n, b_rstd.to(rstd.dtype.element_ty), mask=m_l)
            tl.store(logit + g_l * N + i_n, b_logit.to(logit.dtype.element_ty), mask=m_l)
    else:
        for i_l in range(tl.cdiv(L, BL)):
            o_l = (i_l * BL + tl.arange(0, BL)).to(tl.int64)
            m_l = o_l < L
            g_l = L_OFFSET + o_l
            b_v_sq = tl.zeros([BL], dtype=tl.float32)
            b_v_qw = tl.zeros([BL], dtype=tl.float32)
            for i_d in range(tl.cdiv(D, BD)):
                o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
                m_d = o_d < D
                b_qw = tl.load(qw + o_d, mask=m_d, other=0.).to(tl.float32)
                b_v = tl.load(
                    res + o_l[:, None] * stride_ln + i_n * D + o_d[None, :],
                    mask=m_l[:, None] & m_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                b_v_sq += tl.sum(b_v * b_v, axis=1)
                b_v_qw += tl.sum(b_v * b_qw[None, :], axis=1)
            b_rstd = tl.rsqrt(b_v_sq / D + eps)
            b_logit = b_v_qw * b_rstd
            b_s = tl.where(m_l, b_logit * scale, float('-inf'))
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, axis=0)), b_m
            b_r = exp(b_mp - b_m)
            b_p = exp(b_s - b_m)
            b_acc = b_acc * b_r + tl.sum(b_p, axis=0)
            for i_d in range(tl.cdiv(D, BD)):
                o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
                m_d = o_d < D
                b_v = tl.load(
                    res + o_l[:, None] * stride_ln + i_n * D + o_d[None, :],
                    mask=m_l[:, None] & m_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                b_o_slice = tl.sum(b_p[:, None] * b_v, axis=0)
                p_o = chunk_o + i_n * D + o_d
                b_o = tl.load(p_o, mask=m_d, other=0.).to(tl.float32) * b_r + b_o_slice
                tl.store(p_o, b_o, mask=m_d)
            tl.store(rstd + g_l * N + i_n, b_rstd.to(rstd.dtype.element_ty), mask=m_l)
            tl.store(logit + g_l * N + i_n, b_logit.to(logit.dtype.element_ty), mask=m_l)
    tl.store(chunk_m + i_n, b_m)
    tl.store(chunk_acc + i_n, b_acc)


@triton.jit(do_not_specialize=['N', 'D'])
def attnres_fwd_onorm_kernel(
    o,
    o_mix,
    ow,
    N,
    D,
    eps,
    BD: tl.constexpr,
    N_OFFSET: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64) + N_OFFSET
    b_o_sq = tl.zeros([], dtype=tl.float32)
    for i_d in range(tl.cdiv(D, BD)):
        o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
        m_d = o_d < D
        b_o = tl.load(o_mix + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
        b_o_sq += tl.sum(tl.where(m_d, b_o * b_o, 0.0), axis=0)
    b_o_rstd = tl.rsqrt(b_o_sq / D + eps)
    for i_d in range(tl.cdiv(D, BD)):
        o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
        m_d = o_d < D
        b_o = tl.load(o_mix + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
        b_ow = tl.load(ow + o_d, mask=m_d, other=0.).to(tl.float32)
        tl.store(o + i_n * D + o_d, (b_o * b_o_rstd * b_ow).to(o.dtype.element_ty), mask=m_d)


@triton.jit(do_not_specialize=['L', 'N', 'D', 'L_OFFSET'])
def attnres_bwd_dv_chunk_kernel(
    qw,
    res,
    ow,
    rstd,
    logit,
    lse,
    do,
    do_eff,
    o_mix,
    ds_scratch,
    p_scratch,
    dow_partial,
    dres,
    dqw,
    L,
    N,
    D,
    stride_ln,
    eps,
    scale,
    BL: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
    SINGLE_D_TILE: tl.constexpr,
    L_OFFSET,
    N_OFFSET: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64) + N_OFFSET
    b_lse = tl.load(lse + i_n).to(tl.float32)

    if HAS_ONORM:
        b_o_sq = tl.zeros([], dtype=tl.float32)
        for i_d in range(tl.cdiv(D, BD)):
            o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
            m_d = o_d < D
            b_o_pre = tl.load(o_mix + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_o_sq += tl.sum(tl.where(m_d, b_o_pre * b_o_pre, 0.0), axis=0)
        b_o_rstd = tl.rsqrt(b_o_sq / D + eps)
        b_c1 = tl.zeros([], dtype=tl.float32)
        for i_d in range(tl.cdiv(D, BD)):
            o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
            m_d = o_d < D
            b_do = tl.load(do + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_o_pre = tl.load(o_mix + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_ow = tl.load(ow + o_d, mask=m_d, other=0.).to(tl.float32)
            b_c1 += tl.sum(tl.where(m_d, b_o_pre * b_o_rstd * b_ow * b_do, 0.0), axis=0)
        b_c1 /= D
        b_delta_val = tl.zeros([], dtype=tl.float32)
        for i_d in range(tl.cdiv(D, BD)):
            o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
            m_d = o_d < D
            b_do = tl.load(do + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_o_pre = tl.load(o_mix + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_ow = tl.load(ow + o_d, mask=m_d, other=0.).to(tl.float32)
            b_xhat = b_o_pre * b_o_rstd
            tl.store(dow_partial + i_n * D + o_d, (b_xhat * b_do).to(dow_partial.dtype.element_ty), mask=m_d)
            b_do = (b_ow * b_do - b_xhat * b_c1) * b_o_rstd
            tl.store(do_eff + i_n * D + o_d, b_do, mask=m_d)
            b_delta_val += tl.sum(tl.where(m_d, b_do * b_o_pre, 0.0), axis=0)
    else:
        b_delta_val = tl.zeros([], dtype=tl.float32)
        for i_d in range(tl.cdiv(D, BD)):
            o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
            m_d = o_d < D
            b_do = tl.load(do_eff + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_o_pre = tl.load(o_mix + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_delta_val += tl.sum(tl.where(m_d, b_do * b_o_pre, 0.0), axis=0)

    if SINGLE_D_TILE:
        o_d = tl.arange(0, BD).to(tl.int64)
        m_d = o_d < D
        b_qw = tl.load(qw + o_d, mask=m_d, other=0.).to(tl.float32)
        b_do = tl.load(do_eff + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
        b_dqw = tl.zeros([BD], dtype=tl.float32)
        for i_l in range(tl.cdiv(L, BL)):
            o_l = (i_l * BL + tl.arange(0, BL)).to(tl.int64)
            m_l = o_l < L
            g_l = L_OFFSET + o_l
            m_v = m_l[:, None] & m_d[None, :]
            b_v = tl.load(
                res + o_l[:, None] * stride_ln + i_n * D + o_d[None, :],
                mask=m_v,
                other=0.0,
            ).to(tl.float32)
            b_rstd = tl.load(rstd + g_l * N + i_n, mask=m_l, other=0.).to(tl.float32)
            b_logit = tl.load(logit + g_l * N + i_n, mask=m_l, other=0.).to(tl.float32)
            b_p = tl.where(m_l, exp(b_logit * scale - b_lse), 0.0)
            b_dp = tl.sum(b_v * b_do[None, :], axis=1)
            b_ds = b_p * (b_dp - b_delta_val) * scale
            b_k = b_v * b_rstd[:, None]
            b_dv = b_p[:, None] * b_do[None, :] + (b_ds * b_rstd)[:, None] * (
                b_qw[None, :] - b_k * (b_logit / D)[:, None]
            )
            tl.store(
                dres + o_l[:, None] * stride_ln + i_n * D + o_d[None, :],
                b_dv.to(dres.dtype.element_ty),
                mask=m_v,
            )
            b_dqw += tl.sum(b_ds[:, None] * b_k, axis=0)
        p_dqw = dqw + i_n * D + o_d
        tl.store(
            p_dqw,
            tl.load(p_dqw, mask=m_d, other=0.).to(tl.float32) + b_dqw,
            mask=m_d,
        )
        return

    for i_l in range(tl.cdiv(L, BL)):
        o_l = (i_l * BL + tl.arange(0, BL)).to(tl.int64)
        m_l = o_l < L
        g_l = L_OFFSET + o_l
        b_rstd = tl.load(rstd + g_l * N + i_n, mask=m_l, other=0.).to(tl.float32)
        b_logit = tl.load(logit + g_l * N + i_n, mask=m_l, other=0.).to(tl.float32)
        b_p = tl.where(m_l, exp(b_logit * scale - b_lse), 0.0)
        b_dp = tl.zeros([BL], dtype=tl.float32)
        for i_d in range(tl.cdiv(D, BD)):
            o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
            m_d = o_d < D
            b_do = tl.load(do_eff + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_v = tl.load(
                res + o_l[:, None] * stride_ln + i_n * D + o_d[None, :],
                mask=m_l[:, None] & m_d[None, :],
                other=0.0,
            ).to(tl.float32)
            b_dp += tl.sum(b_v * b_do[None, :], axis=1)
        b_ds = b_p * (b_dp - b_delta_val) * scale
        tl.store(ds_scratch + o_l * N + i_n, b_ds, mask=m_l)
        tl.store(p_scratch + o_l * N + i_n, b_p, mask=m_l)

    for i_d in range(tl.cdiv(D, BD)):
        o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
        m_d = o_d < D
        b_qw = tl.load(qw + o_d, mask=m_d, other=0.).to(tl.float32)
        b_do = tl.load(do_eff + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
        b_dqw = tl.zeros([BD], dtype=tl.float32)
        for i_l in range(tl.cdiv(L, BL)):
            o_l = (i_l * BL + tl.arange(0, BL)).to(tl.int64)
            m_l = o_l < L
            m_v = m_l[:, None] & m_d[None, :]
            b_p = tl.load(p_scratch + o_l * N + i_n, mask=m_l, other=0.).to(tl.float32)
            b_ds = tl.load(ds_scratch + o_l * N + i_n, mask=m_l, other=0.).to(tl.float32)
            b_v = tl.load(
                res + o_l[:, None] * stride_ln + i_n * D + o_d[None, :],
                mask=m_v,
                other=0.0,
            ).to(tl.float32)
            b_rstd = tl.load(rstd + (L_OFFSET + o_l) * N + i_n, mask=m_l, other=0.).to(tl.float32)
            b_logit = tl.load(logit + (L_OFFSET + o_l) * N + i_n, mask=m_l, other=0.).to(tl.float32)
            b_k = b_v * b_rstd[:, None]
            b_dv = b_p[:, None] * b_do[None, :] + (b_ds * b_rstd)[:, None] * (
                b_qw[None, :] - b_k * (b_logit / D)[:, None]
            )
            tl.store(
                dres + o_l[:, None] * stride_ln + i_n * D + o_d[None, :],
                b_dv.to(dres.dtype.element_ty),
                mask=m_v,
            )
            b_dqw += tl.sum(b_ds[:, None] * b_k, axis=0)
        p_dqw = dqw + i_n * D + o_d
        tl.store(
            p_dqw,
            tl.load(p_dqw, mask=m_d, other=0.).to(tl.float32) + b_dqw,
            mask=m_d,
        )


@triton.jit(do_not_specialize=['N', 'D'])
def attnres_bwd_kernel_dqdw_npu(
    q,
    w,
    dqw,
    dow_partial,
    dq,
    dw,
    dow,
    N,
    D,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
):
    i_d = tl.program_id(0)
    o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
    m_d = o_d < D
    b_dqw = tl.zeros([BD], dtype=tl.float32)
    b_dow = tl.zeros([BD], dtype=tl.float32)
    for i_n in range(N):
        b_dqw += tl.load(dqw + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
        if HAS_ONORM:
            b_dow += tl.load(dow_partial + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
    b_q = tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32)
    b_w = tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)
    tl.store(dq + o_d, b_dqw * b_w, mask=m_d)
    tl.store(dw + o_d, b_dqw * b_q, mask=m_d)
    if HAS_ONORM:
        tl.store(dow + o_d, b_dow, mask=m_d)


def _resolve_o_mix(
    o_pre: torch.Tensor | None,
    o_mix_saved: torch.Tensor,
) -> torch.Tensor:
    if o_mix_saved.numel() > 0:
        return o_mix_saved
    if o_pre is not None:
        return o_pre.float()
    raise RuntimeError('attnres bwd requires either o_pre or saved o_mix')


def _dres_chunk(
    flat_dvs: list[torch.Tensor | None],
    sources: Sequence[torch.Tensor],
    l0: int,
    ll: int,
    chunk: torch.Tensor,
) -> torch.Tensor:
    if ll == 1:
        if flat_dvs[l0] is None:
            flat_dvs[l0] = torch.empty_like(sources[l0])
        return flat_dvs[l0].unsqueeze(0)
    return torch.empty_like(chunk)


def fused_attnres_fwd_npu(
    q: torch.Tensor,
    sources: Sequence[torch.Tensor],
    w: torch.Tensor,
    ow: torch.Tensor | None,
    eps: float,
    scale: float,
    checkpoint_level: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    L, N, D = len(sources), *sources[0].shape
    dtype = sources[0].dtype
    save_opre = checkpoint_level == 0
    save_o_mix = checkpoint_level == 1

    o = torch.empty((N, D), device=sources[0].device, dtype=dtype)
    o_pre = torch.empty((N, D), device=sources[0].device, dtype=dtype) if save_opre else None
    o_mix = torch.zeros((N, D), device=sources[0].device, dtype=torch.float32)
    lse = torch.empty(N, device=sources[0].device, dtype=torch.float32)
    rstd = torch.empty((L, N), device=sources[0].device, dtype=torch.float32)
    logit = torch.empty_like(rstd)
    qw = q.float() * w.float()

    m = torch.full((N,), float('-inf'), device=sources[0].device, dtype=torch.float32)
    acc = torch.zeros(N, device=sources[0].device, dtype=torch.float32)
    for l0, ll, chunk, stride_ln, N, D in _iter_l_chunks(sources):
        chunk_m = torch.empty(N, device=chunk.device, dtype=torch.float32)
        chunk_acc = torch.empty(N, device=chunk.device, dtype=torch.float32)
        chunk_o = torch.zeros(N, D, device=chunk.device, dtype=torch.float32)
        bl, bd, single_d = _get_bl_bd(ll, D, mem_mult=_FWD_MEM_MULT)
        _launch_n(
            attnres_fwd_chunk_kernel,
            N,
            qw=qw,
            res=chunk,
            rstd=rstd,
            logit=logit,
            chunk_m=chunk_m,
            chunk_acc=chunk_acc,
            chunk_o=chunk_o,
            L=ll,
            N=N,
            D=D,
            stride_ln=stride_ln,
            eps=eps,
            scale=scale,
            BL=bl,
            BD=bd,
            SINGLE_D_TILE=single_d,
            L_OFFSET=l0,
        )
        m, acc, o_mix = _merge_online_softmax_o(m, acc, o_mix, chunk_m, chunk_acc, chunk_o)
        del chunk, chunk_m, chunk_acc, chunk_o

    lse.copy_(m + torch.log(acc))
    o_mix.div_(acc.unsqueeze(-1))
    o_mix_saved = o_mix if save_o_mix else torch.empty(0, device=o.device, dtype=torch.float32)

    if save_opre:
        o_pre.copy_(o_mix.to(dtype))
        if ow is None:
            o.copy_(o_pre)
    elif ow is None:
        o.copy_(o_mix.to(dtype))
    if ow is not None:
        _launch_n(
            attnres_fwd_onorm_kernel,
            N,
            o=o,
            o_mix=o_mix,
            ow=ow,
            N=N,
            D=D,
            eps=eps,
            BD=_get_bd(1, D, memory_multiplier=2.0),
        )
    return o, o_pre, o_mix_saved, rstd, logit, lse


def fused_attnres_bwd_npu(
    do: torch.Tensor,
    q: torch.Tensor,
    sources: Sequence[torch.Tensor],
    w: torch.Tensor,
    ow: torch.Tensor | None,
    o_pre: torch.Tensor | None,
    o_mix_saved: torch.Tensor,
    rstd: torch.Tensor,
    logit: torch.Tensor,
    lse: torch.Tensor,
    eps: float,
    scale: float,
    checkpoint_level: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
    del checkpoint_level
    has_onorm = ow is not None
    N, D = do.shape
    flat_dvs: list[torch.Tensor | None] = [None] * len(sources)

    do_eff = torch.empty_like(do, dtype=torch.float32)
    dqw = torch.zeros_like(do, dtype=torch.float32)
    dq, dw = torch.empty_like(q), torch.empty_like(w)
    dow = torch.empty_like(ow) if has_onorm else None
    dow_partial = torch.empty_like(do, dtype=torch.float32) if has_onorm else do_eff
    scratch_stub = torch.empty(1, device=do.device, dtype=torch.float32)
    qw = q.float() * w.float()

    o_mix = _resolve_o_mix(o_pre, o_mix_saved)
    if not has_onorm:
        do_eff.copy_(do)

    for l0, ll, chunk, stride_ln, N, D in _iter_l_chunks(sources):
        dres = _dres_chunk(flat_dvs, sources, l0, ll, chunk)
        bl, bd, single_d = _get_bl_bd(ll, D, mem_mult=_BWD_DV_MEM_MULT)
        if single_d:
            ds_scratch = p_scratch = scratch_stub
        else:
            ds_scratch = torch.empty(ll, N, device=chunk.device, dtype=torch.float32)
            p_scratch = torch.empty(ll, N, device=chunk.device, dtype=torch.float32)
        _launch_n(
            attnres_bwd_dv_chunk_kernel,
            N,
            qw=qw,
            res=chunk,
            ow=ow if ow is not None else qw,
            rstd=rstd,
            logit=logit,
            lse=lse,
            do=do,
            do_eff=do_eff,
            o_mix=o_mix,
            ds_scratch=ds_scratch,
            p_scratch=p_scratch,
            dow_partial=dow_partial,
            dres=dres,
            dqw=dqw,
            L=ll,
            N=N,
            D=D,
            stride_ln=stride_ln,
            eps=eps,
            scale=scale,
            BL=bl,
            BD=bd,
            HAS_ONORM=has_onorm,
            SINGLE_D_TILE=single_d,
            L_OFFSET=l0,
        )
        if not single_d:
            del ds_scratch, p_scratch
        if ll > 1:
            for i in range(ll):
                idx = l0 + i
                if flat_dvs[idx] is None:
                    flat_dvs[idx] = torch.empty_like(sources[idx])
                flat_dvs[idx].copy_(dres[i])
            del dres
        del chunk

    bd = _get_bd(1, D, memory_multiplier=2.0)
    attnres_bwd_kernel_dqdw_npu[(triton.cdiv(D, bd),)](
        q=q,
        w=w,
        dqw=dqw,
        dow_partial=dow_partial,
        dq=dq,
        dw=dw,
        dow=dow if dow is not None else dq,
        N=N,
        D=D,
        BD=bd,
        HAS_ONORM=has_onorm,
    )
    assert all(t is not None for t in flat_dvs)
    return dq, dw, dow, flat_dvs  # type: ignore[return-value]


class FusedAttnresNpuFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        query: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor | None,
        rms_eps: float,
        scale: float,
        return_weights: bool,
        checkpoint_level: int,
        *residuals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        o, o_pre, o_mix_saved, rstd, logit, lse = fused_attnres_fwd_npu(
            query, residuals, rms_weight, output_rms_weight,
            rms_eps, scale, checkpoint_level,
        )
        ctx.save_for_backward(
            query, rms_weight, output_rms_weight, o_pre, o_mix_saved, rstd, logit, lse, *residuals,
        )
        ctx.eps, ctx.scale, ctx.checkpoint_level = rms_eps, scale, checkpoint_level
        p = (logit * scale - lse).exp() if return_weights else o.new_empty(0)
        ctx.mark_non_differentiable(p)
        return o, p

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dp: torch.Tensor | None = None):
        del dp
        query, rms_weight, output_rms_weight, o_pre, o_mix_saved, rstd, logit, lse, *residuals = ctx.saved_tensors
        dq, dw, dow, flat_dvs = fused_attnres_bwd_npu(
            do, query, residuals, rms_weight, output_rms_weight,
            o_pre, o_mix_saved, rstd, logit, lse, ctx.eps, ctx.scale, ctx.checkpoint_level,
        )
        return (dq, dw, dow, None, None, None, None, *flat_dvs)


def fused_attnres_npu(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
    checkpoint_level: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    output_shape = residuals[0].shape
    D = output_shape[-1]
    flat = tuple(_flat_residual(r, D) for r in residuals)
    o, p = FusedAttnresNpuFunction.apply(
        query, rms_weight, output_rms_weight, rms_eps, scale,
        return_weights, checkpoint_level, *flat,
    )
    o = o.view(output_shape)
    if return_weights:
        return o, p.view(len(residuals), *output_shape[:-1])
    return o


__all__ = [
    'FusedAttnresNpuFunction',
    'fused_attnres_bwd_npu',
    'fused_attnres_fwd_npu',
    'fused_attnres_npu',
]
