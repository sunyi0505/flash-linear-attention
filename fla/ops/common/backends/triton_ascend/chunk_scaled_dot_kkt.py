# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""chunk_scaled_dot_kkt_fwd adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    max_grid_axis_chunks,
)

_NUM_WARPS = 4
_BC = 16
# One [BC,BC] fp32 tile + two [BC,BK] operand tiles.
_KKT_MEM_MULT = 4.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BK = 8
_MAX_BK = 64


def _hv_t_npu_arg(x: torch.Tensor | None, HV: int) -> tuple[torch.Tensor | None, bool]:
    """Transpose to [B, HV, T] when HV>1 for contiguous token-axis loads."""
    if x is None or HV == 1:
        return x, False
    return x.transpose(1, 2).contiguous(), True


def _get_bk(BC: int, K: int) -> int:
    return compute_row_tile_block_size(
        BC,
        K,
        _KKT_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK,
        min_block=8,
        max_block=min(_MAX_BK, triton.next_power_of_2(K)),
    )


def _get_bc(BT: int) -> int:
    # Avoid 16x16 sub-tiling: column-offset block_ptr stores misalign on NPU.
    return BT if BT <= 64 else _BC


@triton.jit
def _t_npu_base(
    tensor,
    bos,
    i_b,
    i_h,
    T_seq,
    T_CONTIG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HV: tl.constexpr,
):
    if T_CONTIG:
        if IS_VARLEN:
            return tensor + bos + i_h * T_seq
        return tensor + i_b * HV * T_seq + i_h * T_seq
    return tensor + bos * HV + i_h


@triton.jit
def _t_block_ptr(base, T, offset, BC, T_CONTIG: tl.constexpr, HV: tl.constexpr):
    if T_CONTIG:
        return tl.make_block_ptr(base, (T,), (1,), (offset,), (BC,), (0,))
    return tl.make_block_ptr(base, (T,), (HV,), (offset,), (BC,), (0,))


def _launch_kkt_kernel(kernel, *, NT: int, bh_total: int, kernel_kwargs: dict) -> None:
    max_nt = max_grid_axis_chunks(NT, bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    chunk_indices = kernel_kwargs.get('chunk_indices')
    cu_seqlens = kernel_kwargs.get('cu_seqlens')
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        if cu_seqlens is not None and chunk_indices is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['NT_OFFSET'] = nt_off
        max_bh = max_grid_axis_chunks(bh_total, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            kernel[(nt_len, bh_len)](num_warps=_NUM_WARPS, **kernel_kwargs)


@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel_npu(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    T_seq,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    BETA_T_CONTIG: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T

    if i_t * BT >= T:
        return

    k_base = k + (bos * H + i_h // (HV // H)) * K
    A_base = A + (bos * HV + i_h) * BT
    beta_base = _t_npu_base(beta, bos, i_b, i_h, T_seq, BETA_T_CONTIG, IS_VARLEN, HV)
    if USE_G:
        g_base = _t_npu_base(g, bos, i_b, i_h, T_seq, G_T_CONTIG, IS_VARLEN, HV)

    o_i = tl.arange(0, BC)
    n_sub = BT // BC

    for s in range(n_sub):
        i_tc_s = i_t * BT + s * BC
        m_s = (i_tc_s + o_i) < T
        p_bs = _t_block_ptr(beta_base, T, i_tc_s, BC, BETA_T_CONTIG, HV)
        b_bs = tl.load(p_bs, boundary_check=(0,))
        if USE_G:
            p_gs = _t_block_ptr(g_base, T, i_tc_s, BC, G_T_CONTIG, HV)
            b_gs = tl.load(p_gs, boundary_check=(0,))

        for c in range(s + 1):
            i_tc_c = i_t * BT + c * BC
            m_c = (i_tc_c + o_i) < T
            b_A = tl.zeros([BC, BC], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_ks = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
                p_kc = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
                b_ks = tl.load(p_ks, boundary_check=(0, 1))
                b_kc = tl.load(p_kc, boundary_check=(0, 1))
                b_A += tl.dot(b_ks, tl.trans(b_kc), allow_tf32=False)

            if USE_G:
                p_gc = _t_block_ptr(g_base, T, i_tc_c, BC, G_T_CONTIG, HV)
                b_gc = tl.load(p_gc, boundary_check=(0,))
                b_A *= exp2(b_gs[:, None] - b_gc[None, :])
            b_A *= b_bs[:, None]

            if s == c:
                m_blk = (o_i[:, None] > o_i[None, :]) & (m_s[:, None] & m_s[None, :])
            else:
                m_blk = m_s[:, None] & m_c[None, :]
            b_A = tl.where(m_blk, b_A, 0)

            p_A = tl.make_block_ptr(A_base, (T, BT), (HV * BT, 1), (i_tc_s, c * BC), (BC, BC), (1, 0))
            tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


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
    B, T, H, K, HV = *k.shape, beta.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, HV, BT, device=k.device, dtype=output_dtype)
    BC = _get_bc(BT)
    BK = _get_bk(BC, K)
    use_g = g is not None
    g_arg, g_t_contig = _hv_t_npu_arg(g, HV)
    beta_arg, beta_t_contig = _hv_t_npu_arg(beta, HV)
    _launch_kkt_kernel(
        chunk_scaled_dot_kkt_fwd_kernel_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs={
            'k': k,
            'g': g_arg if use_g else beta_arg,
            'beta': beta_arg,
            'A': A,
            'cu_seqlens': cu_seqlens,
            'chunk_indices': chunk_indices,
            'T': T,
            'T_seq': T,
            'H': H,
            'HV': HV,
            'K': K,
            'BT': BT,
            'BC': BC,
            'BK': BK,
            'USE_G': use_g,
            'IS_VARLEN': cu_seqlens is not None,
            'G_T_CONTIG': g_t_contig,
            'BETA_T_CONTIG': beta_t_contig,
            'NT_OFFSET': 0,
            'BH_OFFSET': 0,
        },
    )
    return A
