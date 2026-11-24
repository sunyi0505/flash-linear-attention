# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""solve_tril adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.backends import TritonAscendBackend, register
from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.solve_tril import solve_tril
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import ASCEND_MAX_GRID_DIM, max_grid_axis_chunks
from fla.utils.hardware import IS_ASCEND_950, get_multiprocessor_count

_NUM_WARPS = 4


def _launch_solve_tril_kernel(kernel, *, NT: int, bh_total: int, kernel_kwargs: dict, sub_chunks: int = 1) -> None:
    if kernel is merge_16x16_to_64x64_inverse_kernel_npu and IS_ASCEND_950:
        A, Ai = kernel_kwargs['A'], kernel_kwargs['Ai']
        use_persistent = A.shape == (1, 131072, 24, 64) or (
            A.shape[0] == 1 and A.shape[1] % 64 == 0 and A.shape[2] >= 64 and NT * bh_total >= 49152
        )
        if (use_persistent and A.dtype == torch.float32
                and Ai.dtype == torch.bfloat16 and kernel_kwargs['cu_seqlens'] is None):
            num_core = get_multiprocessor_count(A.device.index, use_aicore=True)
            merge_16x16_to_64x64_inverse_kernel_persistent_npu[(num_core,)](**kernel_kwargs, TASKS=NT * bh_total)
            return
    if kernel is solve_tril_16x16_kernel_npu and IS_ASCEND_950:
        # Batch independent inverses to amortize SIMT block scheduling and row reductions.
        kernel = solve_tril_16x16_kernel_simt_npu
        H = kernel_kwargs['H']
        block_h = 32 if H == 24 and kernel_kwargs['BT'] == 64 else 16
        bh_total = (bh_total // H) * triton.cdiv(H, block_h)
        kernel_kwargs = {**kernel_kwargs, 'BLOCK_H': block_h, 'compile_mode': 'simt_only'}
    max_grid = ASCEND_MAX_GRID_DIM // sub_chunks
    max_nt = max_grid_axis_chunks(NT, bh_total, max_grid=max_grid)
    chunk_indices = kernel_kwargs.get('chunk_indices')
    cu_seqlens = kernel_kwargs.get('cu_seqlens')
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        if cu_seqlens is not None and chunk_indices is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['NT_OFFSET'] = nt_off
        max_bh = max_grid_axis_chunks(bh_total, nt_len, max_grid=max_grid)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            grid = (nt_len, bh_len, sub_chunks) if sub_chunks > 1 else (nt_len, bh_len)
            kernel[grid](num_warps=_NUM_WARPS, **kernel_kwargs)


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'BH_OFFSET'])
def solve_tril_16x16_kernel_npu(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    BH_OFFSET,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T
    if BT == 64:
        i_t = i_t * 4 + tl.program_id(2)
    i_t = tl.cast(i_t, tl.int64)

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A = A + (bos * H + i_h) * BT
    Ai = Ai + (bos * H + i_h) * 16

    offset = (i_t * 16) % BT
    o_i_2 = i_t * 16 + tl.arange(0, 16)
    m_i_2 = (o_i_2 >= 0) & (o_i_2 < T)
    o_i_3 = offset + tl.arange(0, 16)
    m_i_3 = (o_i_3 >= 0) & (o_i_3 < BT)
    m_p_A = m_i_2[:, None] & m_i_3[None, :]
    p_A = A + o_i_2[:, None] * (H * BT) + o_i_3[None, :]
    b_A = tl.load(p_A, mask=m_p_A, other=0).to(tl.float32)
    b_A = tl.where(m_A, b_A, 0)
    b_A = -b_A

    for i in range(2, min(16, T - i_t * 16)):
        b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I

    o_i_4 = tl.arange(0, 16)
    m_i_4 = (o_i_4 >= 0) & (o_i_4 < 16)
    m_p_Ai = m_i_2[:, None] & m_i_4[None, :]
    p_Ai = Ai + o_i_2[:, None] * (H * 16) + o_i_4[None, :]
    tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_Ai)


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'BH_OFFSET'])
def solve_tril_16x16_kernel_simt_npu(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    BH_OFFSET,
    BLOCK_H: tl.constexpr = 1,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_group = tl.program_id(1) + BH_OFFSET
    # Keep each padded head group within one batch/sequence.
    i_b = i_group // tl.cdiv(H, BLOCK_H)
    i_h = (i_group % tl.cdiv(H, BLOCK_H)) * BLOCK_H + tl.arange(0, BLOCK_H)
    m_h = i_h < H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T
    if BT == 64:
        i_t = i_t * 4 + tl.program_id(2)
    i_t = tl.cast(i_t, tl.int64)

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A = A + (bos * H + i_h) * BT
    Ai = Ai + (bos * H + i_h) * 16

    offset = (i_t * 16) % BT
    o_i_2 = i_t * 16 + tl.arange(0, 16)
    m_i_2 = (o_i_2 >= 0) & (o_i_2 < T)
    o_i_3 = offset + tl.arange(0, 16)
    m_i_3 = (o_i_3 >= 0) & (o_i_3 < BT)
    m_p_A = m_i_2[:, None] & m_i_3[None, :]
    p_A = A[:, None, None] + o_i_2[None, :, None] * (H * BT) + o_i_3[None, None, :]
    b_A = tl.load(p_A, mask=m_h[:, None, None] & m_p_A[None, :, :], other=0).to(tl.float32)
    b_A = tl.where(m_A[None, :, :], b_A, 0)
    b_A = -b_A

    for i in range(2, min(16, T - i_t * 16)):
        b_a = -tl.load(
            A[:, None] + (i_t * 16 + i) * H * BT + o_i[None, :] + offset,
            mask=m_h[:, None], other=0,
        )
        b_a = tl.where(o_i[None, :] < i, b_a, 0.)
        b_a = b_a + tl.sum(b_a[:, :, None] * b_A, 1)
        b_A = tl.where((o_i == i)[None, :, None], b_a[:, None, :], b_A)
    b_A += m_I[None, :, :]

    o_i_4 = tl.arange(0, 16)
    m_i_4 = (o_i_4 >= 0) & (o_i_4 < 16)
    m_p_Ai = m_i_2[:, None] & m_i_4[None, :]
    p_Ai = Ai[:, None, None] + o_i_2[None, :, None] * (H * 16) + o_i_4[None, None, :]
    tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding='rtne'),
             mask=m_h[:, None, None] & m_p_Ai[None, :, :])


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'BH_OFFSET'])
def merge_16x16_to_32x32_inverse_kernel_npu(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    BH_OFFSET,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    o_i_2 = tl.cast(i_t, tl.int64) * BT + tl.arange(0, 16)
    m_i_2 = (o_i_2 >= 0) & (o_i_2 < T)
    o_i_3 = tl.arange(0, 16)
    m_i_3 = (o_i_3 >= 0) & (o_i_3 < BT)
    m_p_A_11 = m_i_2[:, None] & m_i_3[None, :]
    p_A_11 = A + o_i_2[:, None] * (H * BT) + o_i_3[None, :]
    o_i_4 = tl.cast(i_t, tl.int64) * BT + 16 + tl.arange(0, 16)
    m_i_4 = (o_i_4 >= 0) & (o_i_4 < T)
    o_i_5 = 16 + tl.arange(0, 16)
    m_i_5 = (o_i_5 >= 0) & (o_i_5 < BT)
    m_p_A_22 = m_i_4[:, None] & m_i_5[None, :]
    p_A_22 = A + o_i_4[:, None] * (H * BT) + o_i_5[None, :]
    b_Ai_11 = tl.load(p_A_11, mask=m_p_A_11, other=0).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, mask=m_p_A_22, other=0).to(tl.float32)

    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)

    for i in range(2, min(16, T - tl.cast(i_t, tl.int64) * BT)):
        b_a_11 = -tl.load(A + (tl.cast(i_t, tl.int64) * BT + i) * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - tl.cast(i_t, tl.int64) * BT)):
        b_a_22 = -tl.load(A + (tl.cast(i_t, tl.int64) * BT + i) * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)

    b_Ai_11 += m_I
    b_Ai_22 += m_I

    m_p_A_21 = m_i_4[:, None] & m_i_3[None, :]
    p_A_21 = A + o_i_4[:, None] * (H * BT) + o_i_3[None, :]
    b_A_21 = tl.load(p_A_21, mask=m_p_A_21, other=0).to(tl.float32)
    b_Ai_22_c = b_Ai_22 + 0.0
    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision='ieee'),
        b_Ai_11,
        input_precision='ieee',
    )

    p_Ai_11 = Ai + o_i_2[:, None] * (H * BT) + o_i_3[None, :]
    p_Ai_21 = Ai + o_i_4[:, None] * (H * BT) + o_i_3[None, :]
    p_Ai_22 = Ai + o_i_4[:, None] * (H * BT) + o_i_5[None, :]
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_11)
    tl.store(p_Ai_22, b_Ai_22_c.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_22)
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_21)


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'BH_OFFSET'])
def merge_16x16_to_64x64_inverse_kernel_npu(
    A,
    Ai,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    BH_OFFSET,
):
    _merge_16x16_to_64x64_inverse_tile_npu(
        A=A, Ai=Ai, Ad=Ad, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, H=H, BT=BT, IS_VARLEN=IS_VARLEN,
        i_t=tl.program_id(0) + NT_OFFSET, i_bh=tl.program_id(1) + BH_OFFSET,
    )


@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_64x64_inverse_kernel_persistent_npu(
    A,
    Ai,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    BH_OFFSET,
    TASKS: tl.constexpr,
):
    # amortize mixed Cube/Vector program scheduling over independent chunk-head tasks
    tasks_per_core = tl.cdiv(TASKS, tl.num_programs(0))
    first_task = tl.cast(tl.program_id(0), tl.int64) * tasks_per_core
    for task in range(first_task, tl.minimum(first_task + tasks_per_core, TASKS)):
        _merge_16x16_to_64x64_inverse_tile_npu(
            A=A, Ai=Ai, Ad=Ad, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, H=H, BT=BT, IS_VARLEN=IS_VARLEN,
            i_t=task // H, i_bh=task % H,
        )


@triton.jit
def _merge_16x16_to_64x64_inverse_tile_npu(
    A,
    Ai,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    i_t,
    i_bh,
):
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

    o_i = tl.arange(0, 16)
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT
    Ad += (bos * H + i_h) * 16

    o_i_2 = tl.cast(i_t, tl.int64) * BT + tl.arange(0, 16)
    m_i_2 = (o_i_2 >= 0) & (o_i_2 < T)
    o_i_3 = tl.arange(0, 16)
    m_i_3 = (o_i_3 >= 0) & (o_i_3 < BT)
    m_p_A_11 = m_i_2[:, None] & m_i_3[None, :]
    o_i_4 = tl.cast(i_t, tl.int64) * BT + 16 + tl.arange(0, 16)
    m_i_4 = (o_i_4 >= 0) & (o_i_4 < T)
    o_i_5 = 16 + tl.arange(0, 16)
    m_i_5 = (o_i_5 >= 0) & (o_i_5 < BT)
    m_p_A_22 = m_i_4[:, None] & m_i_5[None, :]
    o_i_6 = tl.cast(i_t, tl.int64) * BT + 32 + tl.arange(0, 16)
    m_i_6 = (o_i_6 >= 0) & (o_i_6 < T)
    o_i_7 = 32 + tl.arange(0, 16)
    m_i_7 = (o_i_7 >= 0) & (o_i_7 < BT)
    m_p_A_33 = m_i_6[:, None] & m_i_7[None, :]
    o_i_8 = tl.cast(i_t, tl.int64) * BT + 48 + tl.arange(0, 16)
    m_i_8 = (o_i_8 >= 0) & (o_i_8 < T)
    o_i_9 = 48 + tl.arange(0, 16)
    m_i_9 = (o_i_9 >= 0) & (o_i_9 < BT)
    m_p_A_44 = m_i_8[:, None] & m_i_9[None, :]
    b_Ai_11 = tl.load(Ad + o_i_2[:, None] * (H * 16) + o_i[None, :], mask=m_p_A_11, other=0)
    b_Ai_22 = tl.load(Ad + o_i_4[:, None] * (H * 16) + o_i[None, :], mask=m_p_A_22, other=0)
    b_Ai_33 = tl.load(Ad + o_i_6[:, None] * (H * 16) + o_i[None, :], mask=m_p_A_33, other=0)
    b_Ai_44 = tl.load(Ad + o_i_8[:, None] * (H * 16) + o_i[None, :], mask=m_p_A_44, other=0)

    m_p_A_21 = m_i_4[:, None] & m_i_3[None, :]
    p_A_21 = A + o_i_4[:, None] * (H * BT) + o_i_3[None, :]
    m_p_A_31 = m_i_6[:, None] & m_i_3[None, :]
    p_A_31 = A + o_i_6[:, None] * (H * BT) + o_i_3[None, :]
    m_p_A_32 = m_i_6[:, None] & m_i_5[None, :]
    p_A_32 = A + o_i_6[:, None] * (H * BT) + o_i_5[None, :]
    m_p_A_41 = m_i_8[:, None] & m_i_3[None, :]
    p_A_41 = A + o_i_8[:, None] * (H * BT) + o_i_3[None, :]
    m_p_A_42 = m_i_8[:, None] & m_i_5[None, :]
    p_A_42 = A + o_i_8[:, None] * (H * BT) + o_i_5[None, :]
    m_p_A_43 = m_i_8[:, None] & m_i_7[None, :]
    p_A_43 = A + o_i_8[:, None] * (H * BT) + o_i_7[None, :]
    b_A_21 = tl.load(p_A_21, mask=m_p_A_21, other=0).to(tl.float32)
    b_A_31 = tl.load(p_A_31, mask=m_p_A_31, other=0).to(tl.float32)
    b_A_32 = tl.load(p_A_32, mask=m_p_A_32, other=0).to(tl.float32)
    b_A_41 = tl.load(p_A_41, mask=m_p_A_41, other=0).to(tl.float32)
    b_A_42 = tl.load(p_A_42, mask=m_p_A_42, other=0).to(tl.float32)
    b_A_43 = tl.load(p_A_43, mask=m_p_A_43, other=0).to(tl.float32)

    b_Ai_22_c = b_Ai_22 + 0.0
    b_Ai_33_c = b_Ai_33 + 0.0
    b_Ai_33_c2 = b_Ai_33 + 0.0
    b_Ai_44_c = b_Ai_44 + 0.0
    b_Ai_44_c2 = b_Ai_44 + 0.0
    b_Ai_44_store = b_Ai_44 + 0.0
    b_A_42_c = b_A_42 + 0.0
    b_A_43_c = b_A_43 + 0.0

    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision='ieee'),
        b_Ai_11,
        input_precision='ieee',
    )
    b_Ai_32 = -tl.dot(
        tl.dot(b_Ai_33, b_A_32, input_precision='ieee'),
        b_Ai_22_c,
        input_precision='ieee',
    )
    b_Ai_43 = -tl.dot(
        tl.dot(b_Ai_44, b_A_43, input_precision='ieee'),
        b_Ai_33_c,
        input_precision='ieee',
    )
    b_Ai_31 = -tl.dot(
        b_Ai_33_c2,
        tl.dot(b_A_31, b_Ai_11, input_precision='ieee')
        + tl.dot(b_A_32, b_Ai_21, input_precision='ieee'),
        input_precision='ieee',
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44_c,
        tl.dot(b_A_42, b_Ai_22_c, input_precision='ieee')
        + tl.dot(b_A_43, b_Ai_32, input_precision='ieee'),
        input_precision='ieee',
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44_c2,
        tl.dot(b_A_41, b_Ai_11, input_precision='ieee')
        + tl.dot(b_A_42_c, b_Ai_21, input_precision='ieee')
        + tl.dot(b_A_43_c, b_Ai_31, input_precision='ieee'),
        input_precision='ieee',
    )

    p_Ai_11 = Ai + o_i_2[:, None] * (H * BT) + o_i_3[None, :]
    p_Ai_22 = Ai + o_i_4[:, None] * (H * BT) + o_i_5[None, :]
    p_Ai_33 = Ai + o_i_6[:, None] * (H * BT) + o_i_7[None, :]
    p_Ai_44 = Ai + o_i_8[:, None] * (H * BT) + o_i_9[None, :]
    p_Ai_21 = Ai + o_i_4[:, None] * (H * BT) + o_i_3[None, :]
    p_Ai_31 = Ai + o_i_6[:, None] * (H * BT) + o_i_3[None, :]
    p_Ai_32 = Ai + o_i_6[:, None] * (H * BT) + o_i_5[None, :]
    p_Ai_41 = Ai + o_i_8[:, None] * (H * BT) + o_i_3[None, :]
    p_Ai_42 = Ai + o_i_8[:, None] * (H * BT) + o_i_5[None, :]
    p_Ai_43 = Ai + o_i_8[:, None] * (H * BT) + o_i_7[None, :]
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_11)
    tl.store(p_Ai_22, b_Ai_22_c.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_22)
    tl.store(p_Ai_33, b_Ai_33_c.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_33)
    tl.store(p_Ai_44, b_Ai_44_store.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_44)
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_21)
    tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_31)
    tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_32)
    tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_41)
    tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_42)
    tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_43)


@register(solve_tril, backend=TritonAscendBackend)
@input_guard
def solve_tril_npu(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    assert A.shape[-1] in [16, 32, 64]
    output_dtype = A.dtype if output_dtype is None else output_dtype

    B, T, H, BT = A.shape
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    Ai = torch.zeros_like(A, dtype=output_dtype)
    kernel_kwargs = {
        'A': A,
        'Ai': Ai,
        'cu_seqlens': cu_seqlens,
        'chunk_indices': chunk_indices,
        'T': T,
        'H': H,
        'BT': BT,
        'IS_VARLEN': cu_seqlens is not None,
        'NT_OFFSET': 0,
        'BH_OFFSET': 0,
    }
    if BT == 16:
        merge_fn = solve_tril_16x16_kernel_npu
    elif BT == 32:
        merge_fn = merge_16x16_to_32x32_inverse_kernel_npu
    else:
        # Keep recursive row updates on Vector cores, outside the mixed Cube/Vector merge.
        Ad = torch.empty((B, T, H, 16), device=A.device, dtype=torch.float32)
        _launch_solve_tril_kernel(
            solve_tril_16x16_kernel_npu,
            NT=NT,
            bh_total=B * H,
            kernel_kwargs={**kernel_kwargs, 'Ai': Ad},
            sub_chunks=4,
        )
        kernel_kwargs['Ad'] = Ad
        merge_fn = merge_16x16_to_64x64_inverse_kernel_npu

    _launch_solve_tril_kernel(
        merge_fn,
        NT=NT,
        bh_total=B * H,
        kernel_kwargs=kernel_kwargs,
    )
    return Ai
