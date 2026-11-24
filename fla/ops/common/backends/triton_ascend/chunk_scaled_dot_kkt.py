# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Scaled key-key products for Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.backends import TritonAscendBackend, register
from fla.ops.common.backends.triton_ascend.chunk_delta_h import mload, mstore, need_dma_mask
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import compute_row_tile_block_size, get_npu_properties
from fla.utils.hardware import IS_ASCEND_950

# peak live fp32 tiles: b_A[BT,BT], b_k[BT,BK], tl.dot buffer; post-dot gate[BT,BT]
_CHUNK_SCALED_DOT_KKT_MEM_MULT = 4.0
_SAFETY_MARGIN = 0.85
_FALLBACK_BK = 16
_MAX_BK_FWD = 128


def _get_fwd_bk(BT: int, K: int, dtype: torch.dtype) -> int:
    """UB-safe BK tile size for chunk_scaled_dot_kkt_fwd on NPU."""
    # smaller tiles improve overlap for converted fp16/bf16 inputs; fp32 inputs favor larger tiles.
    max_bk = 32 if IS_ASCEND_950 and BT == 64 and K == 256 and dtype != torch.float32 else _MAX_BK_FWD
    return compute_row_tile_block_size(
        row_dim=BT,
        fixed_dim=K,
        memory_multiplier=_CHUNK_SCALED_DOT_KKT_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BK,
        min_block=16,
        max_block=min(max_bk, triton.next_power_of_2(K)),
    )


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'B', 'task_num', 'num_core'])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    task_num: tl.int64,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    MASK_DMA: tl.constexpr,
    GROUP_SIZE: tl.constexpr = 1,
):
    T = T.to(tl.int64)
    bt_stride = B.to(tl.int64) * T
    core_id = tl.program_id(0)
    o_i = tl.arange(0, BT)
    m_causal = o_i[:, None] > o_i[None, :]

    for task_id in tl.range(core_id, task_num, num_core):
        bh = B.to(tl.int64) * (HV // GROUP_SIZE)
        i_t = task_id.to(tl.int64) // bh
        i_bh = task_id.to(tl.int64) % bh
        i_b = i_bh // (HV // GROUP_SIZE)
        i_h = (i_bh % (HV // GROUP_SIZE)) * GROUP_SIZE
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = eos - bos
        else:
            bos = tl.cast(i_b, tl.int64) * T
        o_t = tl.cast(i_t, tl.int64) * BT + o_i
        m_t = o_t < T
        m_A = m_causal & (m_t[:, None] & m_t)
        # 1-token chunks: strictly-lower-tri kkt is 0.
        if tl.cast(i_t, tl.int64) * BT + 1 < T:
            b_gram = tl.zeros([BT, BT], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                o_k = i_k * BK + tl.arange(0, BK)
                m_k = (o_k >= 0) & (o_k < K)
                m_p_k = m_t[:, None] & m_k[None, :]
                p_k = k + (bos * H + i_h // (HV // H)) * K + o_t[:, None] * (H * K) + o_k[None, :]
                b_k = mload(p_k, m_p_k, MASK_DMA).to(tl.float32)
                # ascend tl.dot may clobber lhs; keep rhs on the original tile.
                b_k_lhs = b_k + 0.0
                b_gram = tl.dot(b_k_lhs, tl.trans(b_k), b_gram, allow_tf32=False)

            # Only the raw Gram is shared; each value head retains its own gates.
            for group_head in range(GROUP_SIZE):
                i_hv = i_h + group_head
                p_b = beta + i_hv * bt_stride + bos + o_t
                b_b = mload(p_b, m_t, MASK_DMA).to(tl.float32)
                if USE_G:
                    p_g = g + i_hv * bt_stride + bos + o_t
                    b_g = mload(p_g, m_t, MASK_DMA).to(tl.float32)
                    # Mask before exp2 to avoid overflowing the unused triangle.
                    b_g_diff = tl.where(m_A, b_g[:, None] - b_g[None, :], 0)
                    b_A = b_gram * exp2(b_g_diff)
                else:
                    b_A = b_gram
                b_A = tl.where(m_A, b_A * b_b[:, None], 0)
                p_A = A + (bos * HV + i_hv) * BT + o_t[:, None] * (BT * HV) + o_i[None, :]
                mstore(p_A, b_A.to(p_A.dtype.element_ty), m_t[:, None], MASK_DMA)


def chunk_scaled_dot_kkt_fwd_verifier(
    k,
    g=None,
    beta=None,
    cu_seqlens=None,
    chunk_size=64,
    output_dtype=torch.float32,
    chunk_indices=None,
) -> tuple[bool, str | None]:
    from fla.utils import IS_NPU
    if not IS_NPU:
        return False, "not running on NPU"
    if k.device.type != "npu":
        return False, "input device is not NPU"
    tensors = (k, beta) if g is None else (k, g, beta)
    if all(t.dtype in (torch.float32, torch.float16, torch.bfloat16) for t in tensors):
        return True, None
    return False, "unsupported dtype for NPU chunk_scaled_dot_kkt_fwd"


@register(chunk_scaled_dot_kkt_fwd, backend=TritonAscendBackend, verifier=chunk_scaled_dot_kkt_fwd_verifier)
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
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]` where `H` is the number of query/key heads.
        g (torch.Tensor, Optional):
            The cumulative sum of the gate tensor of shape `[B, T, HV]`. Default: `None`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, HV]` where `HV` is the number of value/output heads.
        cu_seqlens (torch.LongTensor, Optional):
            The cumulative sequence lengths of the input tensor. Default: `None`.
        chunk_size (int, Optional):
            The chunk size. Default: 64.
        output_dtype (torch.dtype, Optional):
            The dtype of the output tensor. Default: `torch.float32`.
        chunk_indices (torch.LongTensor, Optional):
            The chunk indices of the input tensor. Default: `None`.

    Returns:
        A (torch.Tensor):
            beta * K * K^T of shape `[B, T, HV, BT]` where `BT` is the chunk size.
            For GVA, H < HV and HV % H == 0. For standard attention, H == HV.
    """
    B, T, H, K, HV = *k.shape, beta.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.zeros(B, T, HV, BT, device=k.device, dtype=output_dtype)
    BK = _get_fwd_bk(BT=BT, K=K, dtype=k.dtype)
    group_size = 1
    if (
        IS_ASCEND_950 and HV in (2 * H, 3 * H, 4 * H, 8 * H, 16 * H) and K == 128 and BT == 64
        and cu_seqlens is None and T % BT == 0 and k.element_size() == 2
    ):
        # cap reuse at four heads to retain task parallelism for larger GVA ratios.
        group_size = min(HV // H, 4)

    num_core = get_npu_properties()['num_aicore']
    g_arg = torch.permute(g, (2, 0, 1)).contiguous() if g is not None else g
    beta_arg = torch.permute(beta, (2, 0, 1)).contiguous()
    chunk_scaled_dot_kkt_fwd_kernel[(num_core,)](
        k=k,
        g=g_arg,
        beta=beta_arg,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        task_num=NT * B * HV // group_size,
        num_core=num_core,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BK=BK,
        GROUP_SIZE=group_size,
        MASK_DMA=need_dma_mask(T, BT, K, BK, K, BK, cu_seqlens is not None, cu_seqlens),
    )
    return A
