# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Causal 1D convolution kernels adapted for triton-ascend on Huawei NPU."""

import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.utils import prepare_chunk_indices
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    compute_ub_block_size,
    max_grid_axis_chunks,
)

# Peak live fp32 tiles; bwd multipliers calibrated on Ascend910 (192 KiB UB).
# Empirical BT*BD limits: plain bwd <= 1024, bwd with saved y (swish) <= 512.
_FWD_MEM_MULT = 5.0
_FWD_FUSED_MEM_MULT = 5.0
_FWD_INIT_MEM_MULT = 24.0
# Plain conv bwd (swish handled via swish_bwd_npu + dy_conv).
_BWD_MEM_MULT = 24.0
# Fused activation path kept for initial_state / dht cases.
_BWD_ACT_MEM_MULT = 24.0
_BWD_INIT_FP16_MEM_MULT = 48.0
_DH0_MEM_MULT = 4.0
_DH0_ACT_MEM_MULT = 6.0
_STATES_MEM_MULT = 4.0
_UPDATE_MEM_MULT = 3.0
_SEQ_BWD_MEM_MULT = 4.0
_UB_SAFETY_MARGIN = 0.85
_DTYPE_SIZE = 4  # fp32 accumulators in kernel tiles
_MAX_BT = 64
_MAX_BD = 128
_FWD_INIT_MAX_TILE_PRODUCT = 1024
_BWD_MAX_TILE_PRODUCT = 1024
_BWD_ACT_MAX_TILE_PRODUCT = 512
_BWD_INIT_MAX_TILE_PRODUCT = 1024
_BWD_INIT_FP16_MAX_TILE_PRODUCT = 512
_FALLBACK_BD = 16
_FALLBACK_BT = 32
_ACTIVATION_NONE = 0
_ACTIVATION_SWISH = 1


def _activation_id(activation: str | None) -> int:
    if activation in ('swish', 'silu'):
        return _ACTIVATION_SWISH
    return _ACTIVATION_NONE


def _npu_chunk_size(T: int, BT: int) -> int:
    BT = min(max(BT, 1), 64)
    if BT not in (1, 2, 4, 8, 16, 32, 64):
        BT = triton.next_power_of_2(BT)
    # Ascend compiler requires power-of-2 BT; pad with mask when BT > T.
    if T not in (1, 2, 4, 8, 16, 32, 64):
        BT = min(triton.next_power_of_2(T), 64)
    else:
        BT = min(BT, T, 64)
    return BT


def _npu_max_axis_chunks(grid_dim0: int, batch: int = 1) -> int:
    denom = grid_dim0 * batch
    if denom > ASCEND_MAX_GRID_DIM:
        raise RuntimeError(
            f'Ascend Triton grid dim0*batch={denom} exceeds {ASCEND_MAX_GRID_DIM}',
        )
    return max_grid_axis_chunks(1, denom)


def _clamp_tile_product(BT: int, BD: int, max_product: int) -> tuple[int, int]:
    """Shrink BT/BD so their product stays within compiler UB limits."""
    while max_product < BT * BD and BT > 1:
        BT //= 2
    while max_product < BT * BD and BD > 1:
        BD //= 2
    return BD, max(BT, 1)


def _fwd_ub_tile_config(
    T: int,
    BT: int,
    D: int,
    initial_state: torch.Tensor | None,
    activation: str | None = None,
    residual: torch.Tensor | None = None,
) -> tuple[int, int]:
    BT = _npu_chunk_size(T, BT)
    fuse_post = activation in ('swish', 'silu') or residual is not None
    if initial_state is not None:
        mem_mult = _FWD_INIT_MEM_MULT
        max_product = _FWD_INIT_MAX_TILE_PRODUCT
    elif fuse_post:
        mem_mult = _FWD_FUSED_MEM_MULT
        max_product = None
    else:
        mem_mult = _FWD_MEM_MULT
        max_product = None

    BD = compute_row_tile_block_size(
        BT,
        D,
        mem_mult,
        tiling_row=False,
        safety_margin=_UB_SAFETY_MARGIN,
        dtype_size=_DTYPE_SIZE,
        fallback=_FALLBACK_BD,
        min_block=1,
        max_block=64,
    )
    BT = compute_row_tile_block_size(
        T,
        BD,
        mem_mult,
        tiling_row=True,
        safety_margin=_UB_SAFETY_MARGIN,
        dtype_size=_DTYPE_SIZE,
        fallback=_FALLBACK_BT,
        min_block=1,
        max_block=_MAX_BT,
    )
    BT = _npu_chunk_size(T, BT)
    BD = compute_row_tile_block_size(
        BT,
        D,
        mem_mult,
        tiling_row=False,
        safety_margin=_UB_SAFETY_MARGIN,
        dtype_size=_DTYPE_SIZE,
        fallback=_FALLBACK_BD,
        min_block=1,
        max_block=64,
    )
    if max_product is not None:
        BD, BT = _clamp_tile_product(BT, BD, max_product)
    BD = _boost_fwd_bd(BD, T, BT, D, 1)
    return BD, BT


def _bwd_ub_tile_config(
    T: int,
    BT: int,
    D: int,
    initial_state: torch.Tensor | None,
    activation: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[int, int]:
    BT = _npu_chunk_size(T, BT)
    use_activation = activation in ('swish', 'silu')
    if use_activation:
        mem_mult = _BWD_ACT_MEM_MULT
        max_product = _BWD_ACT_MAX_TILE_PRODUCT
    elif initial_state is not None:
        if dtype == torch.float16:
            mem_mult = _BWD_INIT_FP16_MEM_MULT
            max_product = _BWD_INIT_FP16_MAX_TILE_PRODUCT
        else:
            mem_mult = _BWD_MEM_MULT
            max_product = _BWD_INIT_MAX_TILE_PRODUCT
    else:
        mem_mult = _BWD_MEM_MULT
        max_product = _BWD_MAX_TILE_PRODUCT

    BD = compute_row_tile_block_size(
        BT,
        D,
        mem_mult,
        tiling_row=False,
        safety_margin=_UB_SAFETY_MARGIN,
        dtype_size=_DTYPE_SIZE,
        fallback=_FALLBACK_BD,
        min_block=1,
        max_block=_MAX_BD,
    )
    BT = compute_row_tile_block_size(
        T,
        BD,
        mem_mult,
        tiling_row=True,
        safety_margin=_UB_SAFETY_MARGIN,
        dtype_size=_DTYPE_SIZE,
        fallback=_FALLBACK_BT,
        min_block=1,
        max_block=_MAX_BT,
    )
    BT = _npu_chunk_size(T, BT)
    BD = compute_row_tile_block_size(
        BT,
        D,
        mem_mult,
        tiling_row=False,
        safety_margin=_UB_SAFETY_MARGIN,
        dtype_size=_DTYPE_SIZE,
        fallback=_FALLBACK_BD,
        min_block=1,
        max_block=_MAX_BD,
    )
    BD, BT = _clamp_tile_product(BT, BD, max_product)
    return BD, BT


def _clamp_bd_for_grid(B: int, NT: int, D: int, BD: int) -> int:
    """Keep grid0*NT*B within Ascend launch limit; host NT chunking is accounted for."""
    while BD < _MAX_BD:
        grid0 = triton.cdiv(D, BD)
        max_nt = max(1, ASCEND_MAX_GRID_DIM // max(grid0 * B, 1))
        if grid0 * min(NT, max_nt) * B <= ASCEND_MAX_GRID_DIM:
            break
        BD *= 2
    return BD


def _boost_fwd_bd(BD: int, T: int, BT: int, D: int, B: int) -> int:
    """Use wider D tiles on long sequences to cut grid-axis pressure."""
    NT = triton.cdiv(T, BT)
    if D >= 2048 and T >= 16384 and BD < _MAX_BD:
        boosted = min(_MAX_BD, 128)
        return _clamp_bd_for_grid(B, NT, D, boosted)
    return _clamp_bd_for_grid(B, NT, D, BD)


def _npu_tile_config(
    T: int,
    BT: int,
    D: int,
    initial_state: torch.Tensor | None,
    activation: str | None = None,
    residual: torch.Tensor | None = None,
) -> tuple[int, int]:
    BD, BT = _fwd_ub_tile_config(T, BT, D, initial_state, activation, residual)
    return BD, BT


def _npu_bwd_tile_config(
    T: int,
    BT: int,
    D: int,
    dtype: torch.dtype,
    initial_state: torch.Tensor | None,
    activation: str | None = None,
) -> tuple[int, int]:
    BD, BT = _bwd_ub_tile_config(T, BT, D, initial_state, activation, dtype)
    return BD, BT


def _dh0_bd(D: int, activation: str | None) -> int:
    mem_mult = _DH0_ACT_MEM_MULT if activation in ('swish', 'silu') else _DH0_MEM_MULT
    return compute_ub_block_size(
        D,
        mem_mult,
        safety_margin=_UB_SAFETY_MARGIN,
        dtype_size=_DTYPE_SIZE,
        fallback=_FALLBACK_BD,
        max_block=_MAX_BD,
    )


def _states_bd(D: int) -> int:
    return compute_ub_block_size(
        D,
        _STATES_MEM_MULT,
        safety_margin=_UB_SAFETY_MARGIN,
        dtype_size=_DTYPE_SIZE,
        fallback=_FALLBACK_BD,
        max_block=_MAX_BD,
    )


def _update_bd(D: int) -> int:
    return compute_ub_block_size(
        D,
        _UPDATE_MEM_MULT,
        safety_margin=_UB_SAFETY_MARGIN,
        dtype_size=_DTYPE_SIZE,
        fallback=_FALLBACK_BD,
        max_block=_MAX_BD,
    )


def _seq_bwd_block(B: int, T: int, D: int) -> int:
    return compute_ub_block_size(
        B * T * D,
        _SEQ_BWD_MEM_MULT,
        safety_margin=_UB_SAFETY_MARGIN,
        dtype_size=_DTYPE_SIZE,
        fallback=1024,
        min_block=256,
    )


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'SAVE_Y_PRE': lambda args: args['y_pre'] is not None,
})
@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    y,
    y_pre,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_y_n,
    stride_y_t,
    stride_y_d,
    stride_res_n,
    stride_res_t,
    stride_res_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SAVE_Y_PRE: tl.constexpr,
    CHUNK_OFFSET: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        p_x = x + bos * stride_x_t
        p_y = y + bos * stride_y_t
    else:
        i_n = i_b
        i_t = i_t + CHUNK_OFFSET
        bos = (i_b * T).to(tl.int64)
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_n
        p_y = y + tl.cast(i_b, tl.int64) * stride_y_n

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = (o_t >= 0) & (o_t < T)
    b_y = tl.zeros((BT, BD), dtype=tl.float32)

    for i_w in tl.static_range(-W + 1, 1):
        o_x = o_t + i_w
        m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :]
        b_yi = tl.load(
            p_x + o_x[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
            mask=m_x,
            other=0,
        ).to(tl.float32)

        if USE_INITIAL_STATE:
            m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d[None, :]
            b_yi += tl.load(
                initial_state + i_n * D * W + o_d[None, :] * W + (o_x + W)[:, None],
                mask=m_c,
                other=0,
            ).to(tl.float32)

        if HAS_WEIGHT:
            w_col = i_w + W - 1
            b_yi = b_yi * tl.sum(b_w * (o_w == w_col), 1)[None, :]
        b_y += b_yi

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)[None, :]

    if SAVE_Y_PRE:
        if IS_VARLEN:
            p_y_pre = y_pre + bos * stride_y_t
        else:
            p_y_pre = y_pre + tl.cast(i_b, tl.int64) * stride_y_n
        tl.store(
            p_y_pre + o_t[:, None] * stride_y_t + o_d[None, :] * stride_y_d,
            tl.cast(b_y, dtype=y_pre.dtype.element_ty, fp_downcast_rounding='rtne'),
            mask=m_t[:, None] & m_d[None, :],
        )

    if ACTIVATION == 1:
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        if IS_VARLEN:
            p_res = residual + bos * stride_res_t
        else:
            p_res = residual + tl.cast(i_b, tl.int64) * stride_res_n
        b_res = tl.load(
            p_res + o_t[:, None] * stride_res_t + o_d[None, :] * stride_res_d,
            mask=m_t[:, None] & m_d[None, :],
            other=0,
        ).to(tl.float32)
        b_y += b_res

    tl.store(
        p_y + o_t[:, None] * stride_y_t + o_d[None, :] * stride_y_d,
        tl.cast(b_y, dtype=y.dtype.element_ty, fp_downcast_rounding='rtne'),
        mask=m_t[:, None] & m_d[None, :],
    )


def _use_seq_bwd(
    T: int,
    dtype: torch.dtype,
    initial_state: torch.Tensor | None,
    dht: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    return (
        cu_seqlens is None
        and initial_state is None
        and dht is None
        and dtype == torch.bfloat16
        and T <= 16
    )


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['dw'] is not None,
    'HAS_BIAS': lambda args: args['db'] is not None,
    'USE_ACTIVATION': lambda args: args['y'] is not None,
})
@triton.jit
def causal_conv1d_bwd_seq_kernel(
    x,
    y,
    weight,
    dy,
    dx,
    dw,
    db,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_dx_n,
    stride_dx_t,
    stride_dx_d,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    stride_y_n,
    stride_y_t,
    stride_y_d,
    B,
    TC: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n_elements = B * TC * D
    mask = offs < n_elements
    d = offs % D
    tmp = offs // D
    t = tmp % TC
    b = tmp // TC

    b_dx = tl.zeros((BLOCK,), dtype=tl.float32)
    for i_w in tl.static_range(0, W):
        t_dy = t + i_w
        dy_off = b * stride_dy_n + t_dy * stride_dy_t + d * stride_dy_d
        b_dy = tl.load(dy + dy_off, mask=mask & (t_dy < TC), other=0.).to(tl.float32)
        if USE_ACTIVATION:
            y_off = b * stride_y_n + t_dy * stride_y_t + d * stride_y_d
            b_y = tl.load(y + y_off, mask=mask & (t_dy < TC), other=0.).to(tl.float32)
            b_ys = tl.sigmoid(b_y)
            b_dy = b_dy * b_ys * (1.0 + b_y * (1.0 - b_ys))
        if HAS_WEIGHT:
            w_idx = W - i_w - 1
            b_w = tl.load(weight + d * W + w_idx, mask=mask, other=0.).to(tl.float32)
            b_dx += b_dy * b_w
        else:
            b_dx += b_dy

    dx_off = b * stride_dx_n + t * stride_dx_t + d * stride_dx_d
    tl.store(dx + dx_off, b_dx.to(dx.dtype.element_ty), mask=mask)

    if HAS_WEIGHT:
        x_off = b * stride_x_n + t * stride_x_t + d * stride_x_d
        b_x = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
        i_tg = b * TC + t
        for i_w in tl.static_range(0, W):
            t_dy = t + i_w
            dy_off = b * stride_dy_n + t_dy * stride_dy_t + d * stride_dy_d
            b_dy = tl.load(dy + dy_off, mask=mask & (t_dy < TC), other=0.).to(tl.float32)
            if USE_ACTIVATION:
                y_off = b * stride_y_n + t_dy * stride_y_t + d * stride_y_d
                b_y = tl.load(y + y_off, mask=mask & (t_dy < TC), other=0.).to(tl.float32)
                b_ys = tl.sigmoid(b_y)
                b_dy = b_dy * b_ys * (1.0 + b_y * (1.0 - b_ys))
            w_idx = W - i_w - 1
            tl.store(
                dw + (i_tg * D + d) * W + w_idx,
                (b_dy * b_x).to(dw.dtype.element_ty),
                mask=mask,
            )

    if HAS_BIAS:
        i_tg = b * TC + t
        dy_off = b * stride_dy_n + t * stride_dy_t + d * stride_dy_d
        b_dy0 = tl.load(dy + dy_off, mask=mask, other=0.)
        if USE_ACTIVATION:
            y_off = b * stride_y_n + t * stride_y_t + d * stride_y_d
            b_y = tl.load(y + y_off, mask=mask, other=0.).to(tl.float32)
            b_ys = tl.sigmoid(b_y)
            b_dy0 = b_dy0.to(tl.float32) * b_ys * (1.0 + b_y * (1.0 - b_ys))
        tl.store(db + i_tg * D + d, b_dy0.to(db.dtype.element_ty), mask=mask)


def _use_fast_bwd_kernel(
    use_seq: bool,
    initial_state: torch.Tensor | None,
    dht: torch.Tensor | None,
    y_pre: torch.Tensor | None,
    use_swish_split: bool,
) -> bool:
    """Select slim bwd kernel + fused chunk dw/db reduction."""
    if use_seq or initial_state is not None or dht is not None:
        return False
    return y_pre is None or use_swish_split


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['db'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def causal_conv1d_bwd_fast_kernel(
    x,
    weight,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_dx_n,
    stride_dx_t,
    stride_dx_d,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    CHUNK_OFFSET: tl.constexpr,
    NT: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        p_x = x + bos * stride_x_t
        p_dy = dy + bos * stride_dy_t
        p_dx = dx + bos * stride_dx_t
    else:
        i_t = i_t + CHUNK_OFFSET
        i_tg = i_b * NT + i_t
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_n
        p_dy = dy + tl.cast(i_b, tl.int64) * stride_dy_n
        p_dx = dx + tl.cast(i_b, tl.int64) * stride_dx_n

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = (o_t >= 0) & (o_t < T)

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)
    b_x = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_WEIGHT:
        b_x = tl.load(
            p_x + o_t[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
            mask=m_t[:, None] & m_d[None, :],
            other=0,
        ).to(tl.float32)

    for i_w in tl.static_range(0, W):
        o_dy = o_t + i_w
        m_dy = ((o_dy >= 0) & (o_dy < T))[:, None] & m_d[None, :]
        b_dy = tl.load(
            p_dy + o_dy[:, None] * stride_dy_t + o_d[None, :] * stride_dy_d,
            mask=m_dy,
            other=0,
        ).to(tl.float32)
        if HAS_WEIGHT:
            w_col = W - i_w - 1
            w_coeff = tl.load(weight + o_d * W + w_col, mask=m_d, other=0).to(tl.float32)
            b_dx += b_dy * w_coeff[None, :]
            b_dw = tl.sum(b_dy * b_x, 0)
            tl.store(
                dw + i_tg * D * W + o_d * W + w_col,
                b_dw.to(dw.dtype.element_ty),
                mask=m_d,
            )
        else:
            b_dx += b_dy
        if HAS_BIAS and i_w == 0:
            b_db += tl.sum(b_dy, 0)

    if HAS_BIAS:
        b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding='rtne')
        tl.store(db + i_tg * D + o_d, b_db, mask=m_d)

    tl.store(
        p_dx + o_t[:, None] * stride_dx_t + o_d[None, :] * stride_dx_d,
        tl.cast(b_dx, dtype=dx.dtype.element_ty, fp_downcast_rounding='rtne'),
        mask=m_t[:, None] & m_d[None, :],
    )


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['dw'] is not None,
    'HAS_BIAS': lambda args: args['db'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'USE_FINAL_STATE': lambda args: args['dht'] is not None,
    'USE_ACTIVATION': lambda args: args['y'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def causal_conv1d_bwd_kernel(
    x,
    y,
    weight,
    initial_state,
    dht,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_dx_n,
    stride_dx_t,
    stride_dx_d,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    stride_y_n,
    stride_y_t,
    stride_y_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    CHUNK_OFFSET: tl.constexpr,
    NT: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        p_x = x + bos * stride_x_t
        p_dy = dy + bos * stride_dy_t
        p_dx = dx + bos * stride_dx_t
    else:
        i_t = i_t + CHUNK_OFFSET
        i_tg = i_b * NT + i_t
        i_n = i_b
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_n
        p_dy = dy + tl.cast(i_b, tl.int64) * stride_dy_n
        p_dx = dx + tl.cast(i_b, tl.int64) * stride_dx_n

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = (o_t >= 0) & (o_t < T)

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)

    b_x = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_WEIGHT:
        b_x = tl.load(
            p_x + o_t[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
            mask=m_t[:, None] & m_d[None, :],
            other=0,
        ).to(tl.float32)

    for i_w in tl.static_range(0, W):
        o_dy = o_t + i_w
        m_dy = ((o_dy >= 0) & (o_dy < T))[:, None] & m_d[None, :]
        b_dy = tl.load(
            p_dy + o_dy[:, None] * stride_dy_t + o_d[None, :] * stride_dy_d,
            mask=m_dy,
            other=0,
        ).to(tl.float32)

        if USE_ACTIVATION:
            if IS_VARLEN:
                p_y = y + bos * stride_y_t
            else:
                p_y = y + tl.cast(i_b, tl.int64) * stride_y_n
            b_y = tl.load(
                p_y + o_dy[:, None] * stride_y_t + o_d[None, :] * stride_dy_d,
                mask=m_dy,
                other=0,
            ).to(tl.float32)
            b_ys = tl.sigmoid(b_y)
            b_dy = b_dy * b_ys * (1.0 + b_y * (1.0 - b_ys))

        if HAS_WEIGHT:
            w_col = W - i_w - 1
            w_coeff = tl.load(weight + o_d * W + w_col, mask=m_d, other=0).to(tl.float32)
            b_wdy = b_dy * w_coeff[None, :]
            b_dw = tl.sum(b_dy * b_x, 0)
            if USE_INITIAL_STATE:
                mask_head_rows = (o_t < i_w) & (o_t < T)
                b_dy_head = tl.load(
                    p_dy + o_t[:, None] * stride_dy_t + o_d[None, :] * stride_dy_d,
                    mask=(mask_head_rows[:, None] & m_d[None, :]),
                    other=0.0,
                ).to(tl.float32)
                if USE_ACTIVATION:
                    if IS_VARLEN:
                        p_y = y + bos * stride_y_t
                    else:
                        p_y = y + tl.cast(i_b, tl.int64) * stride_y_n
                    b_y_head = tl.load(
                        p_y + o_t[:, None] * stride_y_t + o_d[None, :] * stride_y_d,
                        mask=(mask_head_rows[:, None] & m_d[None, :]),
                        other=0.0,
                    ).to(tl.float32)
                    b_ys_head = tl.sigmoid(b_y_head)
                    b_dy_head = b_dy_head * b_ys_head * (1.0 + b_y_head * (1.0 - b_ys_head))
                o_c = W - i_w + o_t
                mask_c = (mask_head_rows & (o_c >= 1) & (o_c < W))
                b_xc = tl.load(
                    initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None],
                    mask=(mask_c[:, None] & m_d[None, :]),
                    other=0.0,
                ).to(tl.float32)
                b_dw += tl.sum(b_dy_head * b_xc, 0)
            tl.store(dw + i_tg * D * W + o_d * W + w_col, b_dw.to(dw.dtype.element_ty), mask=m_d)
        else:
            b_wdy = b_dy

        if HAS_BIAS and i_w == 0:
            b_db += tl.sum(b_dy, 0)
        b_dx += b_wdy

    if HAS_BIAS:
        b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding='rtne')
        tl.store(db + i_tg * D + o_d, b_db, mask=m_d)

    if USE_FINAL_STATE:
        if i_t * BT + BT >= T - W:
            start_tok = T - (W - 1)
            offset = i_t * BT + tl.arange(0, BT)
            tok_idx = offset - start_tok
            mask = (offset >= start_tok) & (offset < T)
            w_idx = 1 + tok_idx
            dht_off = i_n * D * W + o_d[None, :] * W + w_idx[:, None]
            b_dht = tl.load(dht + dht_off, mask=mask[:, None] & m_d[None, :], other=0.).to(tl.float32)
            b_dx += b_dht

    tl.store(
        p_dx + o_t[:, None] * stride_dx_t + o_d[None, :] * stride_dx_d,
        tl.cast(b_dx, dtype=dx.dtype.element_ty, fp_downcast_rounding='rtne'),
        mask=m_t[:, None] & m_d[None, :],
    )


@triton.heuristics({
    'USE_ACTIVATION': lambda args: args['y'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def compute_dh0_kernel(
    dy,
    y,
    weight,
    dh0,
    cu_seqlens,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    stride_y_n,
    stride_y_t,
    stride_y_d,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    CHUNK_OFFSET: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1) + CHUNK_OFFSET

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = eos - bos
        dy_base = dy + bos * stride_dy_t
    else:
        seq_len = T
        dy_base = dy + tl.cast(i_n, tl.int64) * stride_dy_n

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    for i_w in tl.static_range(1, W):
        b_dh0 = tl.zeros([BD], dtype=tl.float32)

        for t in tl.static_range(0, W - 1):
            if t < i_w:
                w_idx = i_w - 1 - t
                p_dy = dy_base + t * stride_dy_t + o_d * stride_dy_d
                m_t = (t < seq_len) & m_d
                b_dy = tl.load(p_dy, mask=m_t, other=0).to(tl.float32)

                if USE_ACTIVATION:
                    if IS_VARLEN:
                        p_y = y + bos * stride_y_t + t * stride_y_t + o_d * stride_y_d
                    else:
                        p_y = y + tl.cast(i_n, tl.int64) * stride_y_n + t * stride_y_t + o_d * stride_y_d
                    b_y = tl.load(p_y, mask=m_t, other=0).to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))

                b_w_col = tl.load(weight + o_d * W + w_idx, mask=m_d, other=0).to(tl.float32)
                b_dh0 += tl.where(m_t, b_dy * b_w_col, 0)

        p_dh0 = dh0 + i_n * D * W + o_d * W + i_w
        tl.store(p_dh0, b_dh0.to(dh0.dtype.element_ty), mask=m_d)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def causal_conv1d_states_fwd_kernel(
    x,
    initial_state,
    final_state,
    cu_seqlens,
    T,
    D,
    W,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    BD: tl.constexpr,
    BW: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    CHUNK_OFFSET: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1) + CHUNK_OFFSET

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = (eos - bos).to(tl.int32)
        p_x = x + bos * stride_x_t
    else:
        seq_len = T
        p_x = x + tl.cast(i_n, tl.int64) * stride_x_n

    o_w = W - BW + tl.arange(0, BW)
    m_w = o_w >= 0
    o_t = seq_len - BW + tl.arange(0, BW)
    m_t = (o_t >= 0) & (o_t < seq_len)

    b_x = tl.load(
        p_x + o_t[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
        mask=m_t[:, None] & m_d[None, :],
        other=0,
    ).to(tl.float32)

    if USE_INITIAL_STATE:
        if seq_len < BW:
            o_c = W - (BW - seq_len) + tl.arange(0, BW)
            m_c = (o_c >= 0) & (o_c < W)
            b_cache = tl.load(
                initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None],
                mask=m_d[None, :] & m_c[:, None],
                other=0,
            ).to(tl.float32)
            b_x += b_cache

    p_final = final_state + tl.cast(i_n, tl.int64) * D * W + o_d[:, None] * W + o_w[None, :]
    tl.store(p_final, tl.trans(b_x).to(final_state.dtype.element_ty), mask=m_d[:, None] & m_w[None, :])


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
})
@triton.jit
def causal_conv1d_update_kernel(
    x,
    cache,
    residual,
    y,
    weight,
    bias,
    stride_x_n,
    stride_x_d,
    stride_y_n,
    stride_y_d,
    stride_res_n,
    stride_res_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    CHUNK_OFFSET: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1) + CHUNK_OFFSET

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    b_x = tl.load(x + i_n * stride_x_n + o_d * stride_x_d, mask=m_d, other=0).to(tl.float32)

    b_y = tl.zeros((BD,), dtype=tl.float32)
    for iw in tl.static_range(0, W):
        if iw < W - 1:
            b_c = tl.load(cache + i_n * D * W + o_d * W + (iw + 1), mask=m_d, other=0).to(tl.float32)
        else:
            b_c = b_x
        tl.store(
            cache + i_n * D * W + o_d * W + iw,
            tl.cast(b_c, dtype=cache.dtype.element_ty, fp_downcast_rounding='rtne'),
            mask=m_d,
        )
        if HAS_WEIGHT:
            b_y += b_c * tl.load(weight + o_d * W + iw, mask=m_d, other=0).to(tl.float32)
        else:
            b_y += b_c

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d)

    if ACTIVATION == 1:
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        b_y += tl.load(
            residual + i_n * stride_res_n + o_d * stride_res_d,
            mask=m_d,
            other=0,
        ).to(tl.float32)

    tl.store(
        y + i_n * stride_y_n + o_d * stride_y_d,
        tl.cast(b_y, dtype=y.dtype.element_ty, fp_downcast_rounding='rtne'),
        mask=m_d,
    )


def _launch_fwd_core(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None,
    B: int,
    T: int,
    D: int,
    W: int,
    BT: int,
    residual: torch.Tensor | None = None,
    activation: str | None = None,
    BD: int | None = None,
    y_pre: torch.Tensor | None = None,
) -> torch.Tensor:
    if BD is None:
        BD, BT = _npu_tile_config(
            T, BT, D, initial_state, activation=activation, residual=residual,
        )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    BW = triton.next_power_of_2(W)

    stride_x_n, stride_x_t, stride_x_d = x.stride()
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    stride_y_n, stride_y_t, stride_y_d = y.stride()
    stride_res_n = stride_res_t = stride_res_d = 0
    if residual is not None:
        if residual.dim() == 3:
            stride_res_n, stride_res_t, stride_res_d = residual.stride()
        else:
            stride_res_t, stride_res_d = residual.stride()

    max_nt = _npu_max_axis_chunks(triton.cdiv(D, BD), B)
    act_id = _activation_id(activation)
    kernel_kwargs = dict(
        x=x,
        y=y,
        y_pre=y_pre,
        weight=weight,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        BD=BD,
        ACTIVATION=act_id,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        stride_y_n=stride_y_n,
        stride_y_t=stride_y_t,
        stride_y_d=stride_y_d,
        stride_res_n=stride_res_n,
        stride_res_t=stride_res_t,
        stride_res_d=stride_res_d,
    )
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        grid = (triton.cdiv(D, BD), nt_len, B)
        if cu_seqlens is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['CHUNK_OFFSET'] = 0
        else:
            kernel_kwargs['chunk_indices'] = chunk_indices
            kernel_kwargs['CHUNK_OFFSET'] = nt_off
        causal_conv1d_fwd_kernel[grid](**kernel_kwargs)
    return y


@input_guard(no_guard_contiguous=['x'])
def causal_conv1d_fwd_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    BT: int = 64,
    layout_fallback: bool = False,
):
    del layout_fallback
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    B, T, D = x.shape[0], x.shape[1], weight.shape[0]
    W = weight.shape[1]

    BD, BT = _npu_tile_config(
        T, BT, D, initial_state, activation=activation, residual=residual,
    )
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)

    use_swish = activation in ('swish', 'silu')
    save_y_pre = use_swish and x.requires_grad and torch.is_grad_enabled()
    if save_y_pre:
        y_pre = torch.empty_like(x, memory_format=torch.contiguous_format)
        y = _launch_fwd_core(
            x, weight, bias, initial_state, cu_seqlens, chunk_indices, B, T, D, W, BT,
            residual=residual,
            activation=activation,
            BD=BD,
            y_pre=y_pre,
        )
        x._fla_causal_conv1d_y_pre = y_pre
    else:
        y = _launch_fwd_core(
            x, weight, bias, initial_state, cu_seqlens, chunk_indices, B, T, D, W, BT,
            residual=residual,
            activation=activation,
            BD=BD,
        )

    final_state = None
    if output_final_state:
        final_state = causal_conv1d_update_states_npu(
            x=x,
            state_len=W,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )
    return y.view(shape), final_state


def causal_conv1d_bwd_npu(
    x: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    BT: int = 64,
    layout_fallback: bool = False,
):
    del layout_fallback
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    B, T, D = x.shape
    W = weight.shape[1] if weight is not None else None

    use_swish = activation in ('swish', 'silu')
    use_swish_split = use_swish and initial_state is None and dht is None

    BD, BT = _npu_bwd_tile_config(
        T, BT, D, x.dtype, initial_state, None if use_swish_split else activation,
    )
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)

    dr = dy if residual is not None else None

    y_pre = getattr(x, '_fla_causal_conv1d_y_pre', None)
    if y_pre is not None:
        del x._fla_causal_conv1d_y_pre

    dy_conv = dy
    if use_swish:
        if y_pre is None:
            BD_f, BT_f = _npu_tile_config(T, BT, D, initial_state)
            chunk_indices_f = chunk_indices
            if cu_seqlens is not None:
                chunk_indices_f = prepare_chunk_indices(cu_seqlens, BT_f, cu_seqlens_cpu=cu_seqlens_cpu)
            y_pre = _launch_fwd_core(
                x, weight, bias, initial_state, cu_seqlens, chunk_indices_f,
                B, T, D, W, BT_f,
                activation=None,
                BD=BD_f,
            )
        if use_swish_split:
            from fla.modules.backends.triton_ascend.activations import swish_bwd_npu
            dy_conv = swish_bwd_npu(y_pre, dy)
            if not dy_conv.is_contiguous():
                dy_conv = dy_conv.contiguous()

    stride_x_n, stride_x_t, stride_x_d = x.stride()
    use_seq = _use_seq_bwd(T, x.dtype, initial_state, dht, cu_seqlens)
    stride_dy_n, stride_dy_t, stride_dy_d = dy_conv.stride()

    dx = torch.zeros_like(x)
    stride_dx_n, stride_dx_t, stride_dx_d = dx.stride()

    use_fast_kernel = False
    dw = None
    db = None

    if use_seq:
        block = _seq_bwd_block(B, T, D)
        dw = weight.new_empty(B * T, *weight.shape, dtype=torch.float) if weight is not None else None
        db = bias.new_empty(B * T, *bias.shape, dtype=torch.float) if bias is not None else None
        stride_y_n = stride_y_t = stride_y_d = 0
        if y_pre is not None and not use_swish_split:
            stride_y_n, stride_y_t, stride_y_d = y_pre.stride()
        grid = (triton.cdiv(B * T * D, block),)
        causal_conv1d_bwd_seq_kernel[grid](
            x=x,
            y=y_pre if not use_swish_split else None,
            weight=weight,
            dy=dy_conv,
            dx=dx,
            dw=dw,
            db=db,
            stride_x_n=stride_x_n,
            stride_x_t=stride_x_t,
            stride_x_d=stride_x_d,
            stride_dx_n=stride_dx_n,
            stride_dx_t=stride_dx_t,
            stride_dx_d=stride_dx_d,
            stride_dy_n=stride_dy_n,
            stride_dy_t=stride_dy_t,
            stride_dy_d=stride_dy_d,
            stride_y_n=stride_y_n,
            stride_y_t=stride_y_t,
            stride_y_d=stride_y_d,
            B=B,
            TC=T,
            D=D,
            W=W,
            BLOCK=block,
        )
    else:
        NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
        if not dy_conv.is_contiguous():
            dy_conv = dy_conv.contiguous()
        stride_dy_n, stride_dy_t, stride_dy_d = dy_conv.stride()
        stride_y_n = stride_y_t = stride_y_d = 0
        if y_pre is not None and not use_swish_split:
            stride_y_n, stride_y_t, stride_y_d = y_pre.stride()

        use_fast_kernel = _use_fast_bwd_kernel(
            use_seq, initial_state, dht, y_pre, use_swish_split,
        )
        if use_fast_kernel:
            dw = weight.new_empty(B * NT, D, W, device=x.device) if weight is not None else None
            db = bias.new_empty(B * NT, D, device=x.device) if bias is not None else None
            bwd_fn = causal_conv1d_bwd_fast_kernel
        else:
            dw = weight.new_empty(B * NT, *weight.shape, dtype=x.dtype) if weight is not None else None
            db = bias.new_empty(B * NT, *bias.shape, dtype=x.dtype) if bias is not None else None
            bwd_fn = causal_conv1d_bwd_kernel

        max_nt = _npu_max_axis_chunks(triton.cdiv(D, BD), B)
        kernel_kwargs = dict(
            x=x,
            dy=dy_conv,
            dx=dx,
            cu_seqlens=cu_seqlens,
            B=B,
            T=T,
            D=D,
            W=W,
            BT=BT,
            BD=BD,
            stride_x_n=stride_x_n,
            stride_x_t=stride_x_t,
            stride_x_d=stride_x_d,
            stride_dx_n=stride_dx_n,
            stride_dx_t=stride_dx_t,
            stride_dx_d=stride_dx_d,
            stride_dy_n=stride_dy_n,
            stride_dy_t=stride_dy_t,
            stride_dy_d=stride_dy_d,
            NT=NT,
        )
        if use_fast_kernel:
            kernel_kwargs['weight'] = weight
            kernel_kwargs['dw'] = dw
            kernel_kwargs['db'] = db
        else:
            kernel_kwargs['y'] = y_pre if not use_swish_split else None
            kernel_kwargs['weight'] = weight
            kernel_kwargs['initial_state'] = initial_state
            kernel_kwargs['dht'] = dht
            kernel_kwargs['dw'] = dw
            kernel_kwargs['db'] = db
            kernel_kwargs['stride_y_n'] = stride_y_n
            kernel_kwargs['stride_y_t'] = stride_y_t
            kernel_kwargs['stride_y_d'] = stride_y_d

        for nt_off in range(0, NT, max_nt):
            nt_len = min(max_nt, NT - nt_off)
            grid = (triton.cdiv(D, BD), nt_len, B)
            if cu_seqlens is not None:
                kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
                kernel_kwargs['CHUNK_OFFSET'] = 0
                if not use_fast_kernel:
                    kernel_kwargs['dw'] = dw[nt_off:nt_off + nt_len] if weight is not None else None
                    kernel_kwargs['db'] = db[nt_off:nt_off + nt_len] if bias is not None else None
                elif weight is not None:
                    kernel_kwargs['dw'] = dw[nt_off:nt_off + nt_len]
                    kernel_kwargs['db'] = db[nt_off:nt_off + nt_len] if bias is not None else None
            else:
                kernel_kwargs['chunk_indices'] = chunk_indices
                kernel_kwargs['CHUNK_OFFSET'] = nt_off
                if not use_fast_kernel:
                    kernel_kwargs['dw'] = dw
                    kernel_kwargs['db'] = db
            bwd_fn[grid](**kernel_kwargs)

        if use_fast_kernel and (weight is not None or bias is not None):
            if weight is not None:
                dw = dw.sum(0, dtype=torch.float32)
            if bias is not None:
                db = db.sum(0, dtype=torch.float32)

    if weight is not None:
        dw = dw.to(weight) if use_fast_kernel else dw.sum(0).to(weight)
    if bias is not None:
        db = db.to(bias) if use_fast_kernel else db.sum(0).to(bias)

    dh0 = None
    if initial_state is not None:
        dh0 = compute_dh0_npu(
            dy=dy,
            y=y_pre,
            weight=weight,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
        )

    return dx.view(shape), dw, db, dr, dh0


def compute_dh0_npu(
    dy: torch.Tensor,
    y: torch.Tensor | None,
    weight: torch.Tensor,
    initial_state: torch.Tensor,
    activation: str | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    D, W = weight.shape
    N = initial_state.shape[0]
    T = dy.shape[1]

    BD = _dh0_bd(D, activation)
    dh0 = torch.zeros_like(initial_state)

    stride_dy_n = dy.stride(0)
    stride_dy_t = dy.stride(1)
    stride_dy_d = dy.stride(2) if dy.dim() == 3 else dy.stride(-1)
    stride_y_n = stride_y_t = stride_y_d = 0
    if y is not None:
        stride_y_n = y.stride(0)
        stride_y_t = y.stride(1)
        stride_y_d = y.stride(2) if y.dim() == 3 else y.stride(-1)

    max_n = _npu_max_axis_chunks(triton.cdiv(D, BD))
    kernel_kwargs = dict(
        dy=dy,
        y=y if activation in ('swish', 'silu') else None,
        weight=weight,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        stride_dy_n=stride_dy_n,
        stride_dy_t=stride_dy_t,
        stride_dy_d=stride_dy_d,
        stride_y_n=stride_y_n,
        stride_y_t=stride_y_t,
        stride_y_d=stride_y_d,
        T=T,
        D=D,
        W=W,
        BD=BD,
    )
    for n_off in range(0, N, max_n):
        n_len = min(max_n, N - n_off)
        kernel_kwargs['CHUNK_OFFSET'] = n_off
        compute_dh0_kernel[(triton.cdiv(D, BD), n_len)](**kernel_kwargs)
    return dh0


@input_guard(no_guard_contiguous=['x'])
def causal_conv1d_update_states_npu(
    x: torch.Tensor,
    state_len: int,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    layout_fallback: bool = False,
) -> torch.Tensor:
    del layout_fallback
    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        if x.dim() == 2:
            stride_x_n = 0
            stride_x_t, stride_x_d = x.stride()
            T = x.shape[0]
        else:
            stride_x_n = x.stride(0)
            stride_x_t, stride_x_d = x.stride(1), x.stride(2)
            T = x.shape[1]
        D = x.shape[-1]
    else:
        B, T, D = x.shape
        N = B
        stride_x_n, stride_x_t, stride_x_d = x.stride()

    W = state_len
    final_state = torch.empty(N, D, W, dtype=x.dtype, device=x.device)
    BD = _states_bd(D)
    BW = triton.next_power_of_2(W)
    grid_dim0 = triton.cdiv(D, BD)
    max_n = _npu_max_axis_chunks(grid_dim0)
    kernel_kwargs = dict(
        x=x,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
        W=W,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        BW=BW,
        BD=BD,
    )
    for n_off in range(0, N, max_n):
        n_len = min(max_n, N - n_off)
        kernel_kwargs['CHUNK_OFFSET'] = n_off
        causal_conv1d_states_fwd_kernel[(grid_dim0, n_len)](**kernel_kwargs)
    return final_state


@input_guard(no_guard_contiguous=['x'])
def causal_conv1d_update_npu(
    x: torch.Tensor,
    cache: torch.Tensor,
    residual: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = x.shape
    if weight is not None and x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')

    D = x.shape[-1]
    N = x.numel() // D
    W = weight.shape[1] if weight is not None else None
    BD = _update_bd(D)

    if x.dim() == 2:
        stride_x_n = x.stride(0)
        stride_x_d = x.stride(1)
    elif x.dim() == 3 and x.shape[0] == 1:
        stride_x_n = x.stride(1)
        stride_x_d = x.stride(2)
    elif x.dim() == 3:
        stride_x_n = x.stride(0)
        stride_x_d = x.stride(2)
    else:
        raise ValueError(f"Unsupported input shape: {x.shape}")

    y = torch.empty_like(x, memory_format=torch.contiguous_format)

    if y.dim() == 2:
        stride_y_n, stride_y_d = y.stride(0), y.stride(1)
    elif y.dim() == 3 and y.shape[0] == 1:
        stride_y_n, stride_y_d = y.stride(1), y.stride(2)
    elif y.dim() == 3:
        stride_y_n, stride_y_d = y.stride(0), y.stride(2)

    stride_res_n = stride_res_d = 0
    if residual is not None:
        if residual.dim() == 2:
            stride_res_n, stride_res_d = residual.stride(0), residual.stride(1)
        elif residual.dim() == 3 and residual.shape[0] == 1:
            stride_res_n, stride_res_d = residual.stride(1), residual.stride(2)
        elif residual.dim() == 3:
            stride_res_n, stride_res_d = residual.stride(0), residual.stride(2)

    grid_dim0 = triton.cdiv(D, BD)
    max_n = _npu_max_axis_chunks(grid_dim0)
    kernel_kwargs = dict(
        x=x,
        cache=cache,
        residual=residual,
        y=y,
        weight=weight,
        bias=bias,
        stride_x_n=stride_x_n,
        stride_x_d=stride_x_d,
        stride_y_n=stride_y_n,
        stride_y_d=stride_y_d,
        stride_res_n=stride_res_n,
        stride_res_d=stride_res_d,
        D=D,
        W=W,
        BD=BD,
        ACTIVATION=_activation_id(activation),
    )
    for n_off in range(0, N, max_n):
        n_len = min(max_n, N - n_off)
        kernel_kwargs['CHUNK_OFFSET'] = n_off
        causal_conv1d_update_kernel[(grid_dim0, n_len)](**kernel_kwargs)
    return y.view(shape), cache
