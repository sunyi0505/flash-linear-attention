# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl


@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    rstd,
    eps,
    T,
    D,
    BD: tl.constexpr,
    bt_size,
):
    i_t_start = tl.program_id(0)
    for offset in range(0, bt_size):
        i_t = i_t_start * bt_size + offset
        if i_t < T:
            cols = tl.arange(0, BD)
            mask = cols < D

            b_x = tl.load(x + i_t * D + cols, mask=mask, other=0.0).to(tl.float32)
            b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
            b_y = b_x * b_rstd
            tl.store(y + i_t * D + cols, b_y, mask=mask)
            tl.store(rstd + i_t, b_rstd)


@triton.jit
def l2norm_bwd_kernel1(
    y,
    rstd,
    dy,
    dx,
    eps,
    T,
    D,
    BD: tl.constexpr,
    bt_size,
):
    i_t_start = tl.program_id(0)
    for offset in range(0, bt_size):
        i_t = i_t_start * bt_size + offset
        if i_t < T:
            cols = tl.arange(0, BD)
            mask = cols < D
            b_y = tl.load(y + i_t * D + cols, mask=mask, other=0.0).to(tl.float32)
            b_rstd = tl.load(rstd + i_t).to(tl.float32)
            b_dy = tl.load(dy + i_t * D + cols, mask=mask, other=0.0).to(tl.float32)
            b_dx = b_dy * b_rstd - tl.sum(b_dy * b_y) * b_y * b_rstd
            tl.store(dx + i_t * D + cols, b_dx, mask=mask)


@triton.jit
def l2norm_fwd_kernel(
    x,
    y,
    rstd,
    eps,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
    bt_size,
):
    i_t = tl.program_id(0)
    for offset in range(0, bt_size):
        block_start = (i_t * bt_size + offset) * BT
        if block_start < T:
            p_x = tl.make_block_ptr(x, (T, D), (D, 1), (block_start, 0), (BT, BD), (1, 0))
            p_y = tl.make_block_ptr(y, (T, D), (D, 1), (block_start, 0), (BT, BD), (1, 0))
            p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (block_start,), (BT,), (0,))

            b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
            b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
            b_y = b_x * b_rstd[:, None]

            tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))


@triton.jit
def l2norm_bwd_kernel(
    y,
    rstd,
    dy,
    dx,
    eps,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
    bt_size,
):
    i_t_start = tl.program_id(0)
    num_blocks = bt_size

    total_i_t = (T + BT - 1) // BT
    base_tasks_per_block = total_i_t // num_blocks
    remainder_tasks = total_i_t % num_blocks

    if i_t_start < remainder_tasks:
        tasks_this_block = base_tasks_per_block + 1
        start_i_t = i_t_start * tasks_this_block
    else:
        tasks_this_block = base_tasks_per_block
        start_i_t = i_t_start * base_tasks_per_block + remainder_tasks

    for task_idx in range(tasks_this_block):
        i_t = start_i_t + task_idx
        block_start = i_t * BT
        if block_start < T:
            p_y = tl.make_block_ptr(y, (T, D), (D, 1), (block_start, 0), (BT, BD), (1, 0))
            p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (block_start,), (BT,), (0,))
            p_dy = tl.make_block_ptr(dy, (T, D), (D, 1), (block_start, 0), (BT, BD), (1, 0))
            p_dx = tl.make_block_ptr(dx, (T, D), (D, 1), (block_start, 0), (BT, BD), (1, 0))

            b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
            b_rstd = tl.load(p_rstd, boundary_check=(0,)).to(tl.float32)
            b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            b_dx = b_dy * b_rstd[:, None] - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rstd[:, None]
            tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))


# Host launchers migrated from mindspeed_ops/api/triton/l2norm.py (arch32 path)


def _compute_bt(bd):
    target = 8192
    bt = target // bd
    if bt < 1:
        bt = 1
    else:
        bt = 1 << (bt.bit_length() - 1)
    return max(1, min(bt, 256))


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    if D <= 8192:
        NB = triton.cdiv(T, 2048)
        bt_size = 32
        BT = _compute_bt(BD)

        def grid(meta):
            new_bt = meta['BT'] * bt_size
            return (triton.cdiv(T, new_bt),)

        l2norm_fwd_kernel[grid](
            x=x,
            y=y,
            rstd=rstd,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            NB=NB,
            BT=BT,
            bt_size=bt_size,
        )
    else:
        bt_size = 4
        l2norm_fwd_kernel1[(triton.cdiv(T, bt_size),)](
            x=x,
            y=y,
            rstd=rstd,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            bt_size=bt_size,
        )
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def l2norm_bwd(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor, eps: float = 1e-6):
    y_shape_og = y.shape
    y = y.view(-1, dy.shape[-1])
    dy = dy.view(-1, dy.shape[-1])
    assert dy.shape == y.shape
    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]
    MAX_FUSED_SIZE = 65536 // y.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if D <= 8192:
        NB = triton.cdiv(T, 2048)
        bt_size = 40
        BT = _compute_bt(BD)
        l2norm_bwd_kernel[(bt_size,)](
            y=y,
            rstd=rstd,
            dy=dy,
            dx=dx,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            NB=NB,
            BT=BT,
            bt_size=bt_size,
        )
    else:
        bt_size = 4
        l2norm_bwd_kernel1[(triton.cdiv(T, bt_size),)](
            y=y,
            rstd=rstd,
            dy=dy,
            dx=dx,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            bt_size=bt_size,
        )

    return dx.view(y_shape_og)
