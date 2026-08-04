# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.


import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets


@triton.heuristics(
    {
        'USE_G': lambda args: args['g'] is not None,
        'USE_GK': lambda args: args['gk'] is not None,
        'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
        'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
        'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    }
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    T_all = T
    NT_all = NT
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    v_new_base = None
    if IS_VARLEN:
        v = v + (i_h * T_all + bos) * V
        k = k + (i_h * T_all + bos) * K
        w = w + (i_h * T_all + bos) * K
        g = g + i_h * T_all + bos
        h = h + (i_h * NT_all + boh) * K * V
        if SAVE_NEW_VALUE:
            v_new_base = v_new + (i_h * T_all + bos) * V
    else:
        v = v + (i_n * H + i_h) * T * V
        k = k + (i_n * H + i_h) * T * K
        w = w + (i_n * H + i_h) * T * K
        g = g + (i_n * H + i_h) * T
        h = h + (i_n * H + i_h) * NT * K * V
        if SAVE_NEW_VALUE:
            v_new_base = v_new + (i_n * H + i_h) * T * V

    if USE_INITIAL_STATE:
        h0_ptr = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht_ptr = ht + i_nh * K * V

    # Load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # Main recurrence over chunks
    for i_t in range(NT):
        # Store current hidden state h_t
        p_h1 = tl.make_block_ptr(h + i_t * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(h + i_t * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(h + i_t * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(h + i_t * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        # Compute v_residual = v - w @ h
        p_w = tl.make_block_ptr(w, (T, K), (K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))

        p_v = tl.make_block_ptr(v, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(v_new_base, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1

        # Apply output gate g
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)).to(tl.float32) < T
            b_g_last = tl.load(g + last_idx)
            p_g = tl.make_block_ptr(g, (T,), (1,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v *= (m_t * tl.exp(b_g_last - b_g))[:, None]
            b_g_last_exp = tl.exp(b_g_last)
            b_h1 *= b_g_last_exp
            if K > 64:
                b_h2 *= b_g_last_exp
            if K > 128:
                b_h3 *= b_g_last_exp
            if K > 192:
                b_h4 *= b_g_last_exp

        # Apply key gate gk
        if USE_GK:
            o_k1 = tl.arange(0, 64)  # Keep as int for pointer arithmetic
            gk_base_ptr = gk + (i_n * H + i_h) * T * K
            b_gk_last1 = tl.load(gk_base_ptr + last_idx * K + o_k1, mask=(o_k1 < K), other=0.0)
            b_h1 *= tl.exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk_base_ptr + last_idx * K + o_k2, mask=(o_k2 < K), other=0.0)
                b_h2 *= tl.exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk_base_ptr + last_idx * K + o_k3, mask=(o_k3 < K), other=0.0)
                b_h3 *= tl.exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk_base_ptr + last_idx * K + o_k4, mask=(o_k4 < K), other=0.0)
                b_h4 *= tl.exp(b_gk_last4)[:, None]

        b_v = b_v.to(k.dtype.element_ty)

        # Update hidden state: h += k @ v
        p_k = tl.make_block_ptr(k, (K, T), (1, K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_GK:
            p_gk = tl.make_block_ptr(gk_base_ptr, (K, T), (1, K), (0, i_t * BT), (64, BT), (0, 1))
            b_k = (b_k * tl.exp(b_gk_last1[:, None] - tl.load(p_gk, boundary_check=(0, 1)))).to(b_k.dtype)
        b_h1 += tl.dot(b_k, b_v)

        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, K), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                p_gk = tl.make_block_ptr(gk_base_ptr, (K, T), (1, K), (64, i_t * BT), (64, BT), (0, 1))
                b_k = (b_k * tl.exp(b_gk_last2[:, None] - tl.load(p_gk, boundary_check=(0, 1)))).to(b_k.dtype)
            b_h2 += tl.dot(b_k, b_v)

        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, K), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                p_gk = tl.make_block_ptr(gk_base_ptr, (K, T), (1, K), (128, i_t * BT), (64, BT), (0, 1))
                b_k = (b_k * tl.exp(b_gk_last3[:, None] - tl.load(p_gk, boundary_check=(0, 1)))).to(b_k.dtype)
            b_h3 += tl.dot(b_k, b_v)

        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, K), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                p_gk = tl.make_block_ptr(gk_base_ptr, (K, T), (1, K), (192, i_t * BT), (64, BT), (0, 1))
                b_k = (b_k * tl.exp(b_gk_last4[:, None] - tl.load(p_gk, boundary_check=(0, 1)))).to(b_k.dtype)
            b_h4 += tl.dot(b_k, b_v)

    # Store final state
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # default:64
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if initial_state is not None:
        raise ValueError("The chunk_gated_delta_rule_fwd_h does not support initial_state.")

    if gk is not None:
        raise ValueError("The chunk_gated_delta_rule_fwd_h does not support gk.")

    if not save_new_value:
        raise ValueError("The chunk_gated_delta_rule_fwd_h does not support save_new_value is False.")
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, H, NT, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    BV = 128

    v_new = torch.empty_like(u).permute(0, 2, 1, 3).contiguous() if save_new_value else None
    k = k.permute(0, 2, 1, 3).contiguous()
    w = w.permute(0, 2, 1, 3).contiguous()
    u = u.permute(0, 2, 1, 3).contiguous()
    g = g.permute(0, 2, 1).contiguous()
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[(triton.cdiv(V, BV), N * H)](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        NT=NT,
    )
    h = h.permute(0, 2, 1, 3, 4).contiguous()
    v_new = v_new.permute(0, 2, 1, 3).contiguous()
    return h, v_new, final_state


chunk_gated_delta_rule_fwd_h_arch32 = chunk_gated_delta_rule_fwd_h
