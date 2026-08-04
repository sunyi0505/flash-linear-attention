# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.


import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets


# pylint: disable=too-many-nested-blocks
@triton.heuristics(
    {
        'USE_G': lambda args: args['g'] is not None,
        'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    }
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    N: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    T_max = T
    for i_v in range(tl.cdiv(V, BV)):
        for i_n in range(N):
            if IS_VARLEN:
                bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
                T = eos - bos
                NT = tl.cdiv(T, BT)
                boh = tl.load(chunk_offsets + i_n).to(tl.int64)
            else:
                bos, eos = i_n * T, i_n * T + T
                NT = tl.cdiv(T, BT)
                boh = i_n * NT

            core_id = tl.program_id(0)
            total_cores = tl.num_programs(0)
            base_chunks_per_pid = NT // total_cores
            remainder = NT % total_cores

            if core_id < remainder:
                chunks_this_pid = base_chunks_per_pid + 1
                start_idx = core_id * chunks_this_pid
            else:
                chunks_this_pid = base_chunks_per_pid
                start_idx = core_id * base_chunks_per_pid + remainder

            # offset calculation
            for i_h in range(0, H):
                q_offset = (bos * Hg + i_h // (H // Hg)) * K
                k_offset = (bos * Hg + i_h // (H // Hg)) * K
                v_offset = (bos * H + i_h) * V
                o_offset = (bos * H + i_h) * V

                for i_t in range(start_idx, start_idx + chunks_this_pid):
                    i_tg = boh + i_t
                    h_base = h + (i_tg * H + i_h).to(tl.int64) * K * V
                    b_o = tl.zeros([BT, BV], dtype=tl.float32)
                    b_A = tl.zeros([BT, BT], dtype=tl.float32)
                    for i_k in range(tl.cdiv(K, BK)):
                        p_q = tl.make_block_ptr(
                            q + q_offset, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
                        )
                        p_k = tl.make_block_ptr(
                            k + k_offset, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
                        )
                        p_h = tl.make_block_ptr(h_base, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
                        b_q = tl.load(p_q, boundary_check=(0, 1))
                        b_k = tl.load(p_k, boundary_check=(0, 1))
                        b_h = tl.load(p_h, boundary_check=(0, 1))

                        # [BT, BK] @ [BK, BV] -> [BT, BV]
                        b_o += tl.dot(b_q, b_h)
                        # [BT, BK] @ [BK, BT] -> [BT, BT]
                        b_A += tl.dot(b_q, b_k)

                    if USE_G:
                        if IS_VARLEN:
                            p_g = tl.make_block_ptr(g + bos + i_h * T_max, (T,), (1,), (i_t * BT,), (BT,), (0,))
                        else:
                            p_g = tl.make_block_ptr(g + bos * H + i_h * T_max, (T,), (1,), (i_t * BT,), (BT,), (0,))
                        b_g = tl.load(p_g, boundary_check=(0,))
                        b_o = b_o * tl.exp(b_g)[:, None]
                        b_A = b_A * tl.exp(b_g[:, None] - b_g[None, :])
                    if USE_G_GAMMA:
                        b_gamma = tl.load(g_gamma + i_h)
                        b_g = b_gamma * (tl.arange(0, BT) + 1)
                        b_o = b_o * tl.exp(b_g)[:, None]
                        b_A = b_A * tl.exp(b_g[:, None] - b_g[None, :])

                    o_i = tl.arange(0, BT)
                    m_A = o_i[:, None] >= o_i[None, :]
                    b_A = tl.where(m_A, b_A, 0)

                    p_v = tl.make_block_ptr(v + v_offset, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                    p_o = tl.make_block_ptr(o + o_offset, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                    b_v = tl.load(p_v, boundary_check=(0, 1))

                    # to fix mma -> mma layout conversion
                    # already solved by triton v3.2 or higher
                    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
                    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    if cu_seqlens is None:
        N, chunk_offsets = B, None
    else:
        N, chunk_offsets = (
            len(cu_seqlens) - 1,
            prepare_chunk_offsets(cu_seqlens, BT),
        )

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, N * H)

    g = g.transpose(1, 2).contiguous()
    h = h.contiguous()
    CV_kernel_num = 24
    chunk_fwd_kernel_o[(CV_kernel_num,)](
        q,
        k,
        v,
        h,
        g,
        g_gamma,
        o,
        cu_seqlens,
        chunk_offsets,
        scale,
        T=T,
        H=H,
        N=N,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=128,
        BV=128,
        num_warps=4,
        num_stages=2,
    )
    return o
