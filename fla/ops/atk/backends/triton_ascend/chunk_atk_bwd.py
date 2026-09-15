# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""PyTorch ATK chunk backward for Ascend.

The fused Triton chunk_out bwd hangs on triton-ascend (too many live UB
tiles around ``tl.dot`` + squash). Forward already ran on NPU; this path
mirrors that math in PyTorch so backward is hang-free and matches CUDA.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_NPU_CHUNK_LEN = 16


def _pad_time(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad == 0:
        return x
    # pad the time axis (dim 1); F.pad applies from the last dim backward
    return F.pad(x, (0, 0) * (x.ndim - 2) + (0, pad))


def _atk_bwd_chunk_out_torch_one(
    k: torch.Tensor,
    log_g: torch.Tensor,
    beta: torch.Tensor,
    ac: torch.Tensor,
    h0: torch.Tensor | None,
    g_kp: torch.Tensor,
    gk: torch.Tensor,
    gg: torch.Tensor,
    gbeta: torch.Tensor,
    gac_prev: torch.Tensor,
    dh0: torch.Tensor | None,
    g_log_atk_scale_chunks: torch.Tensor,
    log_atk_scale: torch.Tensor,
    logx: float,
    eps: float,
    chunk_len: int,
    seq_index: int = 0,
):
    """In-place chunk_out backward for a single packed sequence batch.

    Tensors are ``[B, T, H, ...]`` / ``ac, gac_prev [B, C, H, D]``.
    ``seq_index`` selects the row written in ``g_log_atk_scale_chunks``.
    """
    B, T, H, D = k.shape
    C = chunk_len
    n_chunks = math.ceil(T / C)
    pad = n_chunks * C - T
    idx = torch.arange(C, device=k.device)

    k_p = _pad_time(k.float(), pad).view(B, n_chunks, C, H, D)
    gkp_p = _pad_time(g_kp.float(), pad).view(B, n_chunks, C, H, D)
    log_g_p = _pad_time(log_g.float(), pad).view(B, n_chunks, C, H)
    beta_p = _pad_time(beta.float(), pad).view(B, n_chunks, C, H)
    token_mask = (idx.view(1, 1, C, 1) + (torch.arange(n_chunks, device=k.device).view(1, n_chunks, 1, 1) * C)) < T
    token_mask = token_mask.expand(B, n_chunks, C, H)

    g_val = torch.exp(log_g_p)
    la_cs = log_g_p.cumsum(dim=2)
    la_cs_roll = torch.zeros_like(la_cs)
    la_cs_roll[:, :, 1:] = la_cs[:, :, :-1]
    base_decays = torch.exp(la_cs_roll * (idx > 0).to(la_cs_roll.dtype).view(1, 1, C, 1))

    causal = idx[:, None] > idx[None, :]
    M = torch.exp(la_cs_roll[:, :, :, None, :] - la_cs[:, :, None, :, :])
    M = M.masked_fill(~causal.view(1, 1, C, C, 1), 0.0)

    U = beta_p.unsqueeze(-1) * k_p.square()

    ac_prev = torch.zeros(B, n_chunks, H, D, device=k.device, dtype=torch.float32)
    if n_chunks > 1:
        ac_prev[:, 1:] = ac[:, :n_chunks - 1].float()
    if h0 is not None:
        ac_prev[:, 0] = h0.float()

    raw_state = base_decays.unsqueeze(-1) * ac_prev.unsqueeze(2) + torch.einsum('bntsh,bnshd->bnthd', M, U)
    A_t = g_val.unsqueeze(-1) * raw_state + U

    center = log_atk_scale.float().view(1, 1, 1, H, 1)
    ell = torch.log(A_t + eps)
    r = ell - center
    abs_r = r.abs()
    one_plus = 1.0 + abs_r
    s = r / one_plus
    ds_dr = 1.0 / (one_plus * one_plus)
    M_mult = torch.exp(-logx * s)

    mask_f = token_mask.unsqueeze(-1).to(A_t.dtype)
    gk_val = gkp_p * M_mult
    gs = (gkp_p * k_p) * (-logx * M_mult)
    gr_local = gs * ds_dr * mask_f
    gA_t = gr_local / (A_t + eps)
    graw = gA_t * g_val.unsqueeze(-1)
    gg_val = (gA_t * raw_state).sum(-1) * g_val
    gU = gA_t
    gk_val = gk_val + 2.0 * k_p * gU * beta_p.unsqueeze(-1)
    gbeta_val = (gU * k_p.square()).sum(-1)
    gbase = (graw * ac_prev.unsqueeze(2)).sum(-1)

    # (M.T @ graw)[s] = sum_t M[t,s] * graw[t]
    gU_from_M = torch.einsum('bntsh,bnthd->bnshd', M, graw)
    gk_val = gk_val + 2.0 * k_p * gU_from_M * beta_p.unsqueeze(-1)
    gbeta_val = gbeta_val + (gU_from_M * k_p.square()).sum(-1)

    gM = torch.einsum('bnthd,bnshd->bntsh', graw, U)
    gM = gM.masked_fill(~causal.view(1, 1, C, C, 1), 0.0)
    ginner = M * gM
    gbase = gbase * (idx > 0).to(gbase.dtype).view(1, 1, C, 1)
    gla_cs_roll = gbase * base_decays + ginner.sum(dim=3)
    gla_cs = -ginner.sum(dim=2)
    gla_cs = gla_cs.clone()
    gla_cs[:, :, :-1] = gla_cs[:, :, :-1] + gla_cs_roll[:, :, 1:]
    gg_val = gg_val + torch.cumsum(gla_cs.flip(2), dim=2).flip(2)

    gac_tile = (graw * base_decays.unsqueeze(-1)).sum(dim=2)  # [B, Nc, H, D]
    if n_chunks > 1:
        gac_prev[:, :n_chunks - 1] = gac_tile[:, 1:]
    if dh0 is not None:
        dh0.add_(gac_tile[:, 0])

    g_center_b = -gr_local.sum(dim=(2, 4))  # [B, Nc, H]
    n_store = min(n_chunks, g_log_atk_scale_chunks.shape[1])
    g_log_atk_scale_chunks[seq_index:seq_index + B, :n_store] = g_center_b[:, :n_store]

    gk_out = (gk_val * mask_f).reshape(B, n_chunks * C, H, D)[:, :T]
    gg_out = (gg_val * token_mask.to(gg_val.dtype)).reshape(B, n_chunks * C, H)[:, :T]
    gbeta_out = (gbeta_val * token_mask.to(gbeta_val.dtype)).reshape(B, n_chunks * C, H)[:, :T]
    gk.copy_(gk_out)
    gg.copy_(gg_out)
    gbeta.copy_(gbeta_out)


def atk_backward_pass_chunks_torch(
    a: torch.Tensor,
    sa: torch.Tensor,
    ac: torch.Tensor,
    h0: torch.Tensor | None,
    gac_from_out: torch.Tensor,
    ga: torch.Tensor,
    gsa: torch.Tensor,
    dh0: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_len: int,
    T: int,
):
    del a, T
    n, n_chunks, h, d = ac.shape
    if cu_seqlens is None:
        seq_chunks = [n_chunks] * n
    else:
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        seq_chunks = [int(math.ceil(int(s) / chunk_len)) for s in seqlens]

    for i in range(n):
        nc = seq_chunks[i]
        if nc <= 0:
            continue
        carry = torch.zeros(h, d, device=ac.device, dtype=torch.float32)
        for c in range(nc - 1, -1, -1):
            carry = carry + gac_from_out[i, c]
            ga[i, c] = carry
            decay = torch.exp(sa[i, c])
            if c > 0:
                gsa[i, c] = (carry * ac[i, c - 1]).sum(-1) * decay
            elif h0 is not None:
                gsa[i, c] = (carry * h0[i].float()).sum(-1) * decay
            else:
                gsa[i, c] = 0
            carry = carry * decay.unsqueeze(-1)
        if dh0 is not None:
            dh0[i].add_(carry)


def atk_backward_chunk_summary_torch(
    k: torch.Tensor,
    log_g: torch.Tensor,
    beta: torch.Tensor,
    ga: torch.Tensor,
    gsa: torch.Tensor,
    gk: torch.Tensor,
    gg: torch.Tensor,
    gbeta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_len: int,
):
    """Per-chunk summary backward. Avoids NPU atomics / reverse-cumsum hangs."""

    def _one(k_s, log_g_s, beta_s, ga_s, gsa_s, gk_s, gg_s, gbeta_s):
        B, T, H, D = k_s.shape
        C = chunk_len
        n_chunks = math.ceil(T / C)
        pad = n_chunks * C - T
        k_p = _pad_time(k_s.float(), pad).view(B, n_chunks, C, H, D)
        log_g_p = _pad_time(log_g_s.float(), pad).view(B, n_chunks, C, H)
        beta_p = _pad_time(beta_s.float(), pad).view(B, n_chunks, C, H)
        nc = min(n_chunks, ga_s.shape[1])
        ga_p = ga_s[:, :nc].float()
        gsa_p = gsa_s[:, :nc].float()
        sa_val = log_g_p.sum(dim=2)
        decays = torch.exp(sa_val.unsqueeze(2) - log_g_p.cumsum(dim=2))
        U = beta_p.unsqueeze(-1) * k_p.square()
        gdecays = (ga_p.unsqueeze(2) * U).sum(-1)
        gU = ga_p.unsqueeze(2) * decays.unsqueeze(-1)
        gbeta_d = (gU * k_p.square()).sum(-1)
        gk_d = 2.0 * k_p * gU * beta_p.unsqueeze(-1)
        gdecays_exp = decays * gdecays
        glog = -torch.cumsum(gdecays_exp.flip(2), dim=2).flip(2) + (gsa_p + gdecays_exp.sum(dim=2)).unsqueeze(2)
        idx = torch.arange(C, device=k_s.device)
        token_mask = (
            idx.view(1, 1, C) + torch.arange(n_chunks, device=k_s.device).view(1, n_chunks, 1) * C
        ) < T
        token_mask = token_mask.unsqueeze(-1).expand(B, n_chunks, C, H)
        gk_s.add_(
            (gk_d * token_mask.unsqueeze(-1).to(gk_d.dtype)).reshape(B, n_chunks * C, H, D)[:, :T]
        )
        gg_s.add_((glog * token_mask.to(glog.dtype)).reshape(B, n_chunks * C, H)[:, :T])
        gbeta_s.add_((gbeta_d * token_mask.to(gbeta_d.dtype)).reshape(B, n_chunks * C, H)[:, :T])

    if cu_seqlens is None:
        _one(k, log_g, beta, ga, gsa, gk, gg, gbeta)
        return
    n = len(cu_seqlens) - 1
    for i in range(n):
        bos, eos = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        if eos <= bos:
            continue
        sl = slice(bos, eos)
        _one(
            k[:, sl], log_g[:, sl], beta[:, sl],
            ga[i:i + 1], gsa[i:i + 1],
            gk[:, sl], gg[:, sl], gbeta[:, sl],
        )


def _atk_backward_chunk_out_npu(
    k, g_raw, beta, ac, initial_A_state, dk_precond,
    gk, gg, gbeta, gac_prev, dh0, g_log_atk_scale_chunks,
    cu_seqlens, log_atk_scale, logx, eps, CHUNK_LEN,
):
    if cu_seqlens is None:
        _atk_bwd_chunk_out_torch_one(
            k, g_raw, beta, ac, initial_A_state, dk_precond,
            gk, gg, gbeta, gac_prev, dh0, g_log_atk_scale_chunks,
            log_atk_scale, logx, eps, CHUNK_LEN, seq_index=0,
        )
        return

    n = len(cu_seqlens) - 1
    for i in range(n):
        bos = int(cu_seqlens[i])
        eos = int(cu_seqlens[i + 1])
        if eos <= bos:
            continue
        sl = slice(bos, eos)
        h0_i = None if initial_A_state is None else initial_A_state[i:i + 1]
        dh0_i = None if dh0 is None else dh0[i:i + 1]
        _atk_bwd_chunk_out_torch_one(
            k[:, sl], g_raw[:, sl], beta[:, sl], ac[i:i + 1], h0_i, dk_precond[:, sl],
            gk[:, sl], gg[:, sl], gbeta[:, sl], gac_prev[i:i + 1], dh0_i,
            g_log_atk_scale_chunks, log_atk_scale, logx, eps, CHUNK_LEN, seq_index=i,
        )


def chunk_atk_bwd_npu(
    k: torch.Tensor,
    g_raw: torch.Tensor,
    beta: torch.Tensor,
    dk_precond: torch.Tensor,
    ac: torch.Tensor,
    a: torch.Tensor,
    sa: torch.Tensor,
    chunk_size: int = 64,
    initial_A_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
    dat: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    del chunk_size
    B, T, H, K = k.shape
    CHUNK_LEN = _NPU_CHUNK_LEN

    is_varlen = cu_seqlens is not None
    NT = math.ceil(T / CHUNK_LEN)
    N = len(cu_seqlens) - 1 if is_varlen else B

    k = k.contiguous()
    g_raw = g_raw.contiguous()
    beta = beta.contiguous()
    dk_precond = dk_precond.contiguous()

    if log_atk_scale is None:
        log_atk_scale = torch.full((H,), -0.2, dtype=torch.float32, device=k.device)
    else:
        log_atk_scale = log_atk_scale.contiguous()

    logx = math.log(x) if x > 0 else 0.0

    gk = torch.zeros_like(k, dtype=torch.float32)
    g_log_atk_scale_chunks = torch.zeros(N, ac.shape[1], H, device=k.device, dtype=torch.float32)
    gg = torch.zeros_like(g_raw, dtype=torch.float32)
    gbeta = torch.zeros_like(beta, dtype=torch.float32)
    gac_prev = torch.zeros_like(ac, dtype=torch.float32)
    if dat is not None:
        # The final ATK state `at` equals `ac[:, last_chunk]`, so its incoming
        # gradient enters the reverse scan as the carry at the final chunk;
        # `gac_prev[:, last_chunk]` is never written by the local-backward kernel.
        if cu_seqlens is None:
            gac_prev[:, NT - 1] += dat
        else:
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
            last_chunk = torch.clamp((seqlens + CHUNK_LEN - 1) // CHUNK_LEN - 1, min=0)
            valid = seqlens > 0
            idx = torch.arange(N, device=k.device)
            gac_prev[idx[valid], last_chunk[valid]] += dat[valid]
    dh0 = torch.zeros(N, H, K, device=k.device, dtype=torch.float32) if initial_A_state is not None else None
    ga = torch.zeros_like(a, dtype=torch.float32)
    gsa = torch.zeros_like(sa, dtype=torch.float32)

    _atk_backward_chunk_out_npu(
        k, g_raw, beta, ac, initial_A_state, dk_precond,
        gk, gg, gbeta, gac_prev, dh0, g_log_atk_scale_chunks,
        cu_seqlens, log_atk_scale, logx, eps, CHUNK_LEN,
    )
    atk_backward_pass_chunks_torch(
        a, sa, ac, initial_A_state, gac_prev, ga, gsa, dh0,
        cu_seqlens, CHUNK_LEN, T,
    )
    atk_backward_chunk_summary_torch(
        k, g_raw, beta, ga, gsa, gk, gg, gbeta, cu_seqlens, CHUNK_LEN,
    )

    g_log_atk_scale = g_log_atk_scale_chunks.sum(dim=(0, 1))
    return gk, gbeta, gg, g_log_atk_scale, dh0
