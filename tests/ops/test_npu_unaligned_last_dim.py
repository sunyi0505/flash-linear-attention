# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""NPU regression tests for last-dim sizes that are not 32-byte aligned.

CANN leftover DMA can alias tail lanes when ``head_dim * element_size % 32 != 0``.
These tests compare against a reference with ``assert_aligned`` (absolute tolerance
only) rather than ``assert_close`` error-ratio gates.
"""

import os

import pytest
import torch
import torch.nn.functional as F
from einops import repeat

from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_fwd_o
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
from fla.ops.gdn2 import chunk_gdn2
from fla.ops.gdn2.naive import naive_recurrent_gdn2
from fla.ops.gla import chunk_gla, fused_recurrent_gla
from fla.ops.kda import chunk_kda
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import (
    IS_NPU,
    assert_aligned,
    device,
    last_dim_byte_aligned,
)
from tests.ops.test_gdn2 import _rand_inputs as gdn2_rand_inputs
from tests.ops.test_gdn_kernels import (
    _make_gate,
    chunk_bwd_dqkwg_ref,
    chunk_fwd_o_ref,
    gdn_bwd_autograd,
)

pytestmark = [
    pytest.mark.skipif(
        not IS_NPU,
        reason='Last-dim 32-byte alignment checks require Ascend NPU',
    ),
]

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')


def _require_unaligned(dim: int, dtype: torch.dtype) -> None:
    assert not last_dim_byte_aligned(dim, dtype), (
        f"expected last dim {dim} with {dtype} to be 32-byte unaligned"
    )


def _check_last_dim(name: str, ref: torch.Tensor, tri: torch.Tensor, dim: int, dtype: torch.dtype) -> None:
    _require_unaligned(dim, dtype)
    # Pass/fail is 2 IEEE-bin ULPs at the observed magnitude (not O(1)), so
    # leftover-lane aliasing fails while Cube MMA vs fp32-ref still passes.
    assert_aligned(
        name,
        ref,
        tri,
        last_dim=dim,
        cmp_dtype=dtype,
    )


def _check_state(name: str, ref: torch.Tensor, tri: torch.Tensor, dim: int, dtype: torch.dtype) -> None:
    _require_unaligned(dim, dtype)
    assert_aligned(
        name,
        ref[..., :dim, :dim],
        tri[..., :dim, :dim],
        cmp_dtype=torch.float32,
    )


def _check_full(name: str, ref: torch.Tensor, tri: torch.Tensor, *, dtype: torch.dtype | None = None) -> None:
    cmp = dtype or (tri.dtype if tri.dtype.is_floating_point else torch.float32)
    assert_aligned(name, ref, tri, cmp_dtype=cmp)


# ---------------------------------------------------------------------------
# GDN per-kernel (chunk_o is the primary leftover-DMA surface)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'use_g', 'dtype'),
    [
        pytest.param(1, 64, 2, 2, 60, True, torch.float16, id='chunk_fwd_o-D60-fp16'),
        pytest.param(1, 64, 2, 2, 20, True, torch.bfloat16, id='chunk_fwd_o-D20-bf16'),
    ],
)
def test_gdn_kernel_chunk_fwd_o(B, T, H, HV, D, use_g, dtype):
    torch.manual_seed(42)
    BT = 64
    scale = D ** -0.5
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    h = torch.randn(B, T // BT, HV, D, D, dtype=dtype, device=device)
    g = torch.randn(B, T, HV, dtype=torch.float32, device=device) * 0.1 if use_g else None

    ref = chunk_fwd_o_ref(q, k, v, h, g, scale, BT)
    tri = chunk_fwd_o(q=q, k=k, v=v, h=h, g=g, scale=scale, chunk_size=BT)
    _check_last_dim('o', ref, tri, D, dtype)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'use_g', 'use_w', 'dtype'),
    [
        pytest.param(1, 64, 2, 2, 60, True, True, torch.float16, id='chunk_bwd_dqkwg-D60-fp16'),
        pytest.param(1, 64, 2, 2, 20, True, True, torch.bfloat16, id='chunk_bwd_dqkwg-D20-bf16'),
    ],
)
def test_gdn_kernel_chunk_bwd_dqkwg(B, T, H, HV, D, use_g, use_w, dtype):
    torch.manual_seed(42)
    BT = 64
    NT = T // BT
    scale = D ** -0.5
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v_new = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    do = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    h = torch.randn(B, NT, HV, D, D, dtype=dtype, device=device)
    dh = torch.randn(B, NT, HV, D, D, dtype=dtype, device=device)
    w = torch.randn(B, T, HV, D, dtype=dtype, device=device) if use_w else None
    dv = torch.randn(B, T, HV, D, dtype=dtype, device=device) if use_w else None
    g = torch.randn(B, T, HV, dtype=torch.float32, device=device) * 0.1 if use_g else None

    dq_ref, dk_ref, dw_ref, dg_ref = chunk_bwd_dqkwg_ref(
        q, k, v_new, do, h, dh, w, dv, g, scale, BT,
    )
    dq_tri, dk_tri, dw_tri, dg_tri = chunk_bwd_dqkwg(
        q=q, k=k, v=v_new, do=do, h=h, dh=dh, w=w, dv=dv, g=g, scale=scale, chunk_size=BT,
    )
    _check_last_dim('dq', dq_ref, dq_tri, D, dtype)
    _check_last_dim('dk', dk_ref, dk_tri, D, dtype)
    if use_w:
        _check_last_dim('dw', dw_ref, dw_tri, D, dtype)
    if use_g:
        assert dg_ref is not None and dg_tri is not None
        _check_full('dg', dg_ref, dg_tri, dtype=torch.float32)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'use_h0', 'dtype'),
    [
        pytest.param(1, 64, 2, 2, 60, True, torch.float16, id='gdn_full_bwd-D60-fp16'),
        pytest.param(1, 64, 2, 2, 20, True, torch.bfloat16, id='gdn_full_bwd-D20-bf16'),
    ],
)
def test_gdn_kernel_full_bwd(B, T, H, HV, D, use_h0, dtype):
    torch.manual_seed(42)
    scale = D ** -0.5
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()
    g = _make_gate(B, T, HV)
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32, device=device) if use_h0 else None
    do = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    dht = torch.randn(B, HV, D, D, dtype=torch.float32, device=device)

    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    ref = gdn_bwd_autograd(q, k, v, beta, g, h0, do, dht, scale)

    q_t = q.detach().requires_grad_()
    k_t = k.detach().requires_grad_()
    v_t = v.detach().requires_grad_()
    beta_t = beta.detach().requires_grad_()
    g_t = g.detach().requires_grad_()
    h0_t = h0.detach().requires_grad_() if h0 is not None else None
    o, final_state = chunk_gated_delta_rule(
        q_t, k_t, v_t, g_t, beta_t, scale, h0_t, output_final_state=True,
    )
    (do * o).sum().add((dht * final_state).sum()).backward()

    _check_last_dim('dq', ref['dq'], q_t.grad, D, dtype)
    _check_last_dim('dk', ref['dk'], k_t.grad, D, dtype)
    _check_last_dim('dv', ref['dv'], v_t.grad, D, dtype)
    _check_full('dbeta', ref['dbeta'], beta_t.grad, dtype=dtype)
    _check_full('dg', ref['dg'], g_t.grad, dtype=torch.float32)
    if use_h0:
        _check_state('dh0', ref['dh0'], h0_t.grad, D, torch.float32)


# ---------------------------------------------------------------------------
# GDN end-to-end
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'dtype'),
    [
        pytest.param(1, 64, 2, 2, 20, torch.float16, id='gdn_chunk-D20-fp16'),
        pytest.param(1, 64, 2, 2, 60, torch.bfloat16, id='gdn_chunk-D60-bf16'),
    ],
)
def test_gdn_chunk(B, T, H, HV, D, dtype):
    torch.manual_seed(42)
    scale = 1.0
    G = HV // H
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, HV, D, dtype=dtype)
    beta = torch.rand(B, T, HV, dtype=torch.float).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32))
    h0 = torch.zeros(B, HV, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, g, h0))

    tri, tri_ht = chunk_gated_delta_rule(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(), g=g.clone(), beta=beta.clone(),
        scale=scale, initial_state=h0.clone(), output_final_state=True,
    )
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = (
        q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad,
    )
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref, ref_ht = naive_recurrent_gated_delta_rule(
        q=F.normalize(repeat(q.clone(), 'b t h d -> b t (h g) d', g=G), p=2, dim=-1),
        k=F.normalize(repeat(k.clone(), 'b t h d -> b t (h g) d', g=G), p=2, dim=-1),
        v=v.clone(), beta=beta.clone(), g=g.clone(), scale=scale,
        output_final_state=True, initial_state=h0.clone(),
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = (
        q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad,
    )

    _check_last_dim('o', ref, tri, D, dtype)
    _check_state('ht', ref_ht, tri_ht, D, torch.float32)
    _check_last_dim('dq', ref_dq, tri_dq, D, dtype)
    _check_last_dim('dk', ref_dk, tri_dk, D, dtype)
    _check_last_dim('dv', ref_dv, tri_dv, D, dtype)
    _check_full('db', ref_dbeta, tri_dbeta, dtype=dtype)
    _check_full('dg', ref_dg, tri_dg, dtype=torch.float32)
    _check_state('dh0', ref_dh0, tri_dh0, D, torch.float32)


# ---------------------------------------------------------------------------
# GDN-2 end-to-end
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'dtype'),
    [
        pytest.param(1, 64, 2, 60, 60, torch.float16, id='gdn2_chunk-K60-V60-fp16'),
        pytest.param(1, 64, 2, 20, 40, torch.bfloat16, id='gdn2_chunk-K20-V40-bf16'),
    ],
)
def test_gdn2_chunk(B, T, H, K, V, dtype):
    q, k, v, g, b, w, _, _ = gdn2_rand_inputs(B, T, H, H, K, V, dtype)
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device)
    for t in (q, k, v, g, b, w, h0):
        t.requires_grad_(True)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    qn = F.normalize(q.float(), p=2, dim=-1).to(dtype)
    kn = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    ref, ref_ht = naive_recurrent_gdn2(
        q=qn, k=kn, v=v, g=g, b=b, w=w, scale=1.0,
        initial_state=h0, output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_grads = {n: t.grad.clone() for n, t in zip(('q', 'k', 'v', 'g', 'b', 'w', 'h0'), (q, k, v, g, b, w, h0))}
    for t in (q, k, v, g, b, w, h0):
        t.grad = None

    tri, tri_ht = chunk_gdn2(
        q=qn, k=kn, v=v, g=g, b=b, w=w, scale=1.0,
        initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_grads = {n: t.grad.clone() for n, t in zip(('q', 'k', 'v', 'g', 'b', 'w', 'h0'), (q, k, v, g, b, w, h0))}

    _check_last_dim('o', ref, tri, V, dtype)
    _check_state('ht', ref_ht, tri_ht, K, torch.float32)
    _check_last_dim('q', ref_grads['q'], tri_grads['q'], K, dtype)
    _check_last_dim('k', ref_grads['k'], tri_grads['k'], K, dtype)
    _check_last_dim('v', ref_grads['v'], tri_grads['v'], V, dtype)
    _check_last_dim('g', ref_grads['g'], tri_grads['g'], K, dtype)
    _check_last_dim('b', ref_grads['b'], tri_grads['b'], K, dtype)
    _check_last_dim('w', ref_grads['w'], tri_grads['w'], V, dtype)
    _check_state('h0', ref_grads['h0'], tri_grads['h0'], K, torch.float32)


# ---------------------------------------------------------------------------
# KDA / GLA end-to-end
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'dtype'),
    [
        pytest.param(1, 64, 2, 2, 20, torch.float16, id='kda_chunk-D20-fp16'),
        pytest.param(1, 64, 2, 2, 60, torch.bfloat16, id='kda_chunk-D60-bf16'),
    ],
)
def test_kda_chunk(B, T, H, HV, D, dtype):
    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, HV, D, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, HV, D, dtype=torch.float))
    beta = torch.randn(B, T, HV, dtype=dtype).sigmoid()
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32)
    q, k, v, g, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, g, beta, h0))
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    ref, ref_ht = naive_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(), g=g.clone(), beta=beta.clone(), scale=1.0,
        initial_state=h0.clone(), output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None

    tri, tri_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(), g=g.clone(), beta=beta.clone(), scale=1.0,
        initial_state=h0.clone(), output_final_state=True, safe_gate=True,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dg, tri_db, tri_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad

    _check_last_dim('o', ref, tri, D, dtype)
    _check_state('ht', ref_ht, tri_ht, D, torch.float32)
    _check_last_dim('dq', ref_dq, tri_dq, D, dtype)
    _check_last_dim('dk', ref_dk, tri_dk, D, dtype)
    _check_last_dim('dv', ref_dv, tri_dv, D, dtype)
    _check_last_dim('dg', ref_dg, tri_dg, D, dtype)
    _check_full('db', ref_db, tri_db, dtype=dtype)
    _check_state('dh0', ref_dh0, tri_dh0, D, torch.float32)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(1, 64, 2, 20, torch.float16, id='gla_chunk-D20-fp16'),
        pytest.param(1, 64, 2, 60, torch.bfloat16, id='gla_chunk-D60-bf16'),
    ],
)
def test_gla_chunk(B, T, H, D, dtype):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    q = torch.rand((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.rand((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.rand((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    g = F.logsigmoid(torch.rand((B, T, H, D), dtype=dtype, device=device)).requires_grad_()
    h0 = torch.rand((B, H, D, D), dtype=torch.float32, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn((B, H, D, D), dtype=torch.float32, device=device)

    tri, tri_ht = chunk_gla(q=q, k=k, v=v, g=g, initial_state=h0, output_final_state=True)
    ((tri * do).sum() + (tri_ht * dht).sum().to(do.dtype)).backward()
    tri_dq, tri_dk, tri_dv, tri_dg, tri_dh0 = q.grad.clone(), k.grad.clone(), v.grad.clone(), g.grad.clone(), h0.grad.clone()
    q.grad = k.grad = v.grad = g.grad = h0.grad = None

    ref, ref_ht = fused_recurrent_gla(q=q, k=k, v=v, gk=g, initial_state=h0, output_final_state=True)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, ref_dk, ref_dv, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, g.grad, h0.grad

    _check_last_dim('o', ref, tri, D, dtype)
    _check_state('ht', ref_ht, tri_ht, D, torch.float32)
    _check_last_dim('dq', ref_dq, tri_dq, D, dtype)
    _check_last_dim('dk', ref_dk, tri_dk, D, dtype)
    _check_last_dim('dv', ref_dv, tri_dv, D, dtype)
    _check_last_dim('dg', ref_dg, tri_dg, D, dtype)
    _check_state('dh0', ref_dh0, tri_dh0, D, torch.float32)
