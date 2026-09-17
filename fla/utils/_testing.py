# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import logging
import math
import warnings

import torch

from ._config import FLA_CI_ENV

logger = logging.getLogger(__name__)


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def last_dim_byte_aligned(n: int, dtype: torch.dtype) -> bool:
    """True when ``n * element_size`` is a multiple of 32 (CANN DMA tail rule)."""
    return n * torch.empty((), dtype=dtype).element_size() % 32 == 0


def aligned_atol_for_dtype(dtype: torch.dtype) -> float:
    """One representable step at O(1) for ``dtype``; used by ``assert_aligned``."""
    if dtype == torch.float16:
        return 2 ** -10
    if dtype == torch.bfloat16:
        return 2 ** -8
    if dtype == torch.float32:
        return 1e-5
    finfo = torch.finfo(dtype)
    return max(finfo.resolution, finfo.eps)


def _ieee_ulp(scale: float, dtype: torch.dtype) -> float:
    """ULP of the IEEE bin containing ``scale`` (constant between powers of two)."""
    finfo = torch.finfo(dtype)
    scale = max(float(scale), float(finfo.tiny))
    return finfo.eps * (2 ** math.floor(math.log2(scale)))


def _default_aligned_atol(ref: torch.Tensor, tri: torch.Tensor, cmp_dtype: torch.dtype) -> float:
    ref_f = ref.to(cmp_dtype).float()
    tri_f = tri.to(cmp_dtype).float()
    scale = torch.maximum(ref_f.abs(), tri_f.abs()).max().clamp(min=1e-3).item()
    # fp32 buffers from fp16/bf16 kernels (dg, ht, dh0) cannot beat bf16 ULP
    # at the observed magnitude; leftover-lane aliasing is still far larger.
    if cmp_dtype == torch.float32:
        return max(aligned_atol_for_dtype(torch.float32), 2 * scale * aligned_atol_for_dtype(torch.bfloat16))
    # ``scale * eps`` undershoots the IEEE bin (and Cube MMA is ~2 ULPs there,
    # same as D=64 aligned). Leftover aliasing is still O(1).
    return max(aligned_atol_for_dtype(cmp_dtype), 2 * _ieee_ulp(scale, cmp_dtype))


def assert_aligned(
    prefix: str,
    ref: torch.Tensor,
    tri: torch.Tensor,
    *,
    last_dim: int | None = None,
    atol: float | None = None,
    cmp_dtype: torch.dtype | None = None,
) -> None:
    """Strict elementwise check for NPU last-dim-unaligned regression tests.

    ``ref`` is cast to ``cmp_dtype`` (default: ``tri.dtype``) before compare.
    Pass/fail is only ``max |ref - tri| <= atol`` — never error ratio or CI soft-pass.
    """
    ref = ref.detach()
    tri = tri.detach()
    if last_dim is not None:
        ref = ref[..., :last_dim]
        tri = tri[..., :last_dim]
    assert ref.shape == tri.shape, f"{prefix}: shape {ref.shape} != {tri.shape}"
    assert not torch.isnan(ref).any(), f"{prefix}: NaN detected in ref"
    assert not torch.isnan(tri).any(), f"{prefix}: NaN detected in tri"

    if cmp_dtype is None:
        cmp_dtype = tri.dtype if tri.dtype.is_floating_point else torch.float32
    ref_c = ref.to(cmp_dtype).float()
    tri_c = tri.to(cmp_dtype).float()
    diff = (ref_c - tri_c).abs()
    max_diff = diff.max().item()
    if atol is None:
        atol = _default_aligned_atol(ref, tri, cmp_dtype)

    msg = f"{prefix:>16} max_abs_diff: {max_diff:.6g} aligned_atol: {atol:.6g}"
    logger.info(msg)
    assert max_diff <= atol, msg


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    error_rate = get_err_ratio(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {error_rate:.6f}"
    logger.info(msg)
    if abs_atol <= err_atol:
        return
    assert not torch.isnan(ref).any(), f"{prefix}: NaN detected in ref"
    assert not torch.isnan(tri).any(), f"{prefix}: NaN detected in tri"
    if warning or (FLA_CI_ENV and (error_rate < 0.01 or abs_atol <= 0.3)):
        if error_rate > ratio:
            warnings.warn(msg)
    else:
        assert error_rate < ratio, msg
