# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""MindSpeed-Ops arch32 l2norm kernels adapted for FLA triton-ascend backends."""

from __future__ import annotations

import torch

from fla.utils import input_guard


@input_guard
def l2norm_fwd_npu(x, eps=1e-6, output_dtype=None):
    from fla.modules.backends.triton_ascend.mindspeed_ops.l2norm import l2norm_fwd

    return l2norm_fwd(x, eps=eps, output_dtype=output_dtype)


@input_guard
def l2norm_bwd_npu(y, rstd, dy, eps=1e-6):
    from fla.modules.backends.triton_ascend.mindspeed_ops.l2norm import l2norm_bwd

    return l2norm_bwd(y, rstd, dy, eps=eps)
