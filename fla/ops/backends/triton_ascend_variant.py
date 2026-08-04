# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Select between FLA-native and MindSpeed-Ops Ascend Triton kernels.

Set ``FLA_TRITON_ASCEND_IMPL=mindspeed_ops`` to dispatch GDN-related Ascend
kernels to the migrated MindSpeed-Ops arch32 implementations under each
``triton_ascend/mindspeed_ops`` package. Default is ``fla``.
"""

from __future__ import annotations

import os

_VALID = {"fla", "mindspeed_ops"}


def triton_ascend_impl() -> str:
    value = os.environ.get("FLA_TRITON_ASCEND_IMPL", "fla").strip().lower()
    if value not in _VALID:
        raise ValueError(
            f"FLA_TRITON_ASCEND_IMPL must be one of {sorted(_VALID)}, got {value!r}"
        )
    return value


def use_mindspeed_ops() -> bool:
    return triton_ascend_impl() == "mindspeed_ops"
