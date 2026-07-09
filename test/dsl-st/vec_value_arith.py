#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
VecValue arithmetic end-to-end ST.

Exercises the four floating-point vector arithmetic operators on
``vector<4xf32>`` values produced by ``scalar.load(..., contiguous=4)`` and
consumed by ``scalar.store``.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto, scalar


VEC = 4
SEED = 0x5761

@pto.jit(
    name="vec_value_arith_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def vec_value_arith_kernel(
    a_ptr: pto.ptr(pto.f32, "gm"),
    o_ptr: pto.ptr(pto.f32, "gm"),
):
    x = scalar.load(a_ptr, 0, contiguous=VEC)
    y = scalar.load(a_ptr, VEC, contiguous=VEC)
    scalar.store(x + y, o_ptr, 0 * VEC)
    scalar.store(x - y, o_ptr, 1 * VEC)
    scalar.store(x * y, o_ptr, 2 * VEC)
    scalar.store(x / y, o_ptr, 3 * VEC)


def make_inputs():
    rng = np.random.RandomState(SEED)
    x = rng.uniform(0.5, 2.0, size=VEC).astype(np.float32)
    y = rng.uniform(0.5, 2.0, size=VEC).astype(np.float32)
    return [np.concatenate([x, y])]


def make_expected(inp):
    x = inp[:VEC]
    y = inp[VEC:]
    return np.concatenate([
        x + y,
        x - y,
        x * y,
        x / y,
    ]).astype(np.float32)


CASES = [
    golden_output_case(
        "vec_value_arith",
        vec_value_arith_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=1.0e-5,
        atol=1.0e-5,
    ),
]


auto_main(globals())
