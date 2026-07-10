#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
VecValue arithmetic end-to-end ST (SIMT).

Each SIMT thread loads two ``vector<4xf32>`` slices from UB via
``scalar.load(..., contiguous=4)`` and writes back the three arithmetic
results (``x+y``, ``x-y``, ``x*y``) using ``scalar.store``.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto, scalar


THREADS = 8
VEC = 4
N = THREADS * VEC  # element count per operand vector (x and y)
F32_BYTES = 4
IN_BYTES = 2 * N * F32_BYTES   # x || y
OUT_BYTES = 3 * N * F32_BYTES  # x+y || x-y || x*y
SEED = 0x5761


@pto.simt
def vec_value_arith_simt_body(a_ub, o_ub):
    tid = pto.get_tid_x()
    base = tid * VEC
    x = scalar.load(a_ub, base, contiguous=VEC)
    y = scalar.load(a_ub, N + base, contiguous=VEC)
    scalar.store(x + y, o_ub, 0 * N + base)
    scalar.store(x - y, o_ub, 1 * N + base)
    scalar.store(x * y, o_ub, 2 * N + base)


@pto.jit(
    name="vec_value_arith_simt_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def vec_value_arith_simt_kernel(
    a_ptr: pto.ptr(pto.f32, "gm"),
    o_ptr: pto.ptr(pto.f32, "gm"),
):
    ub_base = pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    a_ub = pto.addptr(ub_base, 0)
    o_ub = pto.addptr(ub_base, IN_BYTES // F32_BYTES)

    pto.mte_gm_ub(a_ptr, a_ub, 0, IN_BYTES, nburst=(1, IN_BYTES, IN_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    vec_value_arith_simt_body[THREADS, 1, 1](a_ub, o_ub)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_ub_gm(o_ub, o_ptr, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))
    pto.pipe_barrier(pto.Pipe.ALL)


def make_inputs():
    rng = np.random.RandomState(SEED)
    x = rng.uniform(0.5, 2.0, size=N).astype(np.float32)
    y = rng.uniform(0.5, 2.0, size=N).astype(np.float32)
    return [np.concatenate([x, y])]


def make_expected(inp):
    x = inp[:N]
    y = inp[N:]
    return np.concatenate([
        x + y,
        x - y,
        x * y,
    ]).astype(np.float32)


CASES = [
    golden_output_case(
        "vec_value_arith_simt",
        vec_value_arith_simt_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=1.0e-5,
        atol=1.0e-5,
    ),
]


auto_main(globals())
