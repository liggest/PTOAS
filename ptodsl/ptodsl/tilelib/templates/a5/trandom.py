# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for ``pto.trandom``."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


TRANDOM_ONCE_REPEAT = 4
TRANDOM_CONST_0 = 0xD2511F53
TRANDOM_CONST_1 = 0xCD9E8D57
TRANDOM_CONST_KEY_ADD_0 = 0x9E3779B9
TRANDOM_CONST_KEY_ADD_1 = 0xBB67AE85


def _row_major_dst(dst_config, **_):
    return dst_config.b_layout == "row_major"


def _philox_round(ctr0, ctr1, ctr2, ctr3, key0, key1, const0, const1, mask):
    tmp_l0, tmp_h0 = pto.vmull(ctr0, const0, mask)
    tmp_l1, tmp_h1 = pto.vmull(ctr2, const1, mask)
    tmp_h1 = pto.vxor(tmp_h1, ctr1, mask)
    ctr0 = pto.vxor(tmp_h1, key0, mask)
    tmp_h0 = pto.vxor(tmp_h0, ctr3, mask)
    ctr2 = pto.vxor(tmp_h0, key1, mask)
    key0 = pto.vadds(key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), mask)
    key1 = pto.vadds(key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), mask)
    return ctr0, tmp_l1, ctr2, tmp_l0, key0, key1


@tilelib.tile_template(
    op="pto.trandom",
    target="a5",
    name="template_trandom",
    dtypes=[("i32", "i32", "i32", "i32", "i32", "i32", "ui32")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_row_major_dst],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("random", "philox"),
)
def template_trandom(
    key0: pto.i32,
    key1: pto.i32,
    counter0: pto.i32,
    counter1: pto.i32,
    counter2: pto.i32,
    counter3: pto.i32,
    dst: pto.Tile,
):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    repeats = (dst.shape[1] + TRANDOM_ONCE_REPEAT * lanes - 1) // (
        TRANDOM_ONCE_REPEAT * lanes
    )
    rounds = int(pto.get_op_attr("rounds", "10"))

    full_mask = pto.pset_b32(pto.PAT.ALL)
    ctr0_init = pto.vbitcast(pto.vbr(counter0), pto.ui32)
    ctr1_init = pto.vbitcast(pto.vbr(counter1), pto.ui32)
    ctr2_init = pto.vbitcast(pto.vbr(counter2), pto.ui32)
    ctr3_init = pto.vbitcast(pto.vbr(counter3), pto.ui32)
    key0_vec = pto.vbitcast(pto.vbr(key0), pto.ui32)
    key1_vec = pto.vbitcast(pto.vbr(key1), pto.ui32)
    zeros = pto.vbr(pto.ui32(0))
    const0 = pto.vbr(pto.ui32(TRANDOM_CONST_0))
    const1 = pto.vbr(pto.ui32(TRANDOM_CONST_1))
    inc_idx = pto.vbitcast(pto.vci(0), pto.ui32)

    ctr0, carry = pto.vaddc(ctr0_init, inc_idx, full_mask)
    ctr1, carry = pto.vaddcs(ctr1_init, zeros, carry, full_mask)
    ctr2, carry = pto.vaddcs(ctr2_init, zeros, carry, full_mask)
    ctr3, _ = pto.vaddcs(ctr3_init, zeros, carry, full_mask)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        counter_add = lanes
        for repeat in range(0, repeats, 1):
            _ = repeat
            tmp0, tmp1, tmp2, tmp3 = ctr0, ctr1, ctr2, ctr3
            tmp_key0, tmp_key1 = key0_vec, key1_vec
            for _round in range(0, rounds, 1):
                _ = _round
                tmp0, tmp1, tmp2, tmp3, tmp_key0, tmp_key1 = _philox_round(
                    tmp0, tmp1, tmp2, tmp3, tmp_key0, tmp_key1, const0, const1, full_mask
                )

            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(tmp0, dst[row, (counter_add - lanes):], mask)
            counter_add = counter_add + lanes
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(tmp1, dst[row, (counter_add - lanes):], mask)
            counter_add = counter_add + lanes
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(tmp2, dst[row, (counter_add - lanes):], mask)
            counter_add = counter_add + lanes
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(tmp3, dst[row, (counter_add - lanes):], mask)
            counter_add = counter_add + lanes

            ctr0, carry = pto.vaddc(ctr0, pto.vbr(pto.ui32(lanes * TRANDOM_ONCE_REPEAT)), full_mask)
            ctr1, carry = pto.vaddcs(ctr1, zeros, carry, full_mask)
            ctr2, carry = pto.vaddcs(ctr2, zeros, carry, full_mask)
            ctr3, _ = pto.vaddcs(ctr3, zeros, carry, full_mask)
