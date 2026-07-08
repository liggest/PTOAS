# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tmax (ported from lib/TileOps/tmax_template.py)."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


@tilelib.tile_template(
    op="pto.tmax",
    target="a5",
    name="template_tmax",
    dtypes=[("f32", "f32", "f32")],
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    layouts=["row_major"],
    memory_spaces=["ub"],
    priority=0,
    id=0,
    loop_depth=2,
    is_post_update=False,
)
def template_tmax(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            max_val = pto.vmax(lhs, rhs, mask)
            pto.vsts(max_val, dst[row, col:], mask)
