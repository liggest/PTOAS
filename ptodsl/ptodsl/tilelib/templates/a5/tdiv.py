# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tdiv — default precision only.

Ported from lib/TileOps/tdiv_template.py, but only the default-precision branch (plain
pto.vdiv). The high-precision (IEEE-754) path is deferred: it needs a `get_op_attr`
bridge to read the `precisionType` context attr the daemon already receives, plus the
div_hp algorithm — tracked as a follow-up.
"""

from ptodsl import pto
import ptodsl.tilelib as tilelib


@tilelib.tile_template(
    op="pto.tdiv",
    target="a5",
    name="template_tdiv",
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
def template_tdiv(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            divided = pto.vdiv(lhs, rhs, mask)
            pto.vsts(divided, dst[row, col:], mask)
