# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tcmps."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import ub_row_major_constraints


@tilelib.tile_template(
    op="pto.tcmps",
    target="a5",
    name="template_tcmps",
    dtypes=[
        ("f32", "f32", "ui8"),
        ("i32", "i32", "ui8"),
        ("f16", "f16", "ui8"),
        ("i16", "i16", "ui8"),
        ("i8", "i8", "ui8"),
        ("ui8", "ui8", "ui8"),
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=ub_row_major_constraints("src", "dst"),
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("compare", "scalar", "predicate-store"),
)
def template_tcmps(src: pto.Tile, scalar, dst: pto.Tile):
    dtype = src.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    cmp_mode = pto.get_op_attr("cmp_mode", "eq")
    dst_ptr = dst.as_ptr()
    dst_stride = dst.shape[1]

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            value = pto.vlds(src[row, col:])
            cmp = pto.vcmps(value, scalar, mask, cmp_mode)
            store_offset = row * dst_stride + col // 8
            pto.psts(cmp, dst_ptr, store_offset, dist="NORM")
