# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tsel."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import ub_row_major_constraints


@tilelib.tile_template(
    op="pto.tsel",
    target="a5",
    name="template_tsel",
    dtypes=[
        ("i8", "f32", "f32", "f32", "f32"),
        ("i8", "f16", "f16", "f16", "f16"),
        ("i8", "i8", "i8", "i8", "i8"),
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=ub_row_major_constraints("src0", "src1", "tmp", "dst"),
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("select", "predicate-load"),
)
def template_tsel(mask: pto.Tile, src0: pto.Tile, src1: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    _ = tmp
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    mask_ptr = pto.castptr(mask.as_ptr(), pto.ptr(pto.ui8, "ub"))
    mask_stride = mask.shape[1]

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            pred, remained = pto.make_mask(dtype, remained)
            mask_offset = row * mask_stride + col // 8
            select_mask = pto.plds(mask_ptr, mask_offset, dist="NORM")
            if str(dtype) == "f16":
                select_mask = pto.pbitcast(select_mask, pto.mask_b16)
            elif str(dtype) == "f32":
                select_mask = pto.pbitcast(select_mask, pto.mask_b32)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            result = pto.vsel(lhs, rhs, select_mask)
            pto.vsts(result, dst[row, col:], pred)
