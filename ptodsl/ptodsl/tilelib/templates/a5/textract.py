# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for the UB row-major ``pto.textract`` path."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import NUMERIC_DTYPES


def _vec_to_vec_nd(src_memory_space, dst_memory_space, src_config, dst_config, src_dtype, dst_dtype, **_):
    return (
        src_memory_space == "ub"
        and dst_memory_space == "ub"
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config.b_layout == "row_major"
        and dst_config.s_layout == "none_box"
        and src_dtype == dst_dtype
    )


@tilelib.tile_template(
    op="pto.textract",
    target="a5",
    name="template_textract_vec2vec_nd",
    dtypes=[(dtype, "i32", "i32", dtype) for dtype in NUMERIC_DTYPES],
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[_vec_to_vec_nd],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("move", "extract", "ub"),
)
def template_textract_vec2vec_nd(
    src: pto.Tile,
    index_row: pto.i32,
    index_col: pto.i32,
    dst: pto.Tile,
):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            data = pto.vlds(src[index_row + row, index_col + col:])
            pto.vsts(data, dst[row, col:], mask)
