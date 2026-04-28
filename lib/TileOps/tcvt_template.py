# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tcvt."""

import tilelang_dsl as pto


def _supports_basic_rowwise_tcvt(
    src_shape=(),
    src_valid_shape=(),
    dst_shape=(),
    dst_valid_shape=(),
    src_config=None,
    dst_config=None,
):
    if tuple(src_shape) != tuple(dst_shape):
        return False
    if tuple(src_valid_shape) != tuple(dst_valid_shape):
        return False
    if len(src_shape) != 2 or len(dst_shape) != 2:
        return False
    if src_config is None or dst_config is None:
        return False
    if src_config.b_layout != pto.BLayout.ROW_MAJOR:
        return False
    if dst_config.b_layout != pto.BLayout.ROW_MAJOR:
        return False
    if src_config.s_layout != pto.SLayout.NONE_BOX:
        return False
    if dst_config.s_layout != pto.SLayout.NONE_BOX:
        return False
    return True


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.f16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f32_to_f16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
            store_mask, remained = pto.make_mask(pto.f32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.f16,
                full_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.i32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f32_to_i32(src: pto.Tile, dst: pto.Tile):
    dst_dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dst_dtype)):
            mask, remained = pto.make_mask(dst_dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.SAT,
            )
            pto.vsts(converted, dst[row, col:], mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f16, pto.f32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f16_to_f32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.f16, pto.PAT.ALL)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
            store_mask, remained = pto.make_mask(pto.f32, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B16)
            converted = pto.vcvt(
                vec,
                pto.f32,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i32, pto.f32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i32_to_f32(src: pto.Tile, dst: pto.Tile):
    dst_dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dst_dtype)):
            mask, remained = pto.make_mask(dst_dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                mask,
                rnd=rnd,
            )
            pto.vsts(converted, dst[row, col:], mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.bf16, pto.f32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_bf16_to_f32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.bf16, pto.PAT.ALL)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
            store_mask, remained = pto.make_mask(pto.f32, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B16)
            converted = pto.vcvt(
                vec,
                pto.f32,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i16, pto.f32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i16_to_f32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.i16, pto.PAT.ALL)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
            store_mask, remained = pto.make_mask(pto.f32, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B16)
            converted = pto.vcvt(
                vec,
                pto.f32,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i16, pto.i32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i16_to_i32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.i16, pto.PAT.ALL)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.i32)):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B16)
            converted = pto.vcvt(
                vec,
                pto.i32,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i16, pto.ui32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i16_to_ui32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.i16, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.ui32)):
            store_mask, remained = pto.make_mask(pto.ui32, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B16)
            converted = pto.vcvt(
                vec,
                pto.ui32,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i32, pto.i64),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i32_to_i64(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols * 2  # i64 requires double the mask
        for col in range(0, valid_cols, pto.get_lanes(pto.i64)):
            store_mask, remained = pto.make_mask(pto.i64, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B32)
            converted = pto.vcvt(
                vec,
                pto.i64,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.NORM_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.ui8, pto.f16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_ui8_to_f16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.ui8, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f16)):
            store_mask, remained = pto.make_mask(pto.f16, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B8)
            converted = pto.vcvt(
                vec,
                pto.f16,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.ui8, pto.ui16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_ui8_to_ui16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.ui8, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.ui16)):
            store_mask, remained = pto.make_mask(pto.ui16, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B8)
            converted = pto.vcvt(
                vec,
                pto.ui16,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.si8, pto.f16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_si8_to_f16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.si8, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f16)):
            store_mask, remained = pto.make_mask(pto.f16, remained)
            vec_si8 = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B8)
            converted = pto.vcvt(
                vec_si8,
                pto.f16,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.si8, pto.si16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_si8_to_si16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.si8, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.si16)):
            store_mask, remained = pto.make_mask(pto.si16, remained)
            vec_si8 = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B8)
            converted = pto.vcvt(
                vec_si8,
                pto.si16,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.NORM_B16)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.f32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f32_to_f32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
            mask, remained = pto.make_mask(pto.f32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vtrc(vec, mask, rnd=rnd)
            pto.vsts(converted, dst[row, col:], mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f16, pto.i32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f16_to_i32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.i32)):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B16)
            converted = pto.vcvt(
                vec,
                pto.i32,
                full_mask,
                rnd=rnd,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i16, pto.f16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i16_to_f16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.i16, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.i16)):
            store_mask, remained = pto.make_mask(pto.f16, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.f16,
                full_mask,
                rnd=rnd,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i64, pto.f32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i64_to_f32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    for row in range(0, valid_rows, 1):
        remained = valid_cols * 2  # i64 requires double the mask
        full_mask, _ = pto.make_mask(pto.i64, remained)
        for col in range(0, valid_cols, pto.get_lanes(pto.i64)):
            store_mask, remained = pto.make_mask(pto.f32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.f32,
                full_mask,
                rnd=rnd,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B64)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i16, pto.ui8),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i16_to_ui8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.i16, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.i16)):
            store_mask, remained = pto.make_mask(pto.i16, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.ui8,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B16)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i32, pto.i16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i32_to_i16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.i32)):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.i16,
                full_mask,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i32, pto.ui16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i32_to_ui16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.i32)):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.ui16,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.ui32, pto.i16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_ui32_to_i16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.ui32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.ui32)):
            store_mask, remained = pto.make_mask(pto.ui32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.i16,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.ui32, pto.ui16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_ui32_to_ui16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.ui32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.ui32)):
            store_mask, remained = pto.make_mask(pto.ui32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.ui16,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i64, pto.i32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i64_to_i32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    for row in range(0, valid_rows, 1):
        remained = valid_cols * 2  # i64 requires double the mask
        full_mask, _ = pto.make_mask(pto.i64, remained)
        for col in range(0, valid_cols, pto.get_lanes(pto.i64)):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.i32,
                full_mask,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B64)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.bf16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f32_to_bf16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
            store_mask, remained = pto.make_mask(pto.f32, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.bf16,
                full_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.i64),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f32_to_i64(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols * 2  # i64 requires double the mask
        for col in range(0, valid_cols, pto.get_lanes(pto.i64)):
            store_mask, remained = pto.make_mask(pto.i64, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B32)
            converted = pto.vcvt(
                vec,
                pto.i64,
                full_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.NORM_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.bf16, pto.i32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_bf16_to_i32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.bf16, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.i32)):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B16)
            converted = pto.vcvt(
                vec,
                pto.i32,
                full_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.bf16, pto.f16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_bf16_to_f16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask = pto.make_mask(pto.bf16, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.bf16)):
            store_mask, remained = pto.make_mask(pto.f16, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.f16,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                rnd=rnd,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.i16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f32_to_i16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z

    full_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
            store_mask, remained = pto.make_mask(pto.f32, remained)
            vec_f32 = pto.vlds(src[row, col:])
            # sat=OFF NonSatTorch
            vec_i32 = pto.vcvt(
                vec_f32,
                pto.i32,
                full_mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.NOSAT,
            )
            vec_i16 = pto.vcvt(
                vec_i32,
                pto.i16,
                full_mask,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(vec_i16, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f16, pto.i16),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f16_to_i16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    full_mask_b16 = pto.make_mask(pto.f16, pto.PAT.ALL)
    full_mask_b32 = pto.make_mask(pto.i32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec_f16 = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B16)
            # sat=OFF NonSatTorch
            vec_i32 = pto.vcvt(
                vec_f16,
                pto.i32,
                full_mask_b16,
                rnd=rnd,
                part=pto.VcvtPartMode.EVEN,
            )
            vec_i16 = pto.vcvt(
                vec_i32,
                pto.i16,
                full_mask_b32,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(vec_i16, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
    return
