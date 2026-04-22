import tilelang_dsl as pto

# TODO: Add implementation for HIGH_PRECISION type
@pto.vkernel(
    target="a5",
    op="pto.trsqrt"
)
def template_trsqrt(src: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            vinput = pto.vlds(src[row, col:])
            if pto.constexpr(dtype == pto.f16):
                one_scalar = pto.f16(1.0)
            else:
                one_scalar = pto.f32(1.0)
            one = pto.vbr(one_scalar)
            sqrt_result = pto.vsqrt(vinput, mask)
            result = pto.vdiv(one, sqrt_result, mask)
            pto.vsts(result, dst[row, col:], mask)
    return