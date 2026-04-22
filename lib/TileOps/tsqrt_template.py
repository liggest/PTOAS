import tilelang_dsl as pto

# TODO: Add implementation for HIGH_PRECISION type
@pto.vkernel(
    target="a5",
    op="pto.tsqrt"
)
def template_tsqrt(src: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            vinput = pto.vlds(src[row, col:])
            result = pto.vsqrt(vinput, mask)
            pto.vsts(result, dst[row, col:], mask)
    return