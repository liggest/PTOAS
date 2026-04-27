"""TileLang DSL template for pto.tcmp

Aligned with pto-isa/include/pto/npu/a5/TCmp.hpp:
- 32-bit (int32, float, uint32): TCmp_32B with plt_b32 + pdintlv_b8
- 16-bit (int16, half, uint16): TCmp with plt_b16
- 8-bit (int8, uint8): TCmp with plt_b8
"""

import tilelang_dsl as pto


REPEAT_BYTE = 256
CMP_BITS_PER_INDEX = 32


@pto.vkernel(
    target="a5",
    op="pto.tcmp",
    dtypes=[
        (pto.si32, pto.si32, pto.i8),
        (pto.f32, pto.f32, pto.i8),
        (pto.ui32, pto.ui32, pto.i8),
        (pto.si16, pto.si16, pto.i8),
        (pto.f16, pto.f16, pto.i8),
        (pto.ui16, pto.ui16, pto.i8),
        (pto.si8, pto.si8, pto.i8),
        (pto.ui8, pto.ui8, pto.i8),
    ],
    advanced=True,
)
def template_tcmp(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = src0.element_type
    valid_rows, valid_cols = src0.valid_shape
    cmp_mode = pto.get_op_attr("cmp_mode", pto.CmpMode.EQ)

    dtype_size = pto.bytewidth(dtype)
    total_elements = valid_rows * valid_cols
    repeat_elm = REPEAT_BYTE // dtype_size

    src0_ptr = src0.as_ptr()
    src1_ptr = src1.as_ptr()
    dst_ptr = dst.as_ptr()
    dst_u32_ptr = pto.castptr(dst_ptr, pto.ptr(pto.ui32, pto.MemorySpace.UB))

    if pto.constexpr(dtype_size == 4):
        repeat_elm_32b = REPEAT_BYTE // 4
        repeat_times_32b = (total_elements + repeat_elm_32b - 1) // repeat_elm_32b + 1
        loop_times = repeat_times_32b // 2
        remaining = total_elements

        for i in range(0, loop_times, 1):
            preg0, remaining = pto.plt_b32(remaining)
            vreg0 = pto.vlds(src0_ptr, i * 2 * repeat_elm_32b)
            vreg1 = pto.vlds(src1_ptr, i * 2 * repeat_elm_32b)
            preg1 = pto.vcmp(vreg0, vreg1, preg0, cmp_mode)

            preg0, remaining = pto.plt_b32(remaining)
            vreg2 = pto.vlds(src0_ptr, (i * 2 + 1) * repeat_elm_32b)
            vreg3 = pto.vlds(src1_ptr, (i * 2 + 1) * repeat_elm_32b)
            preg2 = pto.vcmp(vreg2, vreg3, preg0, cmp_mode)

            preg1_b8 = pto.pbitcast(preg1, pto.mask_b8)
            preg2_b8 = pto.pbitcast(preg2, pto.mask_b8)
            preg3, preg4 = pto.pdintlv_b8(preg1_b8, preg2_b8)
            pto.psts(preg3, dst_u32_ptr, i * 16, pto.PredicateDist.PK)
    elif pto.constexpr(dtype_size == 2):
        repeat_times = (total_elements + repeat_elm - 1) // repeat_elm
        dst_stride_bytes = (repeat_elm // CMP_BITS_PER_INDEX) * 4
        remaining = total_elements

        for i in range(0, repeat_times, 1):
            preg0, remaining = pto.plt_b16(remaining)
            vreg0 = pto.vlds(src0_ptr, i * repeat_elm)
            vreg1 = pto.vlds(src1_ptr, i * repeat_elm)
            preg1 = pto.vcmp(vreg0, vreg1, preg0, cmp_mode)
            pto.psts(preg1, dst_u32_ptr, i * dst_stride_bytes, pto.PredicateDist.PK)
    elif pto.constexpr(dtype_size == 1):
        repeat_times = (total_elements + repeat_elm - 1) // repeat_elm
        dst_stride_bytes = (repeat_elm // CMP_BITS_PER_INDEX) * 4
        remaining = total_elements

        for i in range(0, repeat_times, 1):
            preg0, remaining = pto.plt_b8(remaining)
            vreg0 = pto.vlds(src0_ptr, i * repeat_elm)
            vreg1 = pto.vlds(src1_ptr, i * repeat_elm)
            preg1 = pto.vcmp(vreg0, vreg1, preg0, cmp_mode)
            pto.psts(preg1, dst_u32_ptr, i * dst_stride_bytes, pto.PredicateDist.PK)

    return