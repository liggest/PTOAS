# ptodsl — PTO Python IR Builders

This directory contains Python scripts that construct PTO MLIR IR modules
programmatically using the MLIR Python bindings. Two complete kernel examples
are provided, each in a **low-level** (raw bindings) and a **high-level**
(utility-wrapped) variant.

---

## Directory layout

```
ptodsl/
├── ptodsl_utils.py                     # Reusable utility wrappers
│
├── tile_and_vpto_builder_lowlevel.py   # TADD kernel – raw bindings
├── tile_and_vpto_builder_highlevel.py  # TADD kernel – ptodsl_utils
│
├── softmax_builder_lowlevel.py         # Softmax kernel – raw bindings
├── softmax_builder_highlevel.py        # Softmax kernel – ptodsl_utils
│
└── check_ir.py                         # IR correctness test for all builders
```

---

## Prerequisites

The ptoas dialect must be installed and the environment set up before use:

```bash
# Install (first time only)
cd /workdir/ptoas_a5
bash quick_install.sh

# Set up environment in every new shell
source set_ptoas_env.sh
```

---

## Running the IR check

```bash
# From ptoas_a5/ptodsl/
python3 check_ir.py

# Or from the repository root (ptoas_a5/)
python3 ptodsl/check_ir.py
```

Expected output when everything is correct:

```
ptodsl IR check
==================================================
  PASS  TADD  low-level
  PASS  TADD  high-level
  PASS  softmax low-level
  PASS  softmax high-level
==================================================
Result: ALL PASS
```

Exit code is `0` on full pass, `1` if any check fails.  
A unified diff of the first 60 diverging lines is printed for each failing case.

---

## Kernel examples

### TADD — simple vector add (vPTO)

| File | Reference |
|---|---|
| `tile_and_vpto_builder_lowlevel.py` | `test/lit/vpto/expand_tileop_to_vpto_result.pto` |
| `tile_and_vpto_builder_highlevel.py` | same |

The kernel performs an element-wise vector add over a 1024-element float32
buffer using 16 iterations of 64-wide vector instructions inside a
`pto.vecscope`.  It exercises:
`pto.castptr`, `pto.addptr`, `pto.plt_b32`, `pto.vlds`, `pto.vadd`,
`pto.vsts`, nested modules (`pto.target_arch` + `pto.kernel_kind`).

### Online softmax update

| File | Reference |
|---|---|
| `softmax_builder_lowlevel.py` | `test/tilelang_st/npu/a5/src/st/testcase/softmax/softmax.pto` |
| `softmax_builder_highlevel.py` | same |

An online softmax update kernel that mixes tile-domain loads/stores with
raw vector compute inside a `pto.vecscope`.  It exercises a significantly
larger set of ops including:
`pto.get_block_idx`, `pto.make_tensor_view`, `pto.partition_view`,
`pto.alloc_tile`, `pto.tload`/`pto.tstore`, `pto.set_flag`/`pto.wait_flag`,
`pto.tile_buf_addr`, `pto.pset_b32`, `pto.vcmax`, `pto.vdup`, `pto.vmax`,
`pto.vexpdif`, `pto.vmul`, `pto.vcadd`, `pto.vdiv`, `pto.barrier`,
`scf.for` with `iter_args`, and `scf.if` with result values.

---

## How the IR check works

`check_ir.py` calls `build()` in each builder, then compares the resulting
module against its reference `.pto` file using MLIR round-trip normalization:

```
generated IR  ──┐
                ├── Module.parse() → canonical string ──── == ──── PASS/FAIL
reference .pto ──┘  (strips comments, normalises SSA names and attr order)
```

**Why round-trip normalization?**

| Issue | Raw text comparison | Round-trip comparison |
|---|---|---|
| `// comment` lines in `.pto` files | breaks | ignored by MLIR parser |
| Named SSA values (`%block_idx`) vs anonymous (`%0`) | breaks | both become `%0`, `%1` … |
| Attribute dict ordering (`{a=1, b=2}` vs `{b=2, a=1}`) | breaks | normalized |
| Constant declaration order | breaks | **preserved** – must match |

Because constant declaration order is preserved after round-trip, builders
must emit constants in the same order as the reference.  The `check_ir.py`
diff output makes such mismatches easy to locate.

---

## `ptodsl_utils.py` – utility reference

The utility module eliminates boilerplate so kernel logic is immediately
readable.  All helpers operate on the **current** MLIR context and insertion
point; no context argument is threaded.

### Type constructors

| Helper | MLIR type |
|---|---|
| `i32_type()` | `i32` |
| `i64_type()` | `i64` |
| `idx_type()` | `index` |
| `ptr_type(elem, space="ub")` | `!pto.ptr<elem, space>` |
| `vreg_type(lanes, elem)` | `!pto.vreg<lanesxelem>` |
| `mask_type(bits="b32")` | `!pto.mask<bits>` |

### Constant builders

| Helper | Op |
|---|---|
| `c_idx(v)` | `arith.constant v : index` |
| `c_i32(v)` | `arith.constant v : i32` |
| `c_i64(v)` | `arith.constant v : i64` |

### Arithmetic

`muli`, `addi`, `subi` — `arith.muli/addi/subi`  
`index_cast(type, val)` — `arith.index_cast`  
`cmpi_sgt(a, b)` — `arith.cmpi sgt`  
`select_val(cond, t, f)` — `arith.select`

### Module / function builders

```python
with pto_context():                          # MLIR Context + PTO dialect
    with vpto_kernel("MyKernel", arch="a5") as mod:   # nested module + func (no args)
        ...

with pto_context():
    with flat_pto_module("a5") as mod:       # flat module + pto.kernel_kind
        with pto_aicore_func("f", [ptr_gm, i32]) as (p, n):  # func with args
            ...
```

### Control-flow helpers

```python
with vecscope():               # pto.vecscope { ... }

with for_range(lo, hi, step) as i:       # scf.for, auto-inserts scf.yield
    ...

with for_range_iter(lo, hi, step, [a, b]) as cf:  # scf.for with iter_args
    x, y = cf.inner_iter_args
    yield_vals(new_x, new_y)             # scf.yield at end of body
final_x, final_y = cf.results           # results accessible after the block

with if_ctx(cond):             # scf.if, no results, auto-inserts scf.yield
    ...

br = if_op_returning(cond, [vreg, vreg])  # scf.if with results + else
with InsertionPoint(br.then_block):
    yield_vals(a, b)
with InsertionPoint(br.else_block):
    yield_vals(c, d)
x, y = br.results
```

### Tile-domain helpers

```python
tv  = tile_view(tv_type, ptr, shape, strides)      # pto.make_tensor_view
ptv = part_view(ptv_type, tv, offsets, sizes)       # pto.partition_view
t   = alloc_tile(tile_type, addr=a, valid_row=r, valid_col=c)  # pto.alloc_tile
tload(part, tile)                                   # pto.tload
tstore(tile, part)                                  # pto.tstore
ub  = tile_ptr(tile, ptr_ub_type)                   # pto.tile_buf_addr
```

### Vector / pointer helpers

```python
ptr  = castptr(int_addr, ptr_type)                  # pto.castptr
ptr2 = addptr(ptr, offset)                          # pto.addptr
v    = vlds(ptr, offset, vreg_type)                 # pto.vlds
v    = vbrc_load(ptr, offset, vreg_type)            # pto.vlds {dist="BRC_B32"}
vsts(v, ptr, offset, mask)                          # pto.vsts
vsts_1pt(v, ptr, offset, mask)                      # pto.vsts {dist="1PT_B32"}
mask, _ = plt_b32(scalar)                           # pto.plt_b32
mask     = pset_b32("PAT_ALL")                      # pto.pset_b32
```

### Vector math (result type inferred from first operand)

```python
vcmax(v, mask)              # cross-lane max reduction
vdup_lowest(v, mask)        # broadcast lane 0 to all lanes
vmax(a, b, mask)            # element-wise max
vexpdif(x, ref, mask)       # exp(x − ref), ODD lanes
vmul(a, b, mask)            # element-wise multiply
vcadd(v, mask)              # cross-lane add (sum reduction)
vadd(a, b, mask)            # element-wise add  (result_type optional)
vdiv(a, b, mask)            # element-wise divide
```

### Hardware / sync

```python
get_block_idx()             # pto.get_block_idx → i64
barrier_all()               # pto.barrier #pto.pipe<PIPE_ALL>
# use pto.set_flag / pto.wait_flag directly (from mlir.dialects.pto)
# use yield_vals(*vals) as shorthand for scf.YieldOp(list(vals))
```
