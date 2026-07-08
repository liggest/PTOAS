# PTODSL TileLib Migration Test Checklist

This page tracks the tests used while migrating PTOAS TileLib expansion from
the legacy TileLang implementation to PTODSL. Run commands from the repository
root.

## Environment

Set up PTOAS, PTODSL, MLIR, and LLVM test-tool paths:

```bash
export PTOAS_ENV_SKIP_SMOKE_TEST=1
source scripts/ptoas_env.sh
export FILECHECK="$LLVM_BUILD_DIR/bin/FileCheck"
```

The Python-only tests do not require rebuilding PTOAS. Tests that invoke
`ptoas` must use a binary rebuilt after the corresponding C++ or TableGen
changes.

## Current migration status

The PTODSL A5 TileLib now has module-level coverage for the legacy
`lib/TileOps/*_template.py` catalog. The remaining work is semantic parity,
version parity, and validation under the real VPTO pipeline.

| Area | Status | Notes |
|---|---|---|
| Template module coverage | Covered | Every legacy TileOps template module has a PTODSL A5 module counterpart. PTODSL also has helper-only modules such as `tbinop.py` and extra split coverage such as `textract_fp.py`. |
| `tpartadd`, `tpartmul`, `tpartmax`, `tpartmin` | Fixed | Partial valid-shape behavior now follows the legacy overlay/pad semantics. Verified with Python catalog tests and `tpart*` ST smoke compile through `--tile-lib-backend=ptodsl`. |
| `trowprod` | Fixed | Uses dtype-specific vector-collapse depth: 6 stages for 32-bit element types and 7 stages for 16-bit element types. Verified with Python catalog tests plus smoke and full `trowprod` ST compile through PTODSL. |
| Partition-view metadata and bridge casts | Fixed | PTODSL metadata can describe partition views, and TileBuf folding handles the bridge casts seen in the VPTO PTODSL path. |
| VPTO default backend | Not switched yet | `--tile-lib-backend` still defaults to TileLang. Switch this only after the parity gaps below are resolved or explicitly accepted. |

## Known parity gaps and suspicious areas

These are the current items to resolve or explicitly validate before making
PTODSL the default TileLib backend for VPTO.

For a more detailed template-by-template parity matrix, see
`tilelib-template-parity-gaps.md`.

| Area | Risk | Current state / next check |
|---|---|---|
| `trowsum` i16 | High | ST smoke includes an i16 case, but PTODSL currently registers `f16`, `f32`, and `i32` only. Add i16 parity or prove the case should not be legal. |
| `tfmod` / `trem` f16 | Medium-high | PTODSL uses the generic remainder path over f16. Legacy expands even/odd halves through f32 and converts back. Numerical differences are likely for f16. |
| `tfillpad`, `tfillpad_expand`, `tfillpad_inplace` | Medium | Smoke ST passes. Full parity still needs physical-padding and inplace column-expansion validation. |
| `tinsert` non-basic modes | High | PTODSL currently covers basic UB vec-to-vec insertion. Legacy covers additional acc/mat/NZ/quant/relu and mode-driven paths. |
| High-precision math attributes | High when used | High-precision paths remain to be ported or validated for `tdiv`, `tdivs`, `trecip`, `tcolexpanddiv`, `trowexpanddiv`, `tlog`, `texp`, and `tsqrt`. |
| `tcvt` versions | Medium | PTODSL has several registered conversions and round-mode handling, but full legacy dtype/version parity still needs ST coverage. |
| `tbinop` tail/version selection | Medium | `tbinop.has_tail` is still marked as a placeholder. Validate multi-candidate `tadd`/`tmul` selection against legacy behavior. |
| Row/column reductions and arg reductions | Medium | Shared implementations render, but full ST should be run for row/column max/min/sum/prod and argmax/argmin families. |
| Cube and GEMV families | Medium-high | Module coverage exists, but these templates are complex enough that render coverage alone is not enough. Run ST or targeted simulator validation. |

Recommended fix order:

1. `trowsum` i16.
2. f16 `tfmod` / `trem` parity.
3. `tinsert` non-basic variants that appear in ST.
4. `tfillpad*` full physical-padding and inplace column-expansion cases.
5. High-precision math attribute paths.
6. `tcvt` and `tbinop` version-selection parity.
7. Full row/column reduction, arg reduction, cube, and GEMV validation.

## Milestone coverage

| Milestone | Test | Purpose |
|---|---|---|
| Legacy baseline | `expand_tile_op_tilelang_tsub.pto` | Confirms the default TileLang backend still works |
| PTODSL TileLib package | `test_tilelib_catalog.py`, `test_tilelib_constraints.py`, `test_tilelib_elementwise.py`, `test_tilelib_render.py`, `test_tilelib_select.py` | Covers the ported template catalog, legality constraints, template registration and selection, and rendering |
| PTODSL daemon | `test_tilelib_daemon.py` | Covers the Unix-socket protocol, scalar/tile operand specs, metadata, rendering, candidate IDs, and caching |
| PTOAS daemon selection | `expand_tile_op_ptodsl_tsub.pto` | Confirms `--tile-lib-backend=ptodsl` starts and uses the PTODSL daemon |
| Separate metadata/render passes | `expand_tile_op_ptodsl_tadd.pto` | Confirms `InsertTemplateAttributes` records compact metadata before `ExpandTileOp` renders |
| Multi-candidate fallback | `expand_tile_op_ptodsl_tadd.pto` | Confirms `ExpandTileOp` renders candidate index zero when several candidates remain |
| Partition TileOps | `tpart*` ST smoke cases | Confirms partial valid-shape `tpart*` templates compile through the PTODSL daemon |
| Row product | `trowprod` ST smoke and full cases | Confirms dtype-specific row-product reduction depth compiles through the PTODSL daemon |

## Python TileLib tests

Run every Python TileLib test:

```bash
python3 -m unittest discover -s ptodsl/tests -p 'test_tilelib_*.py'
```

Run the layers individually:

```bash
python3 ptodsl/tests/test_tilelib_constraints.py
python3 ptodsl/tests/test_tilelib_catalog.py
python3 ptodsl/tests/test_tilelib_elementwise.py
python3 ptodsl/tests/test_tilelib_render.py
python3 ptodsl/tests/test_tilelib_select.py
python3 ptodsl/tests/test_tilelib_daemon.py
```

Each command prints `OK` when successful.

The current PTODSL TileLib catalog intentionally covers only ops registered in
`ptodsl.tilelib.templates`. The table-driven catalog test is the scalable place
to add one representative render case per newly ported TileOp.

## PTOAS integration tests

Run the focused lit tests through the generated site configuration. Start lit
from `build/test/lit`; passing source files under `test/lit` directly bypasses
the generated LLVM configuration:

```bash
"$LLVM_BUILD_DIR/bin/llvm-lit" -sv build/test/lit \
  --filter='expand_tile_op_(ptodsl_tsub|ptodsl_tadd|tilelang_tsub)'
```

### PTODSL positive path: one legal candidate

`pto.tsub` has one legal PTODSL candidate. The test checks that PTOAS expands
it into vector operations:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  --tile-lib-backend=ptodsl \
  test/lit/vpto/expand_tile_op_ptodsl_tsub.pto -o - 2>/dev/null |
"$FILECHECK" test/lit/vpto/expand_tile_op_ptodsl_tsub.pto
```

### PTODSL candidate attributes and multi-candidate fallback

Inspect the compact candidate list inserted before fusion:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-pto-ir \
  --tile-lib-backend=ptodsl \
  --mlir-print-ir-after=pto-insert-template-attributes \
  test/lit/vpto/expand_tile_op_ptodsl_tadd.pto \
  -o /dev/null 2>&1 |
"$FILECHECK" test/lit/vpto/expand_tile_op_ptodsl_tadd.pto \
  --check-prefix=META
```

Confirm that insertion also runs before `FusionPlan` when fusion is enabled:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --enable-op-fusion --emit-pto-ir --tile-lib-backend=ptodsl \
  --mlir-print-ir-before=pto-fusion-plan \
  test/lit/vpto/expand_tile_op_ptodsl_tadd.pto \
  -o /dev/null 2>&1 |
"$FILECHECK" test/lit/vpto/expand_tile_op_ptodsl_tadd.pto \
  --check-prefix=PREFUSION
```

Inspect `ExpandTileOp` immediately after selection and confirm candidate zero
was used:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  --tile-lib-backend=ptodsl \
  --mlir-print-ir-after=pto-expand-tile-op \
  test/lit/vpto/expand_tile_op_ptodsl_tadd.pto \
  -o /dev/null 2>&1 |
"$FILECHECK" test/lit/vpto/expand_tile_op_ptodsl_tadd.pto \
  --check-prefix=SELECT
```

Confirm the selected template expands through the full VPTO pipeline:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  --tile-lib-backend=ptodsl \
  test/lit/vpto/expand_tile_op_ptodsl_tadd.pto -o - 2>/dev/null |
"$FILECHECK" test/lit/vpto/expand_tile_op_ptodsl_tadd.pto \
  --check-prefix=EXPAND
```

### Legacy backend regression before default switch

Omitting `--tile-lib-backend` must continue to select TileLang:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  test/lit/vpto/expand_tile_op_tilelang_tsub.pto -o - 2>/dev/null |
"$FILECHECK" test/lit/vpto/expand_tile_op_tilelang_tsub.pto
```

After the default backend is intentionally changed to PTODSL, replace this
legacy-default check with one of the following:

- an explicit legacy regression using `--tile-lib-backend=tilelang`; and
- a default-backend PTODSL check that omits `--tile-lib-backend` and confirms
  the PTODSL path is selected.

## ST compile validation

The ST runners can be directed to PTODSL with `PTOAS_TILE_LIB_BACKEND=ptodsl`.
Use this before renaming tests or changing the default backend.

Compile one focused ST `.pto` directly:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  --tile-lib-backend=ptodsl --pto-level=level2 \
  test/tilelang_st/npu/a5/src/st/smoke/testcase/trowprod/trowprod.pto \
  -o /tmp/trowprod_vpto.pto
```

Compile the full `trowprod` case file:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  --tile-lib-backend=ptodsl --pto-level=level2 \
  test/tilelang_st/npu/a5/src/st/testcase/trowprod/trowprod.pto \
  -o /tmp/trowprod_full_vpto.pto
```

Compile the fixed `tpart*` smoke cases:

```bash
for op in tpartadd tpartmul tpartmax tpartmin; do
  ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
    --tile-lib-backend=ptodsl --pto-level=level2 \
    "test/tilelang_st/npu/a5/src/st/smoke/testcase/${op}/${op}.pto" \
    -o /tmp/tpart_vpto.pto || exit 1
done
```

Compile all available ST smoke `.pto` files with PTODSL:

```bash
for f in test/tilelang_st/npu/a5/src/st/smoke/testcase/*/*.pto; do
  echo "RUN $f"
  ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
    --tile-lib-backend=ptodsl --pto-level=level2 \
    "$f" -o /tmp/ptodsl_st_smoke_vpto.pto || exit 1
done
```

Known current failure from this broad smoke loop:

```text
trowsum: dtype signature ('i16', 'i16', 'i16') is not supported
```

Treat each new broad-ST failure as either:

- a PTODSL template parity bug;
- a missing candidate/version registration;
- unsupported metadata for a real operand shape/layout; or
- a test that must remain explicitly TileLang-only until the corresponding
  template family is ported.

## Default-switch checklist

Before changing the VPTO TileLib default to PTODSL:

1. Python TileLib tests pass.
2. Focused PTODSL lit tests pass.
3. Fixed `tpart*` and `trowprod` ST smoke/full compile checks pass.
4. Broad ST smoke either passes or has an explicit waiver list.
5. Known parity gaps above are fixed, waived, or guarded by explicit
   `--tile-lib-backend=tilelang` tests.
6. Default-backend lit tests are renamed from `tilelang_*` to either
   `ptodsl_*` or backend-neutral names.
7. Legacy TileLang coverage remains available through explicit
   `--tile-lib-backend=tilelang`.

## Reading the result

`FileCheck` is silent when it succeeds. Immediately check its status with:

```bash
echo $?
```

`0` means the check passed. When fusion begins filtering candidates, add
coverage for the filtered array while retaining the index-zero fallback test.
