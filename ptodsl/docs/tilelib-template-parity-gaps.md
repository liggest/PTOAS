# PTODSL TileLib Template Parity Gaps

This page tracks A5 PTODSL TileLib templates that are known, suspected, or
not yet proven to be on par with the legacy `lib/TileOps` TileLangDSL
templates.

Use this alongside `tilelib-st-failures.md`:

- `tilelib-st-failures.md` tracks concrete ST build/runtime results.
- This file tracks semantic and coverage gaps versus the legacy templates,
  including cases that may pass current smoke ST but still lack full
  TileLangDSL functionality.

## Active ST-Proven Gaps

| Template or family | Current PTODSL state | TileLangDSL behavior | Evidence | What is not on par |
|---|---|---|---|---|
| `tcmp` | Has a packed predicate-store implementation. | Legacy has separate 32-bit, 16-bit, and 8-bit packing paths matching `TCmp.hpp`. | Smoke runtime mismatch: `f32_8x64_gt` and `i32_4x64_ne` compare wrong in `mani_log/all88_20260707-214646/tcmp.log`. | Output packing, predicate store offset, or compare-mode handling is not matching legacy. |
| `tcmps` | Has a scalar compare implementation, but current smoke still fails to instantiate. | Legacy handles 32-bit types with two compares plus `pdintlv_b8`, 16-bit with PK stores, and 8-bit with NORM stores. | `tilelib-st-failures.md` records `ExpandTileOp: failed to instantiate TileLib template for tcmps`. | Candidate selection or scalar operand typing is not matching the legacy callable shape; runtime packing still needs validation after build is fixed. |
| `tcolexpanddiv` | Uses the shared expand-binary divide path for supported signatures. | Legacy supports more dtype/version behavior and high-precision divide through `div_hp.py`. | Smoke asks for `('i32', 'i32', 'i32')`, which PTODSL currently rejects. | Integer dtype/version coverage is missing; high-precision attr parity is also missing. |
| `tcvt` | Implements a small row-wise subset with round/sat/part options. | Legacy `tcvt_template.py` has many conversion families, including more integer widths, bf16/fp4/fp8/hif8 paths, and version-specific behavior. | Smoke asks for a `('f16', 'f32')` signature that PTODSL does not select. | Full dtype matrix and version behavior are not ported. |
| `tdiv` | Default-precision `f32` tile divide only. | Legacy supports default and `precisionType=high_precision`, with broader dtype handling. | Smoke asks for `('f16', 'f16', 'f16')`; PTODSL only registers `f32`. | f16 divide and high-precision divide are missing. |
| `tdivs` | Default-precision scalar divide for f16/f32. | Legacy has two operand orders, default and high-precision paths. | Smoke currently hits a daemon/runtime crash while selecting metadata. | Operand-order, scalar typing, and high-precision parity are not proven. |
| `textract` | Only covers the UB row-major vec-to-vec ND path. | Legacy covers mat/cube extract paths: left/right, same/cross fractal, plus vec-to-vec. | Smoke constraint failure for `template_textract_vec2vec_nd`. | Cube/mat extract variants and some constraints are missing or too narrow. |
| `textract_fp` | Split PTODSL module covers several acc/fp quant signatures. | Legacy implements `textract_fp` in the broader `textract_template.py` with acc-to-mat pre-quant modes. | Existing failure tracker recorded instantiation through `tmatmul`; rerun is needed after later `tmatmul` fixes. | Current status is stale; pre-quant and dependency path parity needs rerun and review. |
| `tlog` | Default-precision unary `vln` only. | Legacy supports default and high-precision log. | Smoke fails to instantiate `tlog`. | Selection/build issue remains; high-precision path is missing. |
| `tmrgsort` | Has single-list and 2/3/4-list templates. | Legacy operand order and dtype signatures match the ST call forms. | Smoke rejects current operand/signature shape: single-list expects 3 operands; multi-list2 rejects `('f16', 'f16', 'f16', 'f16', 'i16')`. | Operand binding and `ex_vec` dtype/signature do not match legacy/ST. |
| `trandom` | Implements a looped Philox-style path. | Legacy unrolls supported round counts and matches ST golden state/counter behavior. | Smoke builds and runs, but compare fails for `int32_1x256` and `int32_4x256`. | Generated random sequence does not match legacy/golden, likely in rounds, counter update, or lane ordering. |
| `trowargmax`, `trowargmin` | Shared row-arg implementation exists. | Legacy supports the ST index dtype and arg output convention. | Smoke rejects `('f32', 'f32', 'ui32')`. | Output index dtype coverage is missing or too narrow. |
| `trowsum` | Shared rowsum registers only a subset. | Legacy supports the i16 rowsum case used by ST. | Smoke rejects `('i16', 'i16', 'i16')`. | i16 row-sum coverage is missing. |
| `tsel` | PTODSL template exists for a constrained mask/select path. | Legacy supports the ST mask/data layout combination. | Smoke reports custom constraints are not satisfied. | Constraints or accepted memory/layout combinations are narrower than legacy. |
| `tsels` | PTODSL scalar select template exists. | Legacy supports the ST scalar/mask dtype combination. | Smoke rejects `('i16', 'i32', 'i32', 'i32', 'i32')`. | Scalar operand and result dtype matrix is incomplete. |
| `tsort32` | Has aligned and tmp-backed unaligned templates. | Legacy supports ST's operand order and tmp/index dtypes. | Smoke sees operand/signature mismatch: current aligned path expects 3 operands, tmp path rejects `('f32', 'ui32', 'f32', 'f32')`. | Operand order and index dtype coverage do not match legacy/ST. |

## Semantic Gaps Not Fully Covered By Current Smoke

| Template or family | Current PTODSL state | TileLangDSL behavior | Risk | What remains |
|---|---|---|---|---|
| High-precision math: `tdiv`, `tdivs`, `trecip`, `tcolexpanddiv`, `trowexpanddiv`, `tlog`, `texp`, `tsqrt` | Mostly default-precision paths; some templates ignore or do not read `precisionType=high_precision`. | Legacy dispatches to `div_hp.py`, `exp_hp.py`, `sqrt_hp.py`, or template-local high-precision logic. | High when attr is used. | Port the helper algorithms and any required PTODSL primitive wrappers, then add attr-specific ST/lit coverage. |
| `tfmod`, `tfmods`, `trem`, `trems` f16 | Shared PTODSL remainder path uses direct half operations. | Legacy expands f16 even/odd halves through f32 and converts back. | Medium-high numerical parity risk. | Add the f16 widen/compute/narrow path or prove current outputs match tolerance for all ST/full cases. |
| `tfillpad_inplace` with column expansion | Smoke now passes no-expansion cases. | Legacy uses `vstus`/`vstas` for unaligned inplace column fill. | Medium-high for non-smoke inplace expansion. | PTODSL currently avoids overwriting valid data but does not implement the legacy unaligned store chain for inplace expansion. |
| `tfillpad` and `tfillpad_expand` physical padding edge cases | Smoke now passes after aligned-boundary fill and valid-tail restore. | Legacy handles physical row/column padding and alignment-specific stores. | Medium. | Run full ST and add targeted cases for partial valid shape, physical dst shape larger than valid shape, and all pad values. |
| `tinsert` | Only basic UB vec-to-vec ND and scalar basic paths are present. | Legacy has acc-to-mat, acc-to-vec ND/DN/NZ, vec-to-mat, NZ, quant/relu, split, and mode-driven variants. | High if those modes appear. | Port the missing mode families or prove VPTO/ST never emits them in this branch. |
| `tcvt` uncommon conversions | A small conversion subset is registered. | Legacy has a much larger dtype/version matrix, including low-precision formats. | Medium-high. | Add missing signatures incrementally from ST failures, then compare against the full legacy dtype matrix. |
| `tbinop` multi-candidate/tail selection | Some elementwise tests pass, but `tbinop.has_tail` is still a simplified selector. | Legacy has more version/tail-specific selection behavior. | Medium. | Validate `tadd`/`tmul` and related multi-candidate paths against full ST and generated VPTO. |
| Cube/GEMV/move families | Module coverage exists and some smoke cases pass. | Legacy templates encode many architecture-specific layout and memory-space cases. | Medium-high. | Treat render/catalog coverage as insufficient; run ST or targeted simulator validation for cube, GEMV, and specialized move paths. |

## Recently Fixed But Worth Rechecking In Full ST

| Template or family | Current result | Why still listed |
|---|---|---|
| `tpartadd`, `tpartmul`, `tpartmax`, `tpartmin` | Smoke passes after preserving `TileSpec.valid_shape`. | Full partial-valid-shape coverage should still be run because these are correctness-sensitive overlay operations. |
| `trowprod` | Smoke/full compile checks pass after dtype-specific collapse depth. | Full runtime compare should remain in the default validation set because the original bug was wrong-output risk, not just render failure. |
| `tload`, `tstore` downstream users | Smoke passes for fixed cases. | Metadata changes are shared by many ops; any new partition/tensor-view shape should be watched for selection regressions. |

## Suggested Priority

1. Fix ST build blockers with simple dtype/signature gaps: `trowsum`, `trowargmax`, `trowargmin`, `tsels`, `tsort32`, `tmrgsort`.
2. Fix known wrong-output cases: `tcmp` and `trandom`.
3. Fix core math coverage: `tdiv`, `tdivs`, `tcolexpanddiv`, `tlog`, then high-precision attr paths.
4. Expand structural coverage: `textract`, `textract_fp`, `tinsert`.
5. Revisit semantic watch items that smoke does not prove: f16 remainder, inplace fillpad expansion, `tcvt` full matrix, and cube/GEMV validation.
