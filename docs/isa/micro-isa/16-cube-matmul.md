# 16. Cube Matrix Multiply (MAT)

> **Category:** Cube unit ops — staged load/store and matrix multiply
> **Raw-op reference:** See `16-cube-matmul-raw.md` for low-level bridge/raw ops

---

## Wrapper-Layer Compute Ops

### `pto.mad`

- **syntax:**
```mlir
pto.mad %lhs, %rhs, %dst, %m, %n, %k
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, i64, i64, i64
```
- **semantics:** Zero-init cube matmul, `dst = lhs * rhs`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | L0A input (`left`) |
| `%rhs` | ptr | L0B input (`right`) |
| `%dst` | ptr | L0C accumulator (`acc`) |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%k` | i64 | K size |
| `unit_flag_ctrl` | i32 attr | Accumulator control flag |
| `disable_gemv` | bool attr | GEMV-disable control bit |

**Constraints:**

- Address spaces must be `left`, `right`, `acc`.
- `unit_flag_ctrl` currently uses `0/2/3` values in existing tests.

**Example:**

```mlir
pto.mad %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c16_i64
  : !pto.ptr<f16, left>, !pto.ptr<f16, right>, !pto.ptr<f32, acc>, i64, i64, i64
```

---

### `pto.mad_acc`

- **syntax:**
```mlir
pto.mad_acc %lhs, %rhs, %dst, %m, %n, %k
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, i64, i64, i64
```
- **semantics:** Accumulating cube matmul, `dst += lhs * rhs`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | L0A input (`left`) |
| `%rhs` | ptr | L0B input (`right`) |
| `%dst` | ptr | L0C accumulator (`acc`) |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%k` | i64 | K size |
| `unit_flag_ctrl` | i32 attr | Accumulator control flag |
| `disable_gemv` | bool attr | GEMV-disable control bit |

**Constraints:**

- Same address space/type family requirements as `pto.mad`.

**Example:**

```mlir
pto.mad_acc %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c16_i64 {unit_flag_ctrl = 2 : i32}
  : !pto.ptr<f16, left>, !pto.ptr<f16, right>, !pto.ptr<f32, acc>, i64, i64, i64
```

---

### `pto.mad_bias`

- **syntax:**
```mlir
pto.mad_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, !pto.ptr<C, bias>, i64, i64, i64
```
- **semantics:** Bias-init cube matmul, `dst = lhs * rhs + bias`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` / `%rhs` / `%dst` / `%m` / `%n` / `%k` | - | Same meaning as `pto.mad` |
| `%bias` | ptr | Bias-table pointer (`!pto.ptr<C, bias>`) |
| `unit_flag_ctrl` | i32 attr | Accumulator control flag |
| `disable_gemv` | bool attr | GEMV-disable control bit |

**Constraints:**

- `%bias` must be in `bias` address space.

**Example:**

```mlir
pto.mad_bias %l0a, %l0b, %l0c, %bt, %c16_i64, %c16_i64, %c16_i64
  : !pto.ptr<f16, left>, !pto.ptr<f16, right>, !pto.ptr<f32, acc>, !pto.ptr<f32, bias>, i64, i64, i64
```

---

### `pto.mad_mx`

- **syntax:**
```mlir
pto.mad_mx %lhs, %rhs, %dst, %m, %n, %k
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, i64, i64, i64
```
- **semantics:** Zero-init MX cube matmul.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | MX L0A input (`left`) |
| `%rhs` | ptr | MX L0B input (`right`) |
| `%dst` | ptr | L0C accumulator (`acc`) |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%k` | i64 | K size, typically matching MX tile granularity |
| `unit_flag_ctrl` | i32 attr | Accumulator control flag |
| `disable_gemv` | bool attr | GEMV-disable control bit |

**Constraints:**

- MX-capable dtype combinations must be respected by backend lowering.

**Example:**

```mlir
pto.mad_mx %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, left>, !pto.ptr<f8E4M3FN, right>, !pto.ptr<f32, acc>, i64, i64, i64
```

---

### `pto.mad_mx_acc`

- **syntax:**
```mlir
pto.mad_mx_acc %lhs, %rhs, %dst, %m, %n, %k
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, i64, i64, i64
```
- **semantics:** Accumulating MX cube matmul.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | MX L0A input (`left`) |
| `%rhs` | ptr | MX L0B input (`right`) |
| `%dst` | ptr | L0C accumulator (`acc`) |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%k` | i64 | K size, typically matching MX tile granularity |
| `unit_flag_ctrl` | i32 attr | Accumulator control flag |
| `disable_gemv` | bool attr | GEMV-disable control bit |

**Constraints:** same as `pto.mad_mx`.

**Example:**

```mlir
pto.mad_mx_acc %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, left>, !pto.ptr<f8E4M3FN, right>, !pto.ptr<f32, acc>, i64, i64, i64
```

---

### `pto.mad_mx_bias`

- **syntax:**
```mlir
pto.mad_mx_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, !pto.ptr<C, bias>, i64, i64, i64
```
- **semantics:** Bias-init MX cube matmul.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | MX L0A input (`left`) |
| `%rhs` | ptr | MX L0B input (`right`) |
| `%dst` | ptr | L0C accumulator (`acc`) |
| `%bias` | ptr | Bias-table pointer (`bias`) |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%k` | i64 | K size, typically matching MX tile granularity |
| `unit_flag_ctrl` | i32 attr | Accumulator control flag |
| `disable_gemv` | bool attr | GEMV-disable control bit |

**Constraints:** same as `pto.mad_mx` plus bias address-space requirement.

**Example:**

```mlir
pto.mad_mx_bias %l0a, %l0b, %l0c, %bt, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, left>, !pto.ptr<f8E4M3FN, right>, !pto.ptr<f32, acc>, !pto.ptr<f32, bias>, i64, i64, i64
```

---

## Cube Bridge Wrapper Ops

### `pto.cube_load`

- **syntax:**
```mlir
pto.cube_load %src, %dst, %len_burst
  nburst(%count, %src_stride, %dst_stride)
  [loop(%count_i, %src_stride_i, %dst_stride_i)]*
  : !pto.ptr<T, gm>, !pto.ptr<T, mat>, i64, i64, i64, i64
```
- **semantics:** Structured GM-to-L1 (`cbuf`) wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | GM source pointer |
| `%dst` | ptr | L1 destination pointer (`mat`) |
| `%len_burst` | i64 | Burst length |
| `nburst(%count, %src_stride, %dst_stride)` | i64 triple | Inner DMA burst count and strides |
| `loop(%count_i, %src_stride_i, %dst_stride_i)` | i64 triple | Optional outer loop triplet, repeatable |

**Constraints:**

- Wrapper lowers to loop/stride setup plus `pto.copy_gm_to_cbuf`.

**Example:**

```mlir
pto.cube_load %a_gm, %l1_a, %c16_i64
  nburst(%c1_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, mat>, i64, i64, i64, i64
```

---

### `pto.cube_store`

- **syntax:**
```mlir
pto.cube_store %src, %dst, %len_burst
  nburst(%count, %src_stride, %dst_stride)
  [loop(%count_i, %src_stride_i, %dst_stride_i)]*
  : !pto.ptr<T, mat>, !pto.ptr<T, ub>, i64, i64, i64, i64
```
- **semantics:** Structured L1 (`cbuf`) to UB wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`mat`) |
| `%dst` | ptr | UB destination pointer |
| `%len_burst` | i64 | Burst length |
| `nburst(%count, %src_stride, %dst_stride)` | i64 triple | Inner DMA burst count and strides |
| `loop(%count_i, %src_stride_i, %dst_stride_i)` | i64 triple | Optional outer loop triplet, repeatable |

**Constraints:**

- Wrapper lowers to `pto.copy_cbuf_to_ubuf` and optional outer loops.

**Example:**

```mlir
pto.cube_store %l1_src, %ub_dst, %c16_i64
  nburst(%c1_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f16, mat>, !pto.ptr<f16, ub>, i64, i64, i64, i64
```

---

### `pto.cube_load_frac`

- **syntax:**
```mlir
pto.cube_load_frac %src, %dst, nd2nz|dn2nz, shape(%n_value, %d_value), src_layout(%src_inner_stride[, %src_outer_stride]), dst_group(%group_count, %dst_loop2_stride, %dst_loop3_stride, %dst_loop4_stride), ctrl(%l2_cache_ctrl, %smallc0_en)
  : !pto.ptr<T, gm>, !pto.ptr<T, mat>, ...
```
- **semantics:** Structured fractal-load wrapper for `nd2nz` / `dn2nz`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | GM source pointer |
| `%dst` | ptr | L1 destination pointer (`mat`) |
| `nd2nz` / `dn2nz` | enum token | Fractal load mode |
| `shape(%n_value, %d_value)` | i64 pair | Logical N and D shape |
| `src_layout(%src_inner_stride[, %src_outer_stride])` | i64 / i64 pair | Source layout stride fields |
| `dst_group(%group_count, %dst_loop2_stride, %dst_loop3_stride, %dst_loop4_stride)` | i64 tuple | Destination group count and nested destination strides |
| `ctrl(%l2_cache_ctrl, %smallc0_en)` | i64, i1 | L2 cache control and small-C0 enable |

**Constraints:**

- Lowers to `set_mte2_nz_para` plus `copy_gm_to_cbuf_multi_*`.

**Example:**

```mlir
pto.cube_load_frac %src, %dst, nd2nz,
  shape(%n, %d),
  src_layout(%sis),
  dst_group(%g, %l2s, %l3s, %l4s),
  ctrl(%l2, %small)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, mat>, nd2nz, shape i64, i64, src_layout(i64), dst_group i64, i64, i64, i64, ctrl i64, i1
```

---

### `pto.bias_load`

- **syntax:**
```mlir
pto.bias_load %src, %dst, %len_burst
  nburst(%count, %src_gap, %dst_gap)
  : !pto.ptr<T, mat>, !pto.ptr<U, bias>, i64, i64, i64, i64
```
- **semantics:** Structured helper for L1 (`cbuf`) to bias-table load.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`mat`) |
| `%dst` | ptr | Bias destination pointer (`bias`) |
| `%len_burst` | i64 | Burst length |
| `%count` | i64 | Burst count |
| `%src_gap` | i64 | Source gap |
| `%dst_gap` | i64 | Destination gap |

**Constraints:**

- Supported type pairs: `f32->f32`, `i32->i32`, `f16->f32`, `bf16->f32`.

**Example:**

```mlir
pto.bias_load %l1_bias, %bt, %c16_i64 nburst(%c1_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f16, mat>, !pto.ptr<f32, bias>, i64, i64, i64, i64
```

---

### `pto.fp_load`

- **syntax:**
```mlir
pto.fp_load %src, %dst, %len_burst
  nburst(%count, %src_gap, %dst_gap)
  : !pto.ptr<T, mat>, !pto.ptr<U, scaling>, i64, i64, i64, i64
```
- **semantics:** Structured helper for L1 (`cbuf`) to Fixpipe Buffer (`scaling`) load.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`mat`) |
| `%dst` | ptr | Fixpipe-buffer destination pointer (`scaling`) |
| `%len_burst` | i64 | Burst length |
| `%count` | i64 | Burst count |
| `%src_gap` | i64 | Source gap |
| `%dst_gap` | i64 | Destination gap |

**Constraints:**

- Lowers to `pto.copy_cbuf_to_fbuf`.
- `%src` must be in `mat`, `%dst` must be in `scaling`.

**Example:**

```mlir
pto.fp_load %l1_fp, %fb_fp, %c2_i64 nburst(%c1_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f32, mat>, !pto.ptr<ui64, scaling>, i64, i64, i64, i64
```

---

### `pto.left_load`

- **syntax:**
```mlir
pto.left_load %src, %dst, %m, %k
  : !pto.ptr<T, mat>, !pto.ptr<T, left>, i64, i64
```
- **semantics:** Structured L1-to-L0A wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`mat`) |
| `%dst` | ptr | L0A destination pointer (`left`) |
| `%m` | i64 | M tile size |
| `%k` | i64 | K tile size |

**Constraints:**

- Lowers to `pto.load_cbuf_to_ca`.

**Example:**

```mlir
pto.left_load %l1_a, %l0a, %c16_i64, %c16_i64
  : !pto.ptr<f16, mat>, !pto.ptr<f16, left>, i64, i64
```

---

### `pto.right_load`

- **syntax:**
```mlir
pto.right_load %src, %dst, %k, %n
  : !pto.ptr<T, mat>, !pto.ptr<T, right>, i64, i64
```
- **semantics:** Structured L1-to-L0B wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`mat`) |
| `%dst` | ptr | L0B destination pointer (`right`) |
| `%k` | i64 | K tile size |
| `%n` | i64 | N tile size |

**Constraints:**

- Lowers to `pto.load_cbuf_to_cb`.

**Example:**

```mlir
pto.right_load %l1_b, %l0b, %c16_i64, %c16_i64
  : !pto.ptr<f16, mat>, !pto.ptr<f16, right>, i64, i64
```

---

### `pto.left_load_mx`

- **syntax:**
```mlir
pto.left_load_mx %src, %dst, %m, %k
  : !pto.ptr<T, mat>, !pto.ptr<T, left>, i64, i64
```
- **semantics:** MX-mode L1-to-L0A wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | MX-formatted L1 source pointer (`mat`) |
| `%dst` | ptr | MX L0A destination pointer (`left`) |
| `%m` | i64 | M tile size |
| `%k` | i64 | K tile size |

**Constraints:**

- Lowers to `pto.load_cbuf_to_ca_mx`.

**Example:**

```mlir
pto.left_load_mx %l1_a, %l0a, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, mat>, !pto.ptr<f8E4M3FN, left>, i64, i64
```

---

### `pto.right_load_mx`

- **syntax:**
```mlir
pto.right_load_mx %src, %dst, %k, %n
  : !pto.ptr<T, mat>, !pto.ptr<T, right>, i64, i64
```
- **semantics:** MX-mode L1-to-L0B wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | MX-formatted L1 source pointer (`mat`) |
| `%dst` | ptr | MX L0B destination pointer (`right`) |
| `%k` | i64 | K tile size |
| `%n` | i64 | N tile size |

**Constraints:**

- Lowers to `pto.load_cbuf_to_cb_mx`.

**Example:**

```mlir
pto.right_load_mx %l1_b, %l0b, %c64_i64, %c16_i64
  : !pto.ptr<f8E4M3FN, mat>, !pto.ptr<f8E4M3FN, right>, i64, i64
```

---

### `pto.acc_store`

`pto.acc_store*` 是结构化的 fixpipe 写回族，用来把 `pto.mad*` 产出的 L0C(`acc`) 结果写到不同目标空间。
从语义上看，这条流水按下面的顺序组织：

1. 读取 `%src` 指向的 L0C 累加结果，并按 `%m/%n` 解释逻辑输出区域。
2. 如指定 `unit_flag(...)`，在消费这次 L0C 结果前执行完成态检查；`check_and_clear` 还会在消费后清除该完成态，便于后续下一轮生产/消费配对。
3. 如指定 `pre_quant(%payload, mode = ...)`，先对 L0C 元素做预量化。这里的 `%payload` 是该量化模式所需的标量参数或 scaling 指针；标量模式允许直接传 `f16`、`bf16`、`f32`，向量模式要求 `scaling` 指针。
4. 如指定 `pre_relu(...)`，在写回前对结果做 ReLU 预处理。`scalar_relu`/`vector_relu` 需要额外 payload；`normal_relu`/`no_relu` 不接 payload。`scalar_relu` 允许直接传 `f16`、`bf16`、`f32` alpha，`vector_relu` 要求 `scaling` 指针。`clip = %clip` 是 `pre_relu(...)` 子句的一部分，用于在支持的目标元素类型上启用 clip 阶段。
5. 按 `nz2nd` / `nz2dn` / `nz2nz` 将 L0C 中的 NZ 累加布局转换成目标布局，并结合 `%src_stride`、`%dst_stride` 以及可选 `loop3(...)`/`%loop0_src_stride`/`%split` 控制跨 tile 的遍历方式。
6. 如指定 `sat`，则在最终写回目标元素类型时启用饱和语义。
7. 将结果写入目标空间。`acc_store` 写 L1(`mat`)，`acc_store_ub` 写 UB，`acc_store_gm` 写 GM；GM 路径还可额外指定原子更新语义。

- **syntax:**
```mlir
pto.acc_store %src, %dst, %m, %n, %src_stride, %dst_stride
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu(%payload, mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat]?
  : ...
```
- **semantics:** Structured L0C (`acc`) to L1 (`cbuf`) wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | buffer-like | L0C source buffer (`acc`)，可为 typed `!pto.ptr` 或等价 memref |
| `%dst` | buffer-like | L1 destination buffer (`mat`)，可为 typed `!pto.ptr` 或等价 memref |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%src_stride` | i64 | 源 NZ 布局在 fixpipe 写回过程中的主 stride 参数 |
| `%dst_stride` | i64 | 目标布局在 fixpipe 写回过程中的主 stride 参数 |
| `unit_flag(...)` | optional clause | 是否在消费这次 L0C 结果前检查完成态；`check_and_clear` 还会在消费后清除完成态 |
| `pre_quant(%payload, mode = ...)` | optional clause | 写回前的预量化；payload 为该量化模式需要的浮点标量或 `scaling` 指针 |
| `pre_relu(..., mode = ...[, clip = %clip])` | optional clause | 写回前的 ReLU 预处理；`clip` 只能作为 `pre_relu` 的一部分出现 |
| `nz2nd` / `nz2dn(%loop0_src_stride)` / `nz2nz(%split)?` | mode clause | L0C NZ 布局到目标布局的写回模式 |
| `loop3(%count, %src_stride3, %dst_stride3)` | optional i64 triple | 额外的外层重复写回控制，用于跨 tile 迭代 |
| `sat` | optional flag | 最终写回到目标元素类型时启用饱和语义 |

**Constraints:**

- 子句顺序固定为 `unit_flag` -> `pre_quant` -> `pre_relu` -> `mode` -> `loop3` -> `sat`。
- `pre_quant` 必须同时提供 payload 和 `mode`。
- `pre_quant` 仅支持 L0C 源元素类型为 `f32` 或 `i32`。
- 标量 `pre_quant` 模式要求 payload 为 `f16`/`bf16`/`f32` 标量；其中 `f16`/`bf16` 会在 lowering 中先扩成 `f32`，再以 32-bit 浮点 bit pattern 形式配置到 fixpipe 标量参数寄存器。向量 `pre_quant` 模式要求 `scaling !pto.ptr<ui64>` payload。
- `pre_relu` 的 payload 规则取决于 `mode`：
  - `no_relu` / `normal_relu` 不接受 payload。
  - `scalar_relu` 要求 payload 为 `f16`/`bf16`/`f32` 标量；其中 `f16`/`bf16` 会在 lowering 中先扩成 `f32`，再以 32-bit 浮点 bit pattern 形式配置到 fixpipe 标量参数寄存器。
  - `vector_relu` 要求 `scaling !pto.ptr<ui64>` payload。
- `clip` 只能出现在 `pre_relu(...)` 中。
- `clip` 仅支持目标元素类型为 `f16`、`ui8`、或有符号 `i4/i8/i16`。
- `clip` payload 必须与目标元素类型匹配：
  - `f16` 目标要求 `f16` payload。
  - `ui8` 目标要求 `ui16` 风格的 16-bit payload；在 PTO IR 中通常写成 `signless i16`。
  - 有符号 `i4/i8/i16` 目标要求有符号或 signless 的 `i4/i8/i16` payload。
- `loop3(...)` 必须三个操作数同时提供。
- `nz2dn` 必须提供 `%loop0_src_stride`；`nz2nd`/`nz2nz` 不接受它。
- 当 `nz2dn(%loop0_src_stride)` 中 `%loop0_src_stride != 1` 时，`unit_flag` 必须关闭。
- `nz2nz` 不接受 `loop3(...)`，且目标元素类型必须是 `f32`。

**Example:**

```mlir
pto.acc_store %l0c, %l1_out, %c16_i64, %c16_i64, %c16_i64, %c16_i64, nz2dn(%c64_i64), loop3(%c3_i64, %c4_i64, %c5_i64)
  : !pto.ptr<f32, acc>, !pto.ptr<f32, mat>, i64, i64, i64, i64, i64, i64, i64, i64
```

---

### `pto.acc_store_gm`

- **syntax:**
```mlir
pto.acc_store_gm %src, %dst, %m, %n, %src_stride, %dst_stride, %sid, %l2_cache_ctrl
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu(%payload, mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat]?
    [, atomic(type = <atomic_type>, op = <atomic_op>)]?
  : ...
```
- **semantics:** Structured L0C (`acc`) to GM wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | buffer-like | L0C source buffer (`acc`)，可为 typed `!pto.ptr` 或等价 memref |
| `%dst` | buffer-like | GM destination buffer，可为 typed `!pto.ptr` 或等价 memref |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%src_stride` | i64 | 源 NZ 布局在 fixpipe 写回过程中的主 stride 参数 |
| `%dst_stride` | i64 | 目标布局在 fixpipe 写回过程中的主 stride 参数 |
| `%sid` | i64 | GM 写回使用的 stream/session 标识参数 |
| `%l2_cache_ctrl` | i64 | GM 路径的 L2 cache 策略参数 |
| (optional clauses) | — | 与 `pto.acc_store` 相同的语义子句，外加 GM 独有的 `atomic(...)` |

**Constraints:**

- GM output path controls (`sid`, `l2_cache_ctrl`) must be provided.
- `atomic(type = ..., op = ...)` 只允许出现在 `pto.acc_store_gm`。
- `atomic` 必须同时提供 `type` 和 `op`。
- 当前 `op` 取值为 `add` / `max` / `min`；`type` 取值为 `f32` / `f16` / `bf16` / `s32` / `s16` / `s8`。

**Example:**

```mlir
pto.acc_store_gm %l0c, %c_gm, %c16_i64, %c16_i64, %c16_i64, %c16_i64, %c0_i64, %c0_i64, nz2nd
  : !pto.ptr<f32, acc>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
```

---

### `pto.acc_store_ub`

- **syntax:**
```mlir
pto.acc_store_ub %src, %dst, %m, %n, %src_stride, %dst_stride, %dual_dst_mode, %sub_blockid
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu(%payload, mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat]?
  : ...
```
- **semantics:** Structured L0C (`acc`) to UB wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | buffer-like | L0C source buffer (`acc`)，可为 typed `!pto.ptr` 或等价 memref |
| `%dst` | buffer-like | UB destination buffer，可为 typed `!pto.ptr` 或等价 memref |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%src_stride` | i64 | 源 NZ 布局在 fixpipe 写回过程中的主 stride 参数 |
| `%dst_stride` | i64 | 目标布局在 fixpipe 写回过程中的主 stride 参数 |
| `%dual_dst_mode` | i64 | UB 路径的双目标写回模式参数 |
| `%sub_blockid` | i64 | UB 路径的子块选择参数 |
| (optional clauses) | — | 与 `pto.acc_store` 相同，但不支持 `atomic(...)` |

**Constraints:**

- 不支持 `atomic(...)`。

**Example:**

```mlir
pto.acc_store_ub %l0c, %ub_out, %c16_i64, %c16_i64, %c16_i64, %c16_i64, %c0_i64, %c0_i64, nz2nd
  : !pto.ptr<f32, acc>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64
```

---

## Current PTOAS Coverage

- VPTO->LLVM (`--vpto-emit-hivm-llvm`) lowers this chapter's ops to
  `llvm.hivm.*` intrinsics with cube-related address spaces.
