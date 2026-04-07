# Tile Lib 向量库方案设计

## 第一章 背景与问题

### 1.1 当前编译栈与编译时长问题

当前从 DSL 到硬件二进制的完整编译栈如下：

```
PTO DSL (TileLang...)
       ↓
     PTOAS (MLIR)
       ↓
  Tile Lib (CCE)          ← C++ 模板库
       ↓
     CCEC                 ← C++ 编译器
       ↓
    LLVM IR
       ↓
    BiSheng
       ↓
  Davinci Binary
```

这条编译栈层次较深。PTOAS 生成 CCE C++ 代码后，需要经过 C++ 模板实例化和 CCEC 编译才能产生 LLVM IR，再由 BiSheng 编译器生成最终的 Davinci 二进制。其中 **C++ 模板实例化和 CCE 编译** 是主要的编译时间瓶颈。

我们希望简化编译栈，跳过 CCE 代码生成和编译的过程，直接从 PTOAS 输出 LLVM IR：

```
PTO DSL (TileLang...) + Tile Lib
       ↓
     PTOAS (MLIR)         ← 直接输出 LLVM IR，跳过 CCE
       ↓
    LLVM IR
       ↓
    BiSheng
       ↓
  Davinci Binary
```

这样可以显著缩短编译时间。但当前的 Tile Lib 是基于 CCE 和 C++ 模板开发的，因此需要用其它方式重新实现 Tile Lib。

### 1.2 PTOAS 中向量库实现的挑战

PTOAS 中目前设计两层粒度的 IR：

- **PTO TileOp**：面向上层用户的高层抽象，操作对象是 `tile_buf`，一条指令表达完整的 tile 语义（如 `pto.tadd`、`pto.tmul`、`pto.tload`）。
- **Vector IR (vPTO)**：面向底层硬件的指令接口，操作对象是 `vreg`/`ptr`，需要显式循环、显式寄存器宽度、显式 mask 处理（如 `pto.vadd`、`pto.vlds`、`pto.vsts`）。

Tile Lib 的一种实现方式是直接使用 Vector IR 编写。以 `pto.tadd`（逐元素加法）为例，在 TileOp层只需一条指令：

```mlir
pto.tadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>)
```

而用 Vector IR 实现同样的语义（`dtype=f32, rows=16, cols=64`），需要展开为完整的向量循环：

```mlir
func.func @TADD(
    %a: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %b: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %c: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    attributes { pto.tile_function = "pto.tadd" } {
  %vecA = pto.tile_buf_addr %a : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
      v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %vecB = pto.tile_buf_addr %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
      v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %vecC = pto.tile_buf_addr %c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
      v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index

  pto.vecscope {
    scf.for %arg0 = %c0 to %c16 step %c1 {           // 遍历 rows
      scf.for %arg1 = %c0 to %c64 step %c64 {         // 遍历 cols，步长=vector_width
        %va = pto.vlds %vecA[%arg0, %arg1] : memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vb = pto.vlds %vecB[%arg0, %arg1] : memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vc = pto.vadd %va, %vb : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>
        pto.vsts %vc, %vecC[%arg0, %arg1] : !pto.vreg<64xf32>, memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
      }
    }
  }
  return
}
```

直接基于 Vector IR 开发 Tile Lib 面临以下困难：

1. **MLIR 语法门槛高**：需要熟悉 `memref`、`index`、`strided` 等 MLIR 数据类型和语法，使用 MLIR 的方式定义变量、表达运算和控制流。
2. **参数组合无法穷举**：`dtype` 有 f16/f32/bf16 等，`rows`/`cols` 可以是任意正整数，为每种 `(op, dtype, rows, cols, layout)` 组合手写向量实现不可行。

因此，直接基于 PTO Vector IR 开发 Tile Lib，技术难度大且工作无法收敛。

## 第二章 方案：使用 Python 开发 Tile Lib

### 2.1 总体思路

为了降低开发门槛并解决参数组合的穷举问题，我们采用 PTO DSL 来编写 Tile Lib 的向量库实现。这套语法定义在 TileLang 中，库开发者使用 Python 编写模板函数，由 PTOAS 编译器在编译时进行实例化。

整体方案：

1. **用 Python DSL 编写模板函数**：使用 `pto.Tile` 数据类型和向量操作接口，按 Tile 指令语义编写向量实现。
2. **编译器实例化模板**：PTOAS 在编译过程中遇到 Tile op 时，调用对应的模板函数，填入具体的 `tile_buf` 类型参数，生成特化后的向量 IR。
3. **inline 到调用点**：特化后的向量 IR 直接 inline 到原 Tile op 的位置，继续后续优化和 lowering 流程。

### 2.2 TADD 模板示例

以 `pto.tadd`（逐元素加法）为例，使用 Python DSL 编写的模板函数如下：

```python
@pto.tile_template(target="a5", op="pto.tadd")
def template_tadd(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = src0.element_type
    elem_size = src0.element_size
    rows, cols = src0.shape
    v_rows, v_cols = src0.valid_shape

    for i in range(0, v_rows, 1):
        remaining = v_cols
        for j in range(0, v_cols, 256 / elem_size):
            all_mask, remaining = pto.make_mask(dtype, remaining)
            vec_a = pto.vlds(a[i, j])
            vec_b = pto.vlds(b[i, j])
            result = pto.vadd(vec_a, vec_b, all_mask)
            pto.vsts(result, c[i, j], all_mask)
```

代码解读：

- **`@pto.tile_template`** 装饰器指示这是一个 `pto.tadd` 指令的模板，会在编译时进行实例化。
- **输入参数**为 3 个 `pto.Tile` 数据类型参数，2 个输入（`src0`、`src1`），1 个输出（`dst`）。
- 通过 **`Tile` 数据类型接口**获取元素类型（`element_type`）、元素大小（`element_size`）、静态 shape（`shape`）和 valid shape（`valid_shape`）信息。
- 通过 **2 层循环**分别遍历 tile 的行和列。
- 通过 **`pto.make_mask`** 指令，根据基础数据类型大小及有效数据数量设置 mask 寄存器。
- 通过 **`pto.vlds`** 指令，以 `a[i, j]` 和 `b[i, j]` 为起始地址分别将数据读入向量寄存器。
- 通过 **`pto.vadd`** 计算相加结果，写入寄存器 `result`。
- 通过 **`pto.vsts`** 将 `result` 写入以 `c[i, j]` 为起始的地址区间。

### 2.3 值模型与 Staging 语义

模板函数中使用的 `pto.Tile` 属性，在模板执行时分为两类不同阶段（stage）的值：

#### 编译期静态值（Compile-time Static）

以下属性在模板实例化时已经确定，由 Python Codegen 在编译期折叠为字面量，**不会**生成 MLIR SSA 值：

| 属性 | 来源 | 说明 |
|------|------|------|
| `element_type` | `tile_buf` 的 `dtype` 字段 | 决定 vreg 类型和向量宽度 |
| `element_size` | 由 `dtype` 推导 | f32→4, f16→2, i8→1 |
| `shape` | `tile_buf` 的 `rows`, `cols` 字段 | **必须是编译期静态值**，参与模板实例化的 specialization key |
| `config` | `tile_buf` 的 blayout/slayout/fractal/pad | 布局和配置信息 |

这些值在 Python 层直接参与运算（如 `256 / elem_size`），结果在编译期确定。

#### 运行时 SSA 值（Runtime Dynamic）

以下属性可能在编译期未知，生成为 MLIR 函数参数或 SSA 值：

| 属性 | 来源 | 说明 |
|------|------|------|
| `valid_shape` | `tile_buf` 的 `v_row`, `v_col` 字段 | **可以是静态也可以是动态** |

当 `valid_shape` 为静态值时，Python Codegen 在编译期折叠（与 `shape` 相同处理方式）；当为动态值时，生成为 MLIR 函数参数（`index` 类型），循环边界等依赖它的地方生成 `scf.for`。

#### 正式约束

1. **`shape` 必须是编译期静态值**，并参与模板实例化的 specialization key。如果 `shape` 为动态值，模板实例化应报错拒绝。
2. **`valid_shape` 可以是静态也可以是动态**。当为静态值时，Python Codegen 侧应检查 `valid_shape <= shape`（逐维度）。
3. **`element_type`、`element_size`、`config` 必须是编译期静态值**，它们决定了模板函数体的结构（vreg 类型、向量宽度、stride 模式等）。

#### 对控制流的影响

```python
rows, cols = src0.shape           # 编译期静态 → Python 层直接展开或折叠
v_rows, v_cols = src0.valid_shape # 可能是动态 → 生成 scf.for

for i in range(0, v_rows, 1):    # v_rows 动态 → scf.for %i = 0 to %v_rows
    for j in range(0, v_cols, 64): # v_cols 动态 → scf.for %j = 0 to %v_cols step 64
        ...

# 对比：如果用 shape（静态），Python 层可以直接展开
for i in range(0, rows, 1):       # rows=16 静态 → Python 展开 16 次迭代
    ...
```

### 2.4 TileLang DSL 语法参考

#### 2.4.1 基础数据类型

| DSL 类型 | 说明 | 位宽 |
|----------|------|------|
| `pto.i8` | 8 位整数 | 8 |
| `pto.i16` | 16 位整数 | 16 |
| `pto.i32` | 32 位整数 | 32 |
| `pto.i64` | 64 位整数 | 64 |
| `pto.f16` | 半精度浮点 | 16 |
| `pto.bf16` | BFloat16 | 16 |
| `pto.f32` | 单精度浮点 | 32 |

Python 字面量自动推导类型：`int` → `pto.i32`，`float` → `pto.f32`。

#### 2.4.2 Tile 数据类型

`pto.Tile` 表示一个带有布局和配置信息的数据块，对应 MLIR 中的 `!pto.tile_buf` 类型。

**Tile 属性接口：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `shape` | `tuple[int, ...]` | Tile 的完整维度（rows, cols） |
| `valid_shape` | `tuple[int, ...]` | 有效数据维度（v_row, v_col），可能小于 shape |
| `element_type` | `Type` | 元素数据类型（如 `pto.f32`） |
| `element_size` | `int` | 元素字节大小（如 f32 → 4） |
| `memory_space` | `MemorySpace` | 内存空间（GM, UB） |
| `config` | `TileConfig` | 布局和 padding 配置 |

**Tile 配置：**

```python
pto.BLayout.ROW_MAJOR     # 行主序
pto.BLayout.COL_MAJOR     # 列主序
pto.SLayout.NONE_BOX      # 无二级布局
pto.PadValue.NULL          # 无 padding
pto.PadValue.ZERO          # 零填充
```

#### 2.4.3 向量操作接口

向量寄存器固定 256 字节宽度，每次处理的元素数量由数据类型决定：f32 → 64 个元素，f16 → 128 个元素。

**Mask 操作：**

| 操作 | 说明 |
|------|------|
| `pto.make_mask(dtype, remaining)` | 根据数据类型和剩余元素数量生成 mask，返回 `(mask, new_remaining)` |
| `pto.make_mask(dtype, PAT.ALL)` | 生成全 1 mask |

**向量 Load/Store：**

| 操作 | 说明 |
|------|------|
| `pto.vlds(tile[i, j])` | 从 Tile 的 `[i, j]` 位置加载一个向量寄存器的数据 |
| `pto.vsts(vec, tile[i, j], mask)` | 将向量寄存器数据写入 Tile 的 `[i, j]` 位置 |

**二元向量运算：**

| 操作 | 说明 |
|------|------|
| `pto.vadd(vec1, vec2, mask)` | 逐元素加法 |
| `pto.vsub(vec1, vec2, mask)` | 逐元素减法 |
| `pto.vmul(vec1, vec2, mask)` | 逐元素乘法 |
| `pto.vdiv(vec1, vec2, mask)` | 逐元素除法 |
| `pto.vmax(vec1, vec2, mask)` | 逐元素取大 |
| `pto.vmin(vec1, vec2, mask)` | 逐元素取小 |

**一元向量运算：**

| 操作 | 说明 |
|------|------|
| `pto.vabs(vec, mask)` | 逐元素绝对值 |
| `pto.vexp(vec, mask)` | 逐元素指数 |
| `pto.vln(vec, mask)` | 逐元素对数 |
| `pto.vsqrt(vec, mask)` | 逐元素开方 |
| `pto.vrelu(vec, mask)` | 逐元素 ReLU |

**向量-标量运算：**

| 操作 | 说明 |
|------|------|
| `pto.vmuls(vec, scalar, mask)` | 向量乘标量 |
| `pto.vadds(vec, scalar, mask)` | 向量加标量 |

#### 2.4.4 控制流

**循环**使用 Python 的 `range` 语法：

```python
for i in range(0, v_rows, 1):
    # 循环体
```

当循环边界来自 `shape`（编译期常量）时，DSL 在 Python 层展开循环；当来自 `valid_shape`（可能是运行时动态值）时，生成 `scf.for` MLIR 循环。

## 第三章 PTOAS 编译器：TileOp Expand

### 3.1 编译流程

PTOAS 编译器的输入可以是 Tile 指令、向量指令、或两者的混合。完整的编译 pipeline 如下：

```
输入：TileOp / 向量指令 / TileOp + 向量指令混合
       ↓
  VF Fusion Analysis        ← 在 TileOp 层分析可融合的操作组
       ↓
  PlanMemory                ← UB 内存分配规划
       ↓
  InsertSync                ← 管线同步插入
       ↓
  Expand TileOp             ← 将 TileOp 替换为对实例化模板函数的调用
       ↓
  Inline                    ← 将模板函数体 inline 到调用点
       ↓
  Fold TileBuf Intrinsics   ← 折叠 tile_buf_addr / tile_valid_rows / tile_valid_cols
       ↓
  VF Fusion                 ← 合并相邻向量循环，消除中间 UB 读写
       ↓
  LLVM IR
```

Tile 指令到向量指令的展开由三个 pass 协作完成：

1. **Expand TileOp**：核心 pass。调用 TileLang Python DSL 实例化模板库，生成以 `tile_buf` 为参数的向量实现函数，将原 Tile op 替换为对该函数的 `func.call`。
2. **Inline**：将模板函数体 inline 到调用点，使模板函数的 `tile_buf` 形参与调用点的实际 `tile_buf` 值绑定。
3. **Fold TileBuf Intrinsics**：折叠 inline 后留下的 `pto.tile_buf_addr`、`pto.tile_valid_rows`、`pto.tile_valid_cols` 等 intrinsic，将 `tile_buf` 的静态属性（地址、shape、布局）折叠为具体的 memref 和常量。

### 3.2 Expand TileOp Pass 的工作流程

以编译时遇到 `pto.tadd` 为例，Expand TileOp pass 的处理步骤如下：

```
Step 1: 识别 Tile Op
───────────────────
  遍历函数体中所有 Tile op（pto.tadd, pto.tsub, ...）
  遇到 pto.tadd ins(%a, %b) outs(%c)
  从所有操作数的 tile_buf 类型提取属性：
    dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
    blayout=row_major, slayout=none_box, fractal=512, pad=0

Step 2: 构造 Specialization Key + 查询缓存
──────────────────────────────────────────
  根据 Tile op 的所有操作数构造 specialization key（见 3.2.1）
  查询实例化缓存：
    如果缓存命中，直接复用已实例化的函数，跳到 Step 4

Step 3: 实例化模板（缓存未命中时执行）
─────────────────────────────────────
  调用 TileLang Python DSL，传入 op 名称和各操作数的 tile_buf 类型信息
  Python DSL 查找匹配的 @vkernel 模板，填入具体 tile_buf 参数进行特化
  输出实例化后的 MLIR 函数（以 tile_buf 为参数，内含向量循环体）
  解析 MLIR 文本，克隆函数到目标 Module，写入缓存

Step 4: 生成调用并替换原 Tile Op
───────────────────────────────
  在原 Tile op 位置插入 func.call @__pto_tilelang_tadd_f32_16_64(%a, %b, %c)
  操作数直接传递（类型均为 tile_buf，无需桥接转换）
  删除原 Tile op
```

#### 3.2.1 Specialization Key 与缓存

模板展开本质上是一个特化过程。当同一个 module 中存在多个相同类型的 Tile op（如多处 `pto.tadd` 且所有 `tile_buf` 操作数类型完全相同），应复用已实例化的结果而非重复展开。

**重要**：SpecKey 必须基于 **所有操作数** 的 `tile_buf` 类型构建，而不仅仅是第一个操作数。因为同一个 op 的不同操作数可能有不同的类型（如不同的 dtype 或 shape），仅用第一个操作数无法区分这些情况。

Expand TileOp pass 维护一个实例化缓存，key 包含以下字段：

| Key 字段 | 说明 |
|----------|------|
| `op_name` | Tile op 名称（如 `tadd`） |
| `operand_types` | **所有操作数**的 tile_buf 类型签名，每个操作数包含以下信息 |
| ├─ `dtype` | 元素数据类型（如 `f32`） |
| ├─ `shape` | Tile 的静态 shape（如 `(16, 64)`） |
| └─ `config` | blayout、slayout、fractal、pad 等配置 |

`valid_shape` **不参与** key——因为它可能是动态的，作为运行时值在 inline 后通过 `pto.tile_valid_rows`/`pto.tile_valid_cols` 提取。相同 `(op, operand_types)` 但不同 `valid_shape` 的 Tile op 可以共享同一份实例化结果。

#### 3.2.2 模板实例化过程

Expand TileOp 通过调用 Python 子进程来实例化模板。具体流程：

1. **调用 Python helper**：`python3 -m tilelang_dsl.expand_helper`，传入 op 名称、各操作数的 dtype/shape/memory_space 等参数。
2. **Python 端处理**：
   - 扫描模板目录下的 `.py` 文件，查找标注了 `@pto.vkernel` 装饰器的模板函数
   - 按 `op` 名称和 `dtype` 签名匹配模板
   - 对所有 `pto.Tile` 参数使用给定的 shape 和 memory_space 进行特化
   - 输出特化后的 MLIR 文本
3. **C++ 端处理**：
   - 解析 MLIR 文本为 `ModuleOp`
   - 提取 `func.func`，克隆到目标 Module 末尾
   - 重命名为 `__pto_tilelang_<op>_<dtype>_<dim0>_<dim1>`（如 `__pto_tilelang_tadd_f32_16_64`），设为 `private` 可见性
   - 存入 specCache

**关键约束**：Python DSL 实例化输出的函数需要满足以下要求：

1. **参数类型为 `!pto.tile_buf`**，而非 memref。DSL 在实例化时将具体的元素类型、静态 shape、布局配置等信息编码进 `tile_buf` 类型参数。
2. **函数必须带有 `pto.tilelang.instance` 属性**（UnitAttr）。Inline pass 通过此属性识别需要内联的模板实例函数，而非依赖函数名前缀。

函数体内部通过以下 intrinsic 从 `tile_buf` 中提取信息：

| Intrinsic | 功能 | 输出类型 |
|-----------|------|----------|
| `pto.tile_buf_addr` | 从 tile_buf 提取数据区域的 memref 指针 | `memref<RxCxdtype, strided<...>, #pto.address_space<...>>` |
| `pto.tile_valid_rows` | 从 tile_buf 提取有效行数 | `index` |
| `pto.tile_valid_cols` | 从 tile_buf 提取有效列数 | `index` |

这样设计的好处是：Expand TileOp pass 的调用点不需要做任何类型桥接，直接将 `tile_buf` 操作数透传给实例化的函数。类型转换和属性提取的工作统一在后续的 Fold pass 中处理。

### 3.3 实例化模板函数的 IR 结构

TileLang DSL 实例化后，生成的 MLIR 函数结构如下（以 `pto.tadd`、`dtype=f32`、`shape=(16,64)` 为例）：

```mlir
func.func @__pto_tilelang_tadd_f32_16_64(
    %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=?, v_col=?,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=?, v_col=?,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %dst:  !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=?, v_col=?,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    attributes { pto.tilelang.instance }
  {

  // 1. 从 tile_buf 提取 memref 地址
  %mSrc0 = pto.tile_buf_addr %src0 : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %mSrc1 = pto.tile_buf_addr %src1 : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %mDst  = pto.tile_buf_addr %dst  : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>

  // 2. 从 tile_buf 提取有效形状（inline 后由 Fold pass 折叠为常量或绑定到实际动态值）
  %v_rows = pto.tile_valid_rows %src0 : ... -> index
  %v_cols = pto.tile_valid_cols %src0 : ... -> index
  %v_cols_i32 = arith.index_cast %v_cols : index to i32  // plt_b32 需要 i32

  // 3. dtype=f32 → vector_width=64（256B / 4B），这是在模板实例化时固化的常量
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  // 4. 向量循环体：按行遍历，按 vreg 宽度分块，带尾部 mask
  pto.vecscope {
    scf.for %i = %c0 to %v_rows step %c1 {
      scf.for %j = %c0 to %v_cols step %c64 iter_args(%remain = %v_cols_i32) -> (i32) {
        %mask, %next = pto.plt_b32 %remain : i32 -> !pto.mask, i32

        %va = pto.vlds %mSrc0[%i, %j]
            : memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vb = pto.vlds %mSrc1[%i, %j]
            : memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vc = pto.vadd %va, %vb, %mask
            : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
        pto.vsts %vc, %mDst[%i, %j], %mask
            : !pto.vreg<64xf32>, memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>, !pto.mask

        scf.yield %next : i32
      }
    }
  }
  return
}
```

当 `valid_shape` 为静态已知值时（如 `v_row=16, v_col=64`），`pto.tile_valid_rows`/`pto.tile_valid_cols` 在 Fold pass 中会被直接折叠为常量 `arith.constant 16 : index`。当 `valid_shape` 为动态值时（`v_row=?, v_col=?`），Fold pass 将其替换为调用点传入的实际动态 `index` 值。

### 3.4 三个 Pass 的输入/输出示例

以下展示一个完整的 `pto.tadd` 从 TileOp 到向量 IR 的变换过程。

#### 3.4.1 输入（TileOp）

```mlir
func.func @TADD(
    %a: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %b: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %c: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>) {
  pto.tadd ins(%a, %b : ...) outs(%c : ...)
  return
}
```

#### 3.4.2 经过 Expand TileOp 后

`pto.tadd` 被替换为 `func.call`，操作数直接传递（类型不变）：

```mlir
func.func @TADD(%a: !pto.tile_buf<...>, %b: !pto.tile_buf<...>, %c: !pto.tile_buf<...>) {
  // pto.tadd 被替换为函数调用，tile_buf 直接透传
  call @__pto_tilelang_tadd_f32_16_64(%a, %b, %c) : (...) -> ()
  return
}

// TileLang DSL 实例化的模板函数（参数为 tile_buf 类型，带 pto.tilelang.instance 属性）
func.func private @__pto_tilelang_tadd_f32_16_64(
    %src0: !pto.tile_buf<...>, %src1: !pto.tile_buf<...>, %dst: !pto.tile_buf<...>)
    attributes { pto.tilelang.instance } {
  %mSrc0 = pto.tile_buf_addr %src0 : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %mSrc1 = pto.tile_buf_addr %src1 : ...
  %mDst  = pto.tile_buf_addr %dst  : ...
  %v_rows = pto.tile_valid_rows %src0 : ... -> index
  %v_cols = pto.tile_valid_cols %src0 : ... -> index
  %v_cols_i32 = arith.index_cast %v_cols : index to i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  pto.vecscope {
    scf.for %i = %c0 to %v_rows step %c1 {
      scf.for %j = %c0 to %v_cols step %c64 iter_args(%remain = %v_cols_i32) -> (i32) {
        %mask, %next = pto.plt_b32 %remain : i32 -> !pto.mask, i32
        %va = pto.vlds %mSrc0[%i, %j] : memref<...> -> !pto.vreg<64xf32>
        %vb = pto.vlds %mSrc1[%i, %j] : memref<...> -> !pto.vreg<64xf32>
        %vc = pto.vadd %va, %vb, %mask : ...
        pto.vsts %vc, %mDst[%i, %j], %mask : ...
        scf.yield %next : i32
      }
    }
  }
  return
}
```

#### 3.4.3 经过 Inline 后

模板函数体被 inline 到 `@TADD` 函数中，形参 `%src0`/`%src1`/`%dst` 与实参 `%a`/`%b`/`%c` 绑定：

```mlir
func.func @TADD(%a: !pto.tile_buf<...>, %b: !pto.tile_buf<...>, %c: !pto.tile_buf<...>) {
  // inline 后，tile_buf_addr / tile_valid_rows / tile_valid_cols 的操作数
  // 绑定到调用点的实际 tile_buf 值
  %mA = pto.tile_buf_addr %a : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %mB = pto.tile_buf_addr %b : ...
  %mC = pto.tile_buf_addr %c : ...
  %v_rows = pto.tile_valid_rows %a : ... -> index
  %v_cols = pto.tile_valid_cols %a : ... -> index
  %v_cols_i32 = arith.index_cast %v_cols : index to i32
  ...
  pto.vecscope {
    scf.for %i = %c0 to %v_rows step %c1 { ... }
  }
  return
}
```

#### 3.4.4 经过 Fold TileBuf Intrinsics 后

Fold pass 将 `pto.tile_buf_addr`、`pto.tile_valid_rows`、`pto.tile_valid_cols` 替换为具体值：

- `pto.tile_buf_addr %a` → 折叠为调用点已知的 memref 值（从 tile_buf 提取底层地址）
- `pto.tile_valid_rows %a` → 如果 `v_row=16` 是静态的，折叠为 `arith.constant 16 : index`；如果是动态的（`v_row=?`），折叠为调用点传入的动态 index 值
- `pto.tile_valid_cols %a` → 同理

折叠后得到最终的纯向量 IR，不再包含任何 tile_buf 引用：

```mlir
func.func @TADD(
    %a: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %b: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %c: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>) {

  %vecA = pto.tile_buf_addr %a : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %vecB = pto.tile_buf_addr %b : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %vecC = pto.tile_buf_addr %c : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index    // ← tile_valid_rows 折叠为常量
  %c64 = arith.constant 64 : index    // ← tile_valid_cols 折叠为常量
  %c64_i32 = arith.constant 64 : i32

  pto.vecscope {
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c64 step %c64 iter_args(%remain = %c64_i32) -> (i32) {
        %mask, %next = pto.plt_b32 %remain : i32 -> !pto.mask, i32
        %va = pto.vlds %vecA[%i, %j]
            : memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vb = pto.vlds %vecB[%i, %j]
            : memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vc = pto.vadd %va, %vb, %mask
            : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
        pto.vsts %vc, %vecC[%i, %j], %mask
            : !pto.vreg<64xf32>, memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>, !pto.mask
        scf.yield %next : i32
      }
    }
  }
  return
}
```

这与 1.2 节中手写的 Vector IR 实现等价，但由 Python DSL 模板自动生成。

### 3.5 模板目录与部署

TileLang DSL 模板文件（`.py`）部署在 PTOAS 工程的固定位置：

```
lib/TileOp/                     ← 模板库根目录
├── tadd_template.py            ← pto.tadd 的模板
├── tsub_template.py            ← pto.tsub 的模板
├── tmul_template.py            ← pto.tmul 的模板
└── ...
```

`tilelang_dsl` Python 包在安装后位于固定的 Python 包路径下。Expand TileOp pass 无需额外的 CLI 选项指定路径——模板目录和包路径在编译器构建时确定。

### 3.6 添加新算子的模板

在模板目录下创建 `.py` 文件，使用 `@pto.vkernel` 装饰器定义模板：

```python
@pto.vkernel(
    op="pto.<op_name>",           # 匹配的 Tile 算子名
    dtypes=[(<dtype>, ...)],      # 支持的 dtype 签名
    advanced=True,                # 启用隐式 vecscope 推断
    name="template_<op_name>",
)
def template_xxx(dst: pto.Tile, src0: pto.Tile, ...):
    # 向量化实现体
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            # ... 向量操作 ...
    return None
```

`expand_helper.py` 自动扫描目录下所有 `.py` 文件，按 `op` 名称和 `dtype` 签名匹配模板。


## 第四章 前置工作

### 4.1 Python DSL 扩展

| 工作项 | 说明 |
|--------|------|
| `@pto.tile_template` 装饰器 | 标记模板函数，指定对应的 Tile op 和 target |
| `pto.Tile` 属性接口 | 支持 `shape`、`valid_shape`、`element_type`、`element_size` 等属性访问 |
| `Tile` 下标访问 | 支持 `tile[i, j]` 语法用于 `vlds`/`vsts` 的地址计算 |
| 动态循环边界 | 当 `valid_shape` 为运行时动态值时，`range` 生成 `scf.for` |

### 4.2 PTOAS 编译器：Expand TileOp Pass

| 工作项 | 说明 |
|--------|------|
| 模板查找机制 | 根据 Tile op 种类和 dtype 匹配 Python DSL 模板 |
| 模板实例化 | 调用 Python DSL，传入具体 `tile_buf` 类型，获取实例化后的 MLIR |
| MLIR 解析与 inline | 解析生成的 MLIR 文本，inline 到调用点，绑定参数 |
| Cleanup | 实例化后运行 canonicalize 清理冗余 |

### 4.3 测试与文档

- Python DSL 模板编写和实例化的单元测试
- Expand TileOp pass 的端到端测试（`pto.tadd` → Vector IR）
- 融合场景测试（多个 Tile op 连续使用后的 VF Fusion）
- 更新 `PTO_IR_manual.md` 和 TileLang DSL Guide
