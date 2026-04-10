# TileLang ST 精度验证框架

## 概述

TileLang ST（System Test）框架用于在 Ascend NPU 硬件或仿真器上端到端验证 TileLang DSL 模板库生成的 kernel 精度。框架参考 pto-isa 的 ST 目录结构和运行流程，但针对 TileLang 的编译路径（`.pto → LLVM IR → .o`，而非 pto-isa 的 `.cpp → -xcce → .o`）做了适配。

### 与 pto-isa ST 的关键差异

| | pto-isa ST | TileLang ST |
|--|-----------|-------------|
| kernel 源码 | 手写 C++（`kernel.cpp`） | PTO DSL（`tadd.pto`） |
| kernel 编译 | `bisheng -xcce kernel.cpp` | `ptoas .pto → .ll` + `bisheng -x ir .ll → .o` |
| 精度比较 | C++ `ResultCmp()`（GTest） | Python `np.allclose`（`compare.py`） |
| 多 case 支持 | 单文件多 GTest TEST_F | 单 `.pto` 多 kernel 函数 + case table |

### 执行流程

```
run_st.py
  ├── set_env        # 设置 ASCEND / simulator 环境变量
  ├── cmake + make   # ptoas→.ll → bisheng→.o → link .so → build 可执行文件
  ├── gen_data.py    # numpy 生成 input + golden（per-case 子目录）
  ├── ./tadd [case]  # 运行 kernel，写 output.bin（per-case 子目录）
  └── compare.py     # np.allclose 逐 case 比较 golden vs output
```

## 目录结构

```
test/tilelang_st/
├── script/
│   └── run_st.py                  # 统一入口脚本
└── npu/
    └── a5/                        # SoC 架构
        └── src/st/
            ├── CMakeLists.txt     # 顶层 CMake（编译器/环境配置）
            └── testcase/
                ├── CMakeLists.txt # pto_tilelang_vec_st() 宏定义 + op 注册
                └── tadd/          # 每个 op 一个目录
                    ├── CMakeLists.txt  # 一行：pto_tilelang_vec_st(tadd)
                    ├── tadd.pto        # kernel DSL（可包含多个函数）
                    ├── launch.cpp      # kernel 声明 + launch wrapper
                    ├── main.cpp        # host driver（case table 驱动）
                    ├── gen_data.py     # 数据生成
                    └── compare.py      # 精度比较
```

## 快速上手

### 运行已有测试

```bash
# 在 NPU 上跑 tadd 全部 case
python3 test/tilelang_st/script/run_st.py -r npu -v a5 -t tadd

# 在仿真器上跑
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tadd

# 只跑某个 case
python3 test/tilelang_st/script/run_st.py -r npu -v a5 -t tadd -c f32_16x64

# 跳过编译（已有 build 产物时）
python3 test/tilelang_st/script/run_st.py -r npu -v a5 -t tadd -w
```

### run_st.py 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `-r, --run-mode` | 运行模式 | `npu` 或 `sim` |
| `-v, --soc-version` | 架构版本 | `a5` |
| `-t, --testcase` | op 名称 | `tadd` |
| `-c, --case` | 指定单个 case（可选） | `f32_16x64` |
| `-p, --ptoas-bin` | ptoas 路径（可选，默认自动查找） | `/path/to/ptoas` |
| `-w, --without-build` | 跳过编译 | — |

ptoas 路径查找顺序：`-p` 参数 → `PTOAS_BIN` 环境变量 → 从脚本位置向上遍历 `build/bin/ptoas`。

## 新增一个 op 测试

以新增 `tsub` 为例。

### 第 1 步：创建目录和文件

```bash
mkdir test/tilelang_st/npu/a5/src/st/testcase/tsub
```

需要创建 6 个文件，下面逐一说明。

### 第 2 步：编写 kernel（tsub.pto）

单个 `.pto` 文件中包含所有 case 对应的 kernel 函数，函数名格式为 `@TSUB_<dtype>_<rows>x<cols>`：

```mlir
module {
  // Case 0: f32 16x64
  func.func @TSUB_f32_16x64(%a_ptr: !pto.ptr<f32>, %b_ptr: !pto.ptr<f32>, %c_ptr: !pto.ptr<f32>) {
    // make_tensor_view × 3, partition_view × 3, alloc_tile × 3
    // tload × 2, tsub, tstore
    ...
    return
  }

  // Case 1: f32 32x32
  func.func @TSUB_f32_32x32(%a_ptr: !pto.ptr<f32>, %b_ptr: !pto.ptr<f32>, %c_ptr: !pto.ptr<f32>) {
    ...
    return
  }
}
```

### 第 3 步：编写 launch wrapper（launch.cpp）

每个 kernel 函数需要一个 `__global__` 声明和一个 `Launch*` C++ wrapper：

```cpp
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#include <stdint.h>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

__global__ AICORE void TSUB_f32_16x64(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTSUB_f32_16x64(float *a, float *b, float *c, void *stream) {
    TSUB_f32_16x64<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

__global__ AICORE void TSUB_f32_32x32(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTSUB_f32_32x32(float *a, float *b, float *c, void *stream) {
    TSUB_f32_32x32<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}
```

**注意：** `__global__`、`AICORE`、`__gm__`、`<<<>>>` 是 CCE 扩展语法，本地 clang 会报错，这是预期行为——launch.cpp 由 bisheng `-xcce` 编译。

### 第 4 步：编写 host driver（main.cpp）

使用 case table 驱动，每个 case 从独立子目录读写数据：

```cpp
#include "test_common.h"
#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                    \
    do {                                                                   \
        const aclError _ret = (expr);                                      \
        if (_ret != ACL_SUCCESS) {                                         \
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n",        \
                         #expr, (int)_ret, __FILE__, __LINE__);            \
            const char *_recent = aclGetRecentErrMsg();                    \
            if (_recent != nullptr && _recent[0] != '\0')                  \
                std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent); \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// launch wrappers
void LaunchTSUB_f32_16x64(float *a, float *b, float *c, void *stream);
void LaunchTSUB_f32_32x32(float *a, float *b, float *c, void *stream);

using LaunchFn = void (*)(float *, float *, float *, void *);

struct TestCase {
    const char *name;     // 与 gen_data.py / compare.py 的 case name 一致
    LaunchFn    launch;
    size_t      rows;
    size_t      cols;
    size_t      elemSize; // bytes per element
};

static const TestCase kCases[] = {
    {"f32_16x64", LaunchTSUB_f32_16x64, 16, 64, sizeof(float)},
    {"f32_32x32", LaunchTSUB_f32_32x32, 32, 32, sizeof(float)},
};

// main() 中循环 kCases，对每个 case：
//   1. 从 ./<case_name>/input1.bin, input2.bin 读数据
//   2. H2D → launch kernel → D2H
//   3. 写 ./<case_name>/output.bin
// 支持 ./tsub [case_name] 过滤单个 case
```

完整实现参考 `testcase/tadd/main.cpp`。

### 第 5 步：数据生成与精度比较

**gen_data.py** — 为每个 case 生成独立子目录的 `input1.bin`、`input2.bin`、`golden.bin`：

```python
import os
import numpy as np

np.random.seed(42)

CASES = [
    {"name": "f32_16x64", "dtype": np.float32, "shape": (16, 64)},
    {"name": "f32_32x32", "dtype": np.float32, "shape": (32, 32)},
]

for case in CASES:
    case_dir = case["name"]
    os.makedirs(case_dir, exist_ok=True)

    input1 = np.random.randint(1, 10, size=case["shape"]).astype(case["dtype"])
    input2 = np.random.randint(1, 10, size=case["shape"]).astype(case["dtype"])
    golden = (input1 - input2).astype(case["dtype"], copy=False)  # tsub: 减法

    input1.tofile(os.path.join(case_dir, "input1.bin"))
    input2.tofile(os.path.join(case_dir, "input2.bin"))
    golden.tofile(os.path.join(case_dir, "golden.bin"))
```

**compare.py** — 逐 case 比较，支持 `python compare.py [case_name]` 过滤：

```python
import sys, os
import numpy as np

CASES = [
    {"name": "f32_16x64", "dtype": np.float32, "eps": 1e-6},
    {"name": "f32_32x32", "dtype": np.float32, "eps": 1e-6},
]

def compare_bin(golden_path, output_path, dtype, eps):
    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    if golden.shape != output.shape:
        return False
    return np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True)

case_filter = sys.argv[1] if len(sys.argv) > 1 else None
all_passed = True
for case in CASES:
    if case_filter and case["name"] != case_filter:
        continue
    ok = compare_bin(
        os.path.join(case["name"], "golden.bin"),
        os.path.join(case["name"], "output.bin"),
        case["dtype"], case["eps"])
    if not ok:
        all_passed = False
if not all_passed:
    sys.exit(2)
```

### 第 6 步：CMake 注册

**testcase/tsub/CMakeLists.txt**（一行）：

```cmake
pto_tilelang_vec_st(tsub)
```

**testcase/CMakeLists.txt** 中注册：

```cmake
set(ALL_TESTCASES
    tadd
    tsub    # <-- 新增
)
```

## 在已有 op 下新增 case

不需要修改 CMake。只需要同步修改 4 个文件：

| 步骤 | 文件 | 修改内容 |
|------|------|---------|
| 1 | `tadd.pto` | 新增 `func.func @TADD_<dtype>_<RxC>(...)` |
| 2 | `launch.cpp` | 新增 `__global__` 声明 + `LaunchTADD_<dtype>_<RxC>` wrapper |
| 3 | `main.cpp` | `kCases[]` 数组新增一行 |
| 4 | `gen_data.py` + `compare.py` | `CASES` 列表各新增一行 |

### 命名约定

- kernel 函数名：`@<OP>_<dtype>_<rows>x<cols>`，例如 `@TADD_f32_16x64`
- launch wrapper：`Launch<OP>_<dtype>_<rows>x<cols>`
- case name / 子目录名：`<dtype>_<rows>x<cols>`，例如 `f32_16x64`
- 三处 case 列表（`main.cpp` `kCases[]`、`gen_data.py` `CASES`、`compare.py` `CASES`）必须保持一致

## CMake 编译流水线

`pto_tilelang_vec_st(NAME)` 宏定义在 `testcase/CMakeLists.txt` 中，完成 4 步编译：

```
Step 1: ptoas  NAME.pto → NAME_kernel.ll     (LLVM IR)
        ptoas --pto-arch=a5 --pto-backend=vpto
              --enable-tile-op-expand --vpto-emit-hivm-llvm
              NAME.pto -o NAME_kernel.ll

Step 2: bisheng NAME_kernel.ll → NAME_kernel.o (object file)
        bisheng --target=hiipu64-hisilicon-cce
                -march=dav-c310-vec
                --cce-aicore-arch=dav-c310-vec
                --cce-aicore-only
                -c -x ir NAME_kernel.ll -o NAME_kernel.o

Step 3: launch.cpp (-xcce) + NAME_kernel.o → libNAME_kernel.so
        bisheng -xcce launch.cpp + link NAME_kernel.o

Step 4: main.cpp (-xc++) → NAME (可执行文件)
        link libNAME_kernel.so + ACL runtime libraries
```

## 精度比较说明

使用 `np.allclose(golden, output, atol=eps, rtol=eps)` 进行比较。不同 dtype 建议的 eps 值：

| dtype | 建议 eps |
|-------|---------|
| float32 | 1e-6 |
| float16 | 1e-3 |
| bfloat16 | 1e-2 |
| int8/int32 | 0（精确匹配） |

比较失败时会输出 max diff、出错位置和 golden/output 值，便于定位问题。
