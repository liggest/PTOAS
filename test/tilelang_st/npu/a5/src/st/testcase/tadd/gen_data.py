#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import os
import numpy as np

np.random.seed(19)

CASES = [
    {"name": "f32_16x64", "dtype": np.float32, "shape": (16, 64)},
    {"name": "f32_32x32", "dtype": np.float32, "shape": (32, 32)},
]

for case in CASES:
    case_dir = case["name"]
    os.makedirs(case_dir, exist_ok=True)

    input1 = np.random.randint(1, 10, size=case["shape"]).astype(case["dtype"])
    input2 = np.random.randint(1, 10, size=case["shape"]).astype(case["dtype"])
    golden = (input1 + input2).astype(case["dtype"], copy=False)

    input1.tofile(os.path.join(case_dir, "input1.bin"))
    input2.tofile(os.path.join(case_dir, "input2.bin"))
    golden.tofile(os.path.join(case_dir, "golden.bin"))
    print(f"[INFO] gen_data: {case['name']} shape={case['shape']} dtype={case['dtype'].__name__}")
