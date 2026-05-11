#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import struct
from pathlib import Path

import numpy as np


M = 40
N = 64
K = 50
FP_QUANT_ELEMS = 64
FP_TRANSPORT_ELEMS = FP_QUANT_ELEMS * 2
SRC_VALUE = np.float16(400.0)
ID_VALUE = np.float16(400.0)


def encode_scale(scale: float) -> np.uint64:
    return np.uint64(struct.unpack("!I", struct.pack("!f", scale))[0])


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    lhs = np.full((M, K), ID_VALUE, dtype=np.float16)
    rhs = np.full((K, N), SRC_VALUE, dtype=np.float16)
    fp = np.full(FP_QUANT_ELEMS, encode_scale(1.0), dtype=np.uint64)

    matmul = lhs.astype(np.float32) @ rhs.astype(np.float32)
    sat_golden = np.clip(
        matmul,
        np.finfo(np.float16).min,
        np.finfo(np.float16).max,
    ).astype(np.float16)
    with np.errstate(over="ignore", invalid="ignore"):
        nosat_golden = matmul.astype(np.float16)

    zero = np.zeros((M, N), dtype=np.float16)
    lhs.reshape(-1).tofile(output_dir / "v1.bin")
    rhs.reshape(-1).tofile(output_dir / "v2.bin")
    fp.view(np.uint32).reshape(FP_TRANSPORT_ELEMS).tofile(output_dir / "v3.bin")
    zero.reshape(-1).tofile(output_dir / "v4.bin")
    zero.reshape(-1).tofile(output_dir / "v5.bin")
    zero.reshape(-1).tofile(output_dir / "v6.bin")
    zero.reshape(-1).tofile(output_dir / "v7.bin")
    zero.reshape(-1).tofile(output_dir / "v8.bin")
    zero.reshape(-1).tofile(output_dir / "v9.bin")

    sat_golden.reshape(-1).tofile(output_dir / "golden_v4.bin")
    nosat_golden.reshape(-1).tofile(output_dir / "golden_v5.bin")
    sat_golden.reshape(-1).tofile(output_dir / "golden_v6.bin")
    nosat_golden.reshape(-1).tofile(output_dir / "golden_v7.bin")
    sat_golden.reshape(-1).tofile(output_dir / "golden_v8.bin")
    nosat_golden.reshape(-1).tofile(output_dir / "golden_v9.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
