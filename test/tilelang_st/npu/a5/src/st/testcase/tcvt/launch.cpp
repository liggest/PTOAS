// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void TCVT_f16_to_si8_1x128(__gm__ uint16_t *src, __gm__ int8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_si8_2x64(__gm__ uint16_t *src, __gm__ int8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_si8_4x32(__gm__ uint16_t *src, __gm__ int8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_si8_2x128(__gm__ uint16_t *src, __gm__ int8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_si8_4x65(__gm__ uint16_t *src, __gm__ int8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_si8_4x200(__gm__ uint16_t *src, __gm__ int8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_si8_1x129(__gm__ uint16_t *src, __gm__ int8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_ui8_1x128(__gm__ uint16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_ui8_2x64(__gm__ uint16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_ui8_4x32(__gm__ uint16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_ui8_2x128(__gm__ uint16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_ui8_4x65(__gm__ uint16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_ui8_4x200(__gm__ uint16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_ui8_1x129(__gm__ uint16_t *src, __gm__ uint8_t *dst);

void LaunchTCVT_f16_to_si8_1x128(void *src, void *dst, void *stream) {
    TCVT_f16_to_si8_1x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int8_t *)dst);
}

void LaunchTCVT_f16_to_si8_2x64(void *src, void *dst, void *stream) {
    TCVT_f16_to_si8_2x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int8_t *)dst);
}

void LaunchTCVT_f16_to_si8_4x32(void *src, void *dst, void *stream) {
    TCVT_f16_to_si8_4x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int8_t *)dst);
}

void LaunchTCVT_f16_to_si8_2x128(void *src, void *dst, void *stream) {
    TCVT_f16_to_si8_2x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int8_t *)dst);
}

void LaunchTCVT_f16_to_si8_4x65(void *src, void *dst, void *stream) {
    TCVT_f16_to_si8_4x65<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int8_t *)dst);
}

void LaunchTCVT_f16_to_si8_4x200(void *src, void *dst, void *stream) {
    TCVT_f16_to_si8_4x200<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int8_t *)dst);
}

void LaunchTCVT_f16_to_si8_1x129(void *src, void *dst, void *stream) {
    TCVT_f16_to_si8_1x129<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int8_t *)dst);
}

void LaunchTCVT_f16_to_ui8_1x128(void *src, void *dst, void *stream) {
    TCVT_f16_to_ui8_1x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_f16_to_ui8_2x64(void *src, void *dst, void *stream) {
    TCVT_f16_to_ui8_2x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_f16_to_ui8_4x32(void *src, void *dst, void *stream) {
    TCVT_f16_to_ui8_4x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_f16_to_ui8_2x128(void *src, void *dst, void *stream) {
    TCVT_f16_to_ui8_2x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_f16_to_ui8_4x65(void *src, void *dst, void *stream) {
    TCVT_f16_to_ui8_4x65<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_f16_to_ui8_4x200(void *src, void *dst, void *stream) {
    TCVT_f16_to_ui8_4x200<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_f16_to_ui8_1x129(void *src, void *dst, void *stream) {
    TCVT_f16_to_ui8_1x129<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint8_t *)dst);
}
