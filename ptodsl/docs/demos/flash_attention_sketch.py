# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Flash Attention redesign sketch.

This file is intentionally a design demo rather than runnable ``ptodsl`` code.
The goal is to make the *proposed* layering explicit and keep the semantic
contracts clean:

    @pto.tkernel          L1: tile-level orchestration and logical blocking
      └─ @pto.ukernel     L2: one KV-block worth of MTE/sync orchestration
           ├─ @pto.cube   L3: matrix products (QK^T and P@V)
           ├─ @pto.simd   L3: row-wise online softmax
           └─ @pto.simt   L3: scalar metadata and output blending

Design rules illustrated here:

1. ``tkernel`` owns logical tiling, tile allocation, and loop scheduling.
   It should not manually spell low-level DMA details for every micro step.
2. ``ukernel`` owns the per-block execution sandwich: stage the current K/V
   block with explicit micro-instructions, synchronize, call hardware-bound
   sub-kernels, and manage scratch/state.
3. ``tkernel`` may use tile ops such as ``tload`` / ``tstore`` at the logical
   scheduling boundary, but ``ukernel`` stays below that abstraction level.
   Once execution enters ``ukernel``, GM<->UB movement is expressed with
   ptr-based micro-instructions such as ``dma_load`` instead of tile ops.
   The DSL may make pointer materialization ergonomic, but the micro-instruction
   boundary itself stays explicit in authored code via ``as_ptr()``.
4. ``simd`` / ``simt`` / ``cube`` are hardware boundaries. They do not expose
   vreg values across the function boundary. Data crosses the boundary through
   UB-backed tiles or typed UB pointers only.
5. Online-softmax state is made explicit with ping-pong tiles
   (``m_prev``/``m_next``, ``l_prev``/``l_next``, ``o_prev``/``o_next``).
   Hiding these dependencies with in-place aliases makes the algorithm harder
   to read and obscures what the DSL needs to express.

The API spellings below are approximate and intentionally favor the redesign
surface over today's exact binding details.

Because this sketch targets a tracing-style frontend, any control flow that
must reach MLIR is expressed with structured DSL constructs such as
``pto.for_`` instead of native Python ``for`` loops.

Scalar literals and simple index/integer conversions are also shown in their
authored form.  The intended frontend behavior is to lift Python ``int``
literals and obvious scalar arithmetic into the corresponding MLIR scalar ops
implicitly, rather than forcing authors to spell ``pto.const(...)`` or
``index_cast(...)`` at every use site.
"""

from ptodsl import pto


# ═══════════════════════════════════════════════════════════════════════════════
# Level 3: hardware-bound sub-kernels
# ═══════════════════════════════════════════════════════════════════════════════
#
# Boundary contract:
# - Tile arguments are UB-backed or cube-local buffers carrying addressable
#   storage.
# - No vector register escapes a simd function.
# - No implicit global-memory access happens inside these kernels.


@pto.cube
def qk_matmul(
    q_tile: pto.Tile,      # UB, [Br, dim]
    k_tile: pto.Tile,      # UB, [Bc, dim]
    q_l0a: pto.Tile,       # LEFT scratch
    k_l0b: pto.Tile,       # RIGHT scratch
    s_acc: pto.Tile,       # ACC scratch
    s_tile: pto.Tile,      # UB, [Br, Bc] output
):
    """
    Compute ``S = Q @ K^T`` for one attention block.

    The key point for the redesign is that the cube kernel consumes UB tiles and
    explicit cube-local scratch, rather than pretending a UB tile can also stand
    in for LEFT/RIGHT/ACC state.
    """
    m = pto.tile_valid_rows(q_tile)
    k = pto.tile_valid_cols(q_tile)
    n = pto.tile_valid_rows(k_tile)

    # Caller owns scratch lifetime.  The cube kernel only expresses dataflow.
    pto.left_load(q_tile, q_l0a, m, k)
    pto.right_load(k_tile, k_l0b, k, n, transpose=True)
    pto.mad(q_l0a, k_l0b, s_acc)
    pto.acc_store_ub(s_acc, s_tile, m, n)


@pto.cube
def pv_matmul(
    p_tile: pto.Tile,      # UB, [Br, Bc]
    v_tile: pto.Tile,      # UB, [Bc, dim]
    p_l0a: pto.Tile,       # LEFT scratch (reused)
    v_l0b: pto.Tile,       # RIGHT scratch (reused)
    pv_acc: pto.Tile,      # ACC scratch (reused)
    pv_tile: pto.Tile,     # UB, [Br, dim] output
):
    """
    Compute ``PV = P @ V`` for the current block.

    This keeps the second matrix product on the cube path as well, instead of
    accidentally collapsing it into an elementwise vector expression.
    """
    m = pto.tile_valid_rows(p_tile)
    k = pto.tile_valid_cols(p_tile)
    n = pto.tile_valid_cols(v_tile)

    pto.left_load(p_tile, p_l0a, m, k)
    pto.right_load(v_tile, v_l0b, k, n)
    pto.mad(p_l0a, v_l0b, pv_acc)
    pto.acc_store_ub(pv_acc, pv_tile, m, n)


@pto.simd
def online_softmax_rows(
    s_tile: pto.Tile,          # UB, [Br, Bc]
    p_tile: pto.Tile,          # UB, [Br, Bc], output
    m_prev_tile: pto.Tile,     # UB, [Br, 1]
    l_prev_tile: pto.Tile,     # UB, [Br, 1]
    m_next_tile: pto.Tile,     # UB, [Br, 1], output
    l_next_tile: pto.Tile,     # UB, [Br, 1], output
    alpha_tile: pto.Tile,      # UB, [Br, 1], output
    beta_tile: pto.Tile,       # UB, [Br, 1], output
    row_start: pto.i32,
    row_stop: pto.i32,
    valid_cols: pto.i32,
):
    """
    Per-row online softmax update.

    For each active row::

        m_next = max(m_prev, row_max(S))
        P      = exp(S - m_next)
        l_next = l_prev * exp(m_prev - m_next) + row_sum(P)
        alpha  = l_prev * exp(m_prev - m_next) / l_next
        beta   = 1 / l_next

    ``alpha`` and ``beta`` are kept explicitly because the output update needs
    both the old accumulator and the newly computed ``P @ V`` contribution.
    """
    with pto.for_(row_start, row_stop, step=1) as row:
        col_mask = pto.make_mask(pto.f32, valid_cols)

        s_row = pto.vlds(s_tile[row, 0:])
        m_prev = pto.lds(m_prev_tile[row, 0])
        l_prev = pto.lds(l_prev_tile[row, 0])

        row_max = pto.vcgmax(s_row, col_mask)
        m_next = pto.max(m_prev, row_max)

        s_shifted = pto.vsubs(s_row, m_next, col_mask)
        p_row = pto.vexp(s_shifted, col_mask)

        row_sum = pto.vcgadd(p_row, col_mask)
        l_scaled = l_prev * pto.exp(m_prev - m_next)
        l_next = l_scaled + row_sum

        alpha = l_scaled / l_next
        beta = 1.0 / l_next

        pto.vsts(p_row, p_tile[row, 0:], col_mask)
        pto.sts(m_next_tile[row, 0], m_next)
        pto.sts(l_next_tile[row, 0], l_next)
        pto.sts(alpha_tile[row, 0], alpha)
        pto.sts(beta_tile[row, 0], beta)


@pto.simt
def blend_output_rows(
    o_prev_tile: pto.Tile,      # UB, [Br, dim]
    pv_tile: pto.Tile,          # UB, [Br, dim]
    alpha_tile: pto.Tile,       # UB, [Br, 1]
    beta_tile: pto.Tile,        # UB, [Br, 1]
    o_next_tile: pto.Tile,      # UB, [Br, dim], output
    row_start: pto.i32,
    row_stop: pto.i32,
    valid_dim: pto.i32,
):
    """
    Update the output accumulator with SIMT-style scalar element work::

        O_next[row, col] = alpha[row] * O_prev[row, col] + beta[row] * PV[row, col]

    This intentionally contrasts with ``online_softmax_rows``: the softmax step
    stays on the SIMD path because it is dominated by row-wise vector math,
    while the final blend is expressed here as explicit scalar work-items over
    the tile domain.
    """
    with pto.for_(row_start, row_stop, step=1) as row:
        alpha = pto.lds(alpha_tile[row, 0])
        beta = pto.lds(beta_tile[row, 0])

        with pto.for_(0, valid_dim, step=1) as col:
            o_prev = pto.lds(o_prev_tile[row, col])
            pv_val = pto.lds(pv_tile[row, col])

            o_next = alpha * o_prev + beta * pv_val
            pto.sts(o_next_tile[row, col], o_next)


@pto.simt
def materialize_tile_bounds(
    meta_ptr: pto.ptr(pto.i32, pto.MemorySpace.UB),   # [out] {row_start, row_stop, valid_cols}
    valid_rows: pto.i32,
    valid_cols: pto.i32,
):
    """
    Materialize tile-local loop bounds for the current block.

    The SIMT kernel stays intentionally small here: it is responsible for
    scalar control metadata, not for rewriting the vector or cube logic.
    """
    pto.sts(meta_ptr + 0, 0)
    pto.sts(meta_ptr + 4, valid_rows)
    pto.sts(meta_ptr + 8, valid_cols)


# ═══════════════════════════════════════════════════════════════════════════════
# Level 2: ukernel — one KV block worth of execution orchestration
# ═══════════════════════════════════════════════════════════════════════════════


@pto.ukernel
def kv_block_process(
    q_tile: pto.Tile,                # UB, reused across inner KV loop
    k_part: pto.PartitionTensorView, # GM view for current K block
    v_part: pto.PartitionTensorView, # GM view for current V block
    k_tile: pto.Tile,                # UB scratch
    v_tile: pto.Tile,                # UB scratch
    o_prev_tile: pto.Tile,           # UB state
    o_next_tile: pto.Tile,           # UB state
    m_prev_tile: pto.Tile,           # UB state
    l_prev_tile: pto.Tile,           # UB state
    m_next_tile: pto.Tile,           # UB state
    l_next_tile: pto.Tile,           # UB state
    s_tile: pto.Tile,                # UB scratch for QK^T
    p_tile: pto.Tile,                # UB scratch for probabilities
    pv_tile: pto.Tile,               # UB scratch for P@V
    alpha_tile: pto.Tile,            # UB scratch
    beta_tile: pto.Tile,             # UB scratch
    q_l0a: pto.Tile,                 # LEFT scratch for Q
    p_l0a: pto.Tile,                 # LEFT scratch for P
    rhs_l0b: pto.Tile,               # RIGHT scratch, reused by K/V
    qk_acc_tile: pto.Tile,           # ACC scratch for QK^T
    pv_acc_tile: pto.Tile,           # ACC scratch for P@V
    meta_ptr: pto.ptr(pto.i32, pto.MemorySpace.UB),
):
    """
    Process one KV block against an already-loaded Q tile.

    The ukernel owns:
    - staging the current K/V block into reusable UB scratch with explicit
      DMA-style micro-instructions,
    - synchronizing the hand-off between MTE, cube, simd, and simt stages,
    - wiring together the explicit state transition
      (prev -> next for m/l/o).
    """
    # ukernel deliberately stays below the tile-op abstraction boundary.
    # Current-block GM->UB staging is expressed as ptr-based DMA instructions.
    pto.dma_load(k_part.as_ptr(), k_tile.as_ptr())
    pto.dma_load(v_part.as_ptr(), v_tile.as_ptr())
    pto.mem_bar(pto.BarrierType.SYNC)

    materialize_tile_bounds(
        meta_ptr,
        pto.tile_valid_rows(q_tile),
        pto.tile_valid_rows(k_tile),
    )
    row_start = pto.lds(meta_ptr + 0)
    row_stop = pto.lds(meta_ptr + 4)
    valid_cols = pto.lds(meta_ptr + 8)

    # 1. S = Q @ K^T
    qk_matmul(q_tile, k_tile, q_l0a, rhs_l0b, qk_acc_tile, s_tile)
    pto.mem_bar(pto.BarrierType.SYNC)

    # 2. Row-wise online softmax over S
    online_softmax_rows(
        s_tile,
        p_tile,
        m_prev_tile,
        l_prev_tile,
        m_next_tile,
        l_next_tile,
        alpha_tile,
        beta_tile,
        row_start,
        row_stop,
        valid_cols,
    )
    pto.mem_bar(pto.BarrierType.SYNC)

    # 3. PV = P @ V
    pv_matmul(p_tile, v_tile, p_l0a, rhs_l0b, pv_acc_tile, pv_tile)
    pto.mem_bar(pto.BarrierType.SYNC)

    # 4. O_next = alpha * O_prev + beta * PV
    blend_output_rows(
        o_prev_tile,
        pv_tile,
        alpha_tile,
        beta_tile,
        o_next_tile,
        row_start,
        row_stop,
        pto.tile_valid_cols(v_tile),
    )
    pto.mem_bar(pto.BarrierType.SYNC)


# ═══════════════════════════════════════════════════════════════════════════════
# Level 1: tkernel — tile-level orchestration
# ═══════════════════════════════════════════════════════════════════════════════


@pto.tkernel
def flash_attention(
    Q: pto.TensorView,      # [batch, seq_q, heads, dim]
    K: pto.TensorView,      # [batch, seq_k, heads, dim]
    V: pto.TensorView,      # [batch, seq_k, heads, dim]
    O: pto.TensorView,      # [batch, seq_q, heads, dim]
):
    """
    Flash Attention top-level orchestration sketch.

    To keep the demo focused, batch/head loops are omitted and we show the
    per-head 2D core: ``[seq, dim]`` for Q/K/V/O.
    """
    Br = 128
    Bc = 128
    seq_q = 4096
    seq_k = 4096
    dim = 64

    q_blocks = (seq_q + Br - 1) // Br
    kv_blocks = (seq_k + Bc - 1) // Bc

    # UB resident logical tiles
    q_tile = pto.alloc_tile(pto.TileType(pto.f32), Br, dim)
    k_tile = pto.alloc_tile(pto.TileType(pto.f32), Bc, dim)
    v_tile = pto.alloc_tile(pto.TileType(pto.f32), Bc, dim)

    o_prev_tile = pto.alloc_tile(pto.TileType(pto.f32), Br, dim)
    o_next_tile = pto.alloc_tile(pto.TileType(pto.f32), Br, dim)
    m_prev_tile = pto.alloc_tile(pto.TileType(pto.f32), Br, 1)
    m_next_tile = pto.alloc_tile(pto.TileType(pto.f32), Br, 1)
    l_prev_tile = pto.alloc_tile(pto.TileType(pto.f32), Br, 1)
    l_next_tile = pto.alloc_tile(pto.TileType(pto.f32), Br, 1)

    s_tile = pto.alloc_tile(pto.TileType(pto.f32), Br, Bc)
    p_tile = pto.alloc_tile(pto.TileType(pto.f32), Br, Bc)
    pv_tile = pto.alloc_tile(pto.TileType(pto.f32), Br, dim)
    alpha_tile = pto.alloc_tile(pto.TileType(pto.f32), Br, 1)
    beta_tile = pto.alloc_tile(pto.TileType(pto.f32), Br, 1)

    # Cube-local scratch is explicit; it should not be conflated with UB tiles.
    q_l0a = pto.alloc_tile(pto.TileType(pto.f16, pto.MemorySpace.LEFT), Br, dim)
    p_l0a = pto.alloc_tile(pto.TileType(pto.f16, pto.MemorySpace.LEFT), Br, Bc)
    rhs_l0b = pto.alloc_tile(pto.TileType(pto.f16, pto.MemorySpace.RIGHT), Bc, dim)
    qk_acc_tile = pto.alloc_tile(pto.TileType(pto.f32, pto.MemorySpace.ACC), Br, Bc)
    pv_acc_tile = pto.alloc_tile(pto.TileType(pto.f32, pto.MemorySpace.ACC), Br, dim)

    # SIMT metadata buffer.  A tiny raw-pointer island is acceptable at the
    # ukernel boundary because this is scalar control data, not user-facing math.
    meta_tile = pto.alloc_tile(pto.TileType(pto.i32), 3, 1)
    meta_ptr = pto.tile_buf_addr(meta_tile)

    q_view = pto.make_tensor_view(Q, shape=[seq_q, dim])
    k_view = pto.make_tensor_view(K, shape=[seq_k, dim])
    v_view = pto.make_tensor_view(V, shape=[seq_k, dim])
    o_view = pto.make_tensor_view(O, shape=[seq_q, dim])

    with pto.for_(0, q_blocks, step=1) as qi:
        q_part = pto.partition_view(q_view, offsets=[qi * Br, 0], sizes=[Br, dim])
        o_part = pto.partition_view(o_view, offsets=[qi * Br, 0], sizes=[Br, dim])

        pto.tload(q_part, q_tile)

        # Initial online-softmax state for this Q block.
        pto.tile_fill(m_prev_tile, float("-inf"))
        pto.tile_fill(l_prev_tile, 0.0)
        pto.tile_fill(o_prev_tile, 0.0)

        with pto.for_(
            0,
            kv_blocks,
            step=1,
            iter_args=(m_prev_tile, m_next_tile, l_prev_tile, l_next_tile, o_prev_tile, o_next_tile),
        ) as kv_loop:
            kj = kv_loop.iv
            m_prev_cur, m_next_cur, l_prev_cur, l_next_cur, o_prev_cur, o_next_cur = kv_loop.iter_args
            k_part = pto.partition_view(k_view, offsets=[kj * Bc, 0], sizes=[Bc, dim])
            v_part = pto.partition_view(v_view, offsets=[kj * Bc, 0], sizes=[Bc, dim])

            kv_block_process(
                q_tile,
                k_part,
                v_part,
                k_tile,
                v_tile,
                o_prev_cur,
                o_next_cur,
                m_prev_cur,
                l_prev_cur,
                m_next_cur,
                l_next_cur,
                s_tile,
                p_tile,
                pv_tile,
                alpha_tile,
                beta_tile,
                q_l0a,
                p_l0a,
                rhs_l0b,
                qk_acc_tile,
                pv_acc_tile,
                meta_ptr,
            )

            # Loop-carried state makes the ping-pong ownership part of the IR.
            pto.yield_(
                m_next_cur,
                m_prev_cur,
                l_next_cur,
                l_prev_cur,
                o_next_cur,
                o_prev_cur,
            )

        _, _, _, _, o_final_tile, _ = kv_loop.results
        pto.tstore(o_final_tile, o_part)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer summary
# ═══════════════════════════════════════════════════════════════════════════════
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ L1  @pto.tkernel    Tile orchestration                                   │
# │                                                                            │
# │   alloc_tile / make_tensor_view / partition_view / tload / tstore         │
# │   outer Q loop + inner KV loop + ping-pong state ownership                │
# │                                                                            │
# │   Key idea: speak in logical tiles and block scheduling, not in           │
# │   instruction-sized address arithmetic.                                   │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ L2  @pto.ukernel    Per-block execution sandwich                          │
# │                                                                            │
# │   explicit dma_load(ptr, ptr) staging for current K/V block, mem_bar,     │
# │   call cube/simd/simt sub-kernels,                                        │
# │   manage scratch/state hand-off                                            │
# │                                                                            │
# │   Key idea: one place owns the "how this block runs on hardware" story.   │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ L3a @pto.cube       Matrix-product kernels                                 │
# │                                                                            │
# │   qk_matmul: Q @ K^T                                                       │
# │   pv_matmul: P @ V                                                         │
# │   explicit LEFT/RIGHT/ACC scratch + UB output                              │
# │                                                                            │
# │   Key idea: UB tiles are inputs/outputs; cube-local state is explicit.    │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ L3b @pto.simd       Row-wise vector math                                   │
# │                                                                            │
# │   online_softmax_rows                                                      │
# │   vreg stays local; persistent state is written back to UB tiles           │
# │                                                                            │
# │   Key idea: no cross-kernel vreg values, only UB-backed state.            │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ L3c @pto.simt       Scalar metadata and pointwise blend                    │
# │                                                                            │
# │   materialize_tile_bounds / blend_output_rows                              │
# │                                                                            │
# │   Key idea: SIMT handles scalar control facts and scalar tile walks.      │
# └──────────────────────────────────────────────────────────────────────────┘
#
#                       dataflow for one KV block
#
#   tkernel alloc/schedule
#          │
#          ▼
#   ukernel loads K/V block and sequences the pipeline
#          │
#          ├─ cube:  Q + K  ───────────────► S
#          ├─ simd:  S + (m_prev, l_prev) ─► P, (m_next, l_next), alpha, beta
#          ├─ cube:  P + V  ───────────────► PV
#          └─ simt:  (o_prev, PV, alpha, beta) ─► o_next
#
#   After each KV block:
#     (m_prev, l_prev, o_prev) := (m_next, l_next, o_next)
#
# The important part for the redesign is not the exact helper spelling, but
# that every cross-stage dependency is visible in the surface language.
