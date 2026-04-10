// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- FoldTileBufIntrinsics.cpp ------------------------------------------===//
//
// After TileLang DSL template functions are inlined, the IR contains
// structured-view intrinsics that reference template parameters:
//
// tile_buf family:
//   - pto.tile_buf_addr   → extract memref address from tile_buf
//   - pto.tile_valid_rows → extract valid row count
//   - pto.tile_valid_cols → extract valid column count
//
// tensor_view family:
//   - pto.tensor_view_addr       → extract memref/ptr from tensor_view
//   - pto.get_tensor_view_dim    → extract dimension size
//   - pto.get_tensor_view_stride → extract dimension stride
//
// This pass resolves them against the concrete values at the call site.
// For tensor_view intrinsics, the pass traces through the full
// unrealized_conversion_cast → memref.subview → memref.reinterpret_cast
// chain to fold directly to constants or SSA operands from the
// reinterpret_cast, without generating intermediate memref.dim /
// memref.extract_strided_metadata ops.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
namespace pto {
  #define GEN_PASS_DEF_FOLDTILEBUFINTRINSICS
  #include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

/// Locate the `pto.bind_tile` op that produced `tileBuf`, expecting the
/// strict pattern emitted by MemrefToTileBuf:
///
///   %bound  = pto.bind_tile %src, %vrow, %vcol : memref -> memref
///   %tile   = builtin.unrealized_conversion_cast %bound : memref -> !pto.tile_buf
///
/// Returns nullptr (with an error emitted on `loc`) if the pattern does not
/// hold — the caller is expected to signal pass failure.
static pto::BindTileOp findBindTileForTileBuf(Value tileBuf, Operation *user) {
  auto cast = tileBuf.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getNumOperands() != 1) {
    user->emitError(
        "FoldTileBufIntrinsics: expected tile_buf to be defined by a "
        "single-operand builtin.unrealized_conversion_cast");
    return nullptr;
  }
  auto bindOp = cast.getOperand(0).getDefiningOp<pto::BindTileOp>();
  if (!bindOp) {
    user->emitError(
        "FoldTileBufIntrinsics: expected unrealized_conversion_cast operand "
        "to be defined by pto.bind_tile");
    return nullptr;
  }
  return bindOp;
}

struct ViewChain {
  UnrealizedConversionCastOp cast;
  memref::SubViewOp subview;
  memref::ReinterpretCastOp reinterpretCast;
  Value baseMemref;
};

static std::optional<ViewChain> traceViewChain(Value tensorView,
                                               Operation *user) {
  Value memrefVal;
  UnrealizedConversionCastOp castOp;

  if (isa<MemRefType>(tensorView.getType())) {
    memrefVal = tensorView;
  } else {
    castOp = tensorView.getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp || castOp.getNumOperands() != 1) {
      user->emitError(
          "FoldTileBufIntrinsics: expected tensor_view to be defined by a "
          "single-operand builtin.unrealized_conversion_cast");
      return std::nullopt;
    }
    memrefVal = castOp.getOperand(0);
    if (!isa<MemRefType>(memrefVal.getType())) {
      user->emitError(
          "FoldTileBufIntrinsics: expected cast operand to be a memref, got ")
          << memrefVal.getType();
      return std::nullopt;
    }
  }

  auto subviewOp = memrefVal.getDefiningOp<memref::SubViewOp>();
  if (!subviewOp) {
    user->emitError("FoldTileBufIntrinsics: expected memref to be defined by "
                    "memref.subview, got ")
        << (memrefVal.getDefiningOp()
                ? memrefVal.getDefiningOp()->getName().getStringRef()
                : StringRef("block argument"));
    return std::nullopt;
  }

  auto rcOp = subviewOp.getSource().getDefiningOp<memref::ReinterpretCastOp>();
  if (!rcOp) {
    user->emitError(
        "FoldTileBufIntrinsics: expected subview source to be defined by "
        "memref.reinterpret_cast, got ")
        << (subviewOp.getSource().getDefiningOp()
                ? subviewOp.getSource().getDefiningOp()->getName().getStringRef()
                : StringRef("block argument"));
    return std::nullopt;
  }

  return ViewChain{castOp, subviewOp, rcOp, rcOp.getSource()};
}

static bool getConstIndexValue(Value v, int64_t &out) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    out = cOp.value();
    return true;
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    out = cInt.value();
    return true;
  }
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue())) {
      out = ia.getInt();
      return true;
    }
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
    return getConstIndexValue(castOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
    return getConstIndexValue(extOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
    return getConstIndexValue(extOp.getIn(), out);
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>())
    return getConstIndexValue(truncOp.getIn(), out);
  return false;
}

static Value getValueOrCreateConstant(OpBuilder &builder, Location loc,
                                      OpFoldResult ofr) {
  if (auto val = dyn_cast<Value>(ofr))
    return val;
  auto intAttr = dyn_cast<IntegerAttr>(cast<Attribute>(ofr));
  assert(intAttr && "expected integer attribute in OpFoldResult");
  return builder.create<arith::ConstantIndexOp>(loc, intAttr.getInt());
}

static bool isAllStaticZero(ArrayRef<OpFoldResult> ofrs) {
  for (OpFoldResult ofr : ofrs) {
    auto attr = dyn_cast<Attribute>(ofr);
    if (!attr)
      return false;
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr || intAttr.getInt() != 0)
      return false;
  }
  return true;
}

static Value computeResultStride(OpBuilder &builder, Location loc,
                                 OpFoldResult rcStride,
                                 OpFoldResult svStride) {
  if (auto attr = dyn_cast<Attribute>(svStride)) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (intAttr && intAttr.getInt() == 1)
      return getValueOrCreateConstant(builder, loc, rcStride);
  }

  Value lhs = getValueOrCreateConstant(builder, loc, rcStride);
  Value rhs = getValueOrCreateConstant(builder, loc, svStride);
  return builder.create<arith::MulIOp>(loc, lhs, rhs);
}

static Value computeLinearOffset(OpBuilder &builder, Location loc,
                                 ArrayRef<OpFoldResult> rcOffsets,
                                 ArrayRef<OpFoldResult> svOffsets,
                                 ArrayRef<OpFoldResult> rcStrides) {
  bool rcAllZero = isAllStaticZero(rcOffsets);
  bool svAllZero = isAllStaticZero(svOffsets);

  if (rcAllZero && svAllZero)
    return Value();

  Value svPart;
  if (!svAllZero) {
    for (auto [svOffset, rcStride] : llvm::zip(svOffsets, rcStrides)) {
      if (auto attr = dyn_cast<Attribute>(svOffset)) {
        auto intAttr = dyn_cast<IntegerAttr>(attr);
        if (intAttr && intAttr.getInt() == 0)
          continue;
      }

      Value off = getValueOrCreateConstant(builder, loc, svOffset);
      Value stride = getValueOrCreateConstant(builder, loc, rcStride);
      Value term = builder.create<arith::MulIOp>(loc, off, stride);
      svPart = svPart ? builder.create<arith::AddIOp>(loc, svPart, term) : term;
    }
  }

  Value rcPart;
  if (!rcAllZero) {
    if (rcOffsets.empty())
      return Value();
    rcPart = getValueOrCreateConstant(builder, loc, rcOffsets.front());
  }

  if (rcPart && svPart)
    return builder.create<arith::AddIOp>(loc, rcPart, svPart);
  return rcPart ? rcPart : svPart;
}

struct FoldTileBufIntrinsicsPass
    : public pto::impl::FoldTileBufIntrinsicsBase<FoldTileBufIntrinsicsPass> {
  using FoldTileBufIntrinsicsBase::FoldTileBufIntrinsicsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // Leftover TileLang template instances (private, uncalled after
    // PTOInlineLibCall) still contain pto.tile_buf_addr / tile_valid_*
    // ops on tile_buf function arguments — they have no bind_tile to
    // fold against and will be removed by later DCE.  Skip them.
    if (func->hasAttr("pto.tilelang.instance"))
      return;

    SmallVector<pto::TileBufAddrOp, 8> addrOps;
    SmallVector<pto::TileValidRowsOp, 8> rowsOps;
    SmallVector<pto::TileValidColsOp, 8> colsOps;
    SmallVector<pto::TensorViewAddrOp, 8> tvAddrOps;
    SmallVector<pto::GetTensorViewDimOp, 8> tvDimOps;
    SmallVector<pto::GetTensorViewStrideOp, 8> tvStrideOps;

    func.walk([&](Operation *op) {
      if (auto addr = dyn_cast<pto::TileBufAddrOp>(op))
        addrOps.push_back(addr);
      else if (auto rows = dyn_cast<pto::TileValidRowsOp>(op))
        rowsOps.push_back(rows);
      else if (auto cols = dyn_cast<pto::TileValidColsOp>(op))
        colsOps.push_back(cols);
      else if (auto tvAddr = dyn_cast<pto::TensorViewAddrOp>(op))
        tvAddrOps.push_back(tvAddr);
      else if (auto tvDim = dyn_cast<pto::GetTensorViewDimOp>(op))
        tvDimOps.push_back(tvDim);
      else if (auto tvStride = dyn_cast<pto::GetTensorViewStrideOp>(op))
        tvStrideOps.push_back(tvStride);
    });

    // Fold pto.tile_buf_addr → bind_tile's source memref (the static-layout
    // pto.pointer_cast result), or further to pto.castptr when the requested
    // result type is already !pto.ptr<...>. This bypasses the dynamic-offset
    // memref produced by bind_tile itself, so downstream vlds/vsts
    // canonicalization sees a clean strided<[..],offset:0> layout.
    for (auto addrOp : addrOps) {
      pto::BindTileOp bindOp = findBindTileForTileBuf(addrOp.getSrc(), addrOp);
      if (!bindOp)
        return signalPassFailure();

      Value srcMemref = bindOp.getSource();
      if (!isa<MemRefType>(srcMemref.getType())) {
        addrOp.emitError(
            "FoldTileBufIntrinsics: pto.bind_tile source is not a memref");
        return signalPassFailure();
      }

      if (auto resultMemrefType = dyn_cast<MemRefType>(addrOp.getDst().getType())) {
        // The declared tile_buf_addr result type may differ from the actual
        // bind_tile source layout (e.g. plain shape vs. strided layout) — the
        // downstream vector ops are polymorphic over strided layouts of the
        // same element type and shape, so retype the result in place.
        if (srcMemref.getType() != resultMemrefType)
          addrOp.getDst().setType(cast<MemRefType>(srcMemref.getType()));
        addrOp.getDst().replaceAllUsesWith(srcMemref);
        addrOp.erase();
        continue;
      }

      auto resultPtrType = dyn_cast<pto::PtrType>(addrOp.getDst().getType());
      if (!resultPtrType) {
        addrOp.emitError(
            "FoldTileBufIntrinsics: tile_buf_addr result must be memref or !pto.ptr");
        return signalPassFailure();
      }

      builder.setInsertionPoint(addrOp);
      Value replacement =
          builder.create<pto::CastPtrOp>(addrOp.getLoc(), resultPtrType, srcMemref);
      addrOp.getDst().replaceAllUsesWith(replacement);
      addrOp.erase();
    }

    // Fold pto.tile_valid_rows → arith.constant (static) or bind_tile's
    // valid_row operand (dynamic).
    for (auto rowsOp : rowsOps) {
      builder.setInsertionPoint(rowsOp);
      auto tbTy = dyn_cast<pto::TileBufType>(rowsOp.getSrc().getType());
      if (!tbTy || tbTy.getValidShape().empty()) {
        rowsOp.emitError("tile_valid_rows: invalid tile_buf type");
        return signalPassFailure();
      }

      int64_t vRow = tbTy.getValidShape()[0];
      Value replacement;
      if (vRow != ShapedType::kDynamic) {
        replacement =
            builder.create<arith::ConstantIndexOp>(rowsOp.getLoc(), vRow);
      } else {
        pto::BindTileOp bindOp =
            findBindTileForTileBuf(rowsOp.getSrc(), rowsOp);
        if (!bindOp)
          return signalPassFailure();
        replacement = bindOp.getValidRow();
        if (!replacement) {
          rowsOp.emitError(
              "tile_valid_rows: dynamic v_row but bind_tile has no "
              "valid_row operand");
          return signalPassFailure();
        }
        // bind_tile's valid_row is `index` (matches tile_valid_rows result),
        // so no type adaptation is required.
        assert(replacement.getType() == rowsOp.getResult().getType() &&
               "tile_valid_rows fold: type mismatch with bind_tile valid_row");
      }
      rowsOp.getResult().replaceAllUsesWith(replacement);
      rowsOp.erase();
    }

    // Fold pto.tile_valid_cols → arith.constant (static) or bind_tile's
    // valid_col operand (dynamic).
    for (auto colsOp : colsOps) {
      builder.setInsertionPoint(colsOp);
      auto tbTy = dyn_cast<pto::TileBufType>(colsOp.getSrc().getType());
      if (!tbTy || tbTy.getValidShape().size() < 2) {
        colsOp.emitError("tile_valid_cols: invalid tile_buf type");
        return signalPassFailure();
      }

      int64_t vCol = tbTy.getValidShape()[1];
      Value replacement;
      if (vCol != ShapedType::kDynamic) {
        replacement =
            builder.create<arith::ConstantIndexOp>(colsOp.getLoc(), vCol);
      } else {
        pto::BindTileOp bindOp =
            findBindTileForTileBuf(colsOp.getSrc(), colsOp);
        if (!bindOp)
          return signalPassFailure();
        replacement = bindOp.getValidCol();
        if (!replacement) {
          colsOp.emitError(
              "tile_valid_cols: dynamic v_col but bind_tile has no "
              "valid_col operand");
          return signalPassFailure();
        }
        assert(replacement.getType() == colsOp.getResult().getType() &&
               "tile_valid_cols fold: type mismatch with bind_tile valid_col");
      }
      colsOp.getResult().replaceAllUsesWith(replacement);
      colsOp.erase();
    }

    for (auto addrOp : tvAddrOps) {
      auto chain = traceViewChain(addrOp.getSrc(), addrOp);
      if (!chain)
        return signalPassFailure();

      builder.setInsertionPoint(addrOp);

      auto resultPtrType = dyn_cast<pto::PtrType>(addrOp.getDst().getType());
      if (!resultPtrType) {
        if (auto resultMemrefType =
                dyn_cast<MemRefType>(addrOp.getDst().getType())) {
          Value base = chain->baseMemref;
          if (base.getType() != resultMemrefType)
            addrOp.getDst().setType(cast<MemRefType>(base.getType()));
          addrOp.getDst().replaceAllUsesWith(base);
          addrOp.erase();
          continue;
        }
        addrOp.emitError(
            "FoldTileBufIntrinsics: tensor_view_addr result must be memref or "
            "!pto.ptr");
        return signalPassFailure();
      }

      Value linearOffset =
          computeLinearOffset(builder, addrOp.getLoc(),
                              chain->reinterpretCast.getMixedOffsets(),
                              chain->subview.getMixedOffsets(),
                              chain->reinterpretCast.getMixedStrides());

      Value basePtr = builder.create<pto::CastPtrOp>(
          addrOp.getLoc(), resultPtrType, chain->baseMemref);
      Value replacement =
          linearOffset
              ? builder.create<pto::AddPtrOp>(addrOp.getLoc(), resultPtrType,
                                              basePtr, linearOffset)
              : basePtr;

      addrOp.getDst().replaceAllUsesWith(replacement);
      addrOp.erase();
    }

    for (auto dimOp : tvDimOps) {
      auto chain = traceViewChain(dimOp.getTensorView(), dimOp);
      if (!chain)
        return signalPassFailure();

      int64_t dimIdx = 0;
      if (!getConstIndexValue(dimOp.getDimIndex(), dimIdx)) {
        dimOp.emitError(
            "FoldTileBufIntrinsics: get_tensor_view_dim requires a constant "
            "dim index");
        return signalPassFailure();
      }

      auto svTy = cast<MemRefType>(chain->subview.getType());
      if (dimIdx < 0 || dimIdx >= svTy.getRank()) {
        dimOp.emitError(
            "FoldTileBufIntrinsics: get_tensor_view_dim dim index out of "
            "bounds");
        return signalPassFailure();
      }

      builder.setInsertionPoint(dimOp);
      Value replacement;
      if (!svTy.isDynamicDim(dimIdx)) {
        replacement =
            builder.create<arith::ConstantIndexOp>(dimOp.getLoc(),
                                                   svTy.getDimSize(dimIdx));
      } else {
        replacement = getValueOrCreateConstant(
            builder, dimOp.getLoc(), chain->subview.getMixedSizes()[dimIdx]);
      }

      dimOp.getResult().replaceAllUsesWith(replacement);
      dimOp.erase();
    }

    for (auto strideOp : tvStrideOps) {
      auto chain = traceViewChain(strideOp.getTensorView(), strideOp);
      if (!chain)
        return signalPassFailure();

      int64_t dimIdx = 0;
      if (!getConstIndexValue(strideOp.getDimIndex(), dimIdx)) {
        strideOp.emitError(
            "FoldTileBufIntrinsics: get_tensor_view_stride requires a "
            "constant dim index");
        return signalPassFailure();
      }

      auto svTy = cast<MemRefType>(chain->subview.getType());
      if (dimIdx < 0 || dimIdx >= svTy.getRank()) {
        strideOp.emitError(
            "FoldTileBufIntrinsics: get_tensor_view_stride dim index out of "
            "bounds");
        return signalPassFailure();
      }

      builder.setInsertionPoint(strideOp);
      Value replacement = computeResultStride(
          builder, strideOp.getLoc(),
          chain->reinterpretCast.getMixedStrides()[dimIdx],
          chain->subview.getMixedStrides()[dimIdx]);

      strideOp.getResult().replaceAllUsesWith(replacement);
      strideOp.erase();
    }

    // Clean up dead unrealized_conversion_cast ops that bridged
    // memref -> partition_tensor_view / tile_buf and are now unused
    // after folding.
    SmallVector<UnrealizedConversionCastOp, 8> deadCasts;
    func.walk([&](UnrealizedConversionCastOp castOp) {
      if (castOp.use_empty() && castOp.getNumOperands() == 1 &&
          isa<MemRefType>(castOp.getOperand(0).getType()) &&
          isa<pto::PartitionTensorViewType, pto::TileBufType>(
              castOp.getResult(0).getType()))
        deadCasts.push_back(castOp);
    });
    for (auto castOp : llvm::reverse(deadCasts))
      castOp.erase();

    while (true) {
      SmallVector<Operation *, 8> deadMemrefOps;
      func.walk([&](Operation *op) {
        if ((isa<memref::SubViewOp>(op) ||
             isa<memref::ReinterpretCastOp>(op)) &&
            op->use_empty())
          deadMemrefOps.push_back(op);
      });
      if (deadMemrefOps.empty())
        break;
      for (auto *op : llvm::reverse(deadMemrefOps))
        op->erase();
    }
  }
};

} // namespace

namespace mlir {
namespace pto {

std::unique_ptr<Pass> createFoldTileBufIntrinsicsPass() {
  return std::make_unique<FoldTileBufIntrinsicsPass>();
}

} // namespace pto
} // namespace mlir
