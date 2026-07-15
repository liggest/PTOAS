// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOMATERIALIZETILEOPSECTIONS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr llvm::StringLiteral kTileOpHelperAttr = "pto.tileop.helper";
constexpr llvm::StringLiteral kTileOpKindAttr = "pto.tileop.kind";
constexpr llvm::StringLiteral kTileOpEffectsAttr = "pto.tileop.effects";

using ValidShapeRequirements = DenseMap<Operation *, SmallVector<unsigned, 2>>;
using ExpandedValidShapeArguments =
    DenseMap<Operation *, DenseMap<unsigned, std::pair<Value, Value>>>;

enum TileArgumentEffect : uint8_t {
  NoEffect = 0,
  ReadEffect = 1,
  WriteEffect = 2,
};

enum class TileOpKind { Vector, Cube };

static bool isScalarType(Type type) { return type.isIntOrIndexOrFloat(); }

static bool isTileOrScalarType(Type type) {
  return isa<TileBufType>(type) || isScalarType(type);
}

static bool isMemoryReferenceType(Type type) {
  return isa<TileBufType, PtrType, MemRefType, TensorViewType,
             PartitionTensorViewType>(type);
}

static bool isDirectSimtOperation(StringRef name) {
  return name.starts_with("pto.get_tid") ||
         name.starts_with("pto.get_block_dim") ||
         name.starts_with("pto.get_grid_dim") ||
         name.starts_with("pto.get_block_idx") ||
         name.starts_with("pto.get_subblock") || name == "pto.get_veccoreid" ||
         name.starts_with("pto.get_lane") ||
         name.starts_with("pto.get_clock") || name.starts_with("pto.vote_") ||
         name.starts_with("pto.shuffle_") || name.starts_with("pto.redux_") ||
         name.starts_with("pto.atomic_") || name == "pto.ldg" ||
         name == "pto.stg" || name == "pto.load" || name == "pto.store" ||
         name == "pto.store_vfsimt_info" || name == "pto.syncthreads" ||
         name.starts_with("pto.threadfence") || name == "pto.keep" ||
         name == "pto.resume";
}

static bool isForbiddenPipeOperation(Operation *op) {
  return isa<RecordEventOp, WaitEventOp, BarrierSyncOp, AicInitializePipeOp,
             AivInitializePipeOp, InitializeL2G2LPipeOp, InitializeL2LPipeOp,
             DeclareEventIdArrayOp, EventIdArrayGetOp, EventIdArraySetOp,
             SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, GetBufOp,
             RlsBufOp, SyncSetOp, SyncWaitOp, BarrierOp, FenceBarrierAllOp,
             TSyncOp, SyncAllOp, DsbOp>(op);
}

static bool isMteDataMovementOperation(StringRef name) {
  return name.starts_with("pto.mte_") || name.starts_with("pto.copy_") ||
         name.starts_with("pto.load_cbuf_to_") ||
         name.starts_with("pto.mgather") || name.starts_with("pto.mscatter");
}

static bool isVectorMicroOperation(StringRef name) {
  if (name == "pto.vecscope" || name == "pto.strict_vecscope")
    return false;
  return name.starts_with("pto.v") || name.starts_with("pto.pset_") ||
         name.starts_with("pto.pge_") || name.starts_with("pto.plt_") ||
         name.starts_with("pto.pltm_") || name.starts_with("pto.ppack") ||
         name.starts_with("pto.punpack") || name.starts_with("pto.pbitcast") ||
         name.starts_with("pto.pnot") || name.starts_with("pto.psel") ||
         name.starts_with("pto.pand") || name.starts_with("pto.por") ||
         name.starts_with("pto.pxor") || name.starts_with("pto.plds") ||
         name.starts_with("pto.pldi") || name.starts_with("pto.psti") ||
         name.starts_with("pto.psts") || name.starts_with("pto.pdintlv") ||
         name.starts_with("pto.pintlv") || name.starts_with("pto.pstu");
}

static bool isCubeMicroOperation(StringRef name) {
  return name.starts_with("pto.mad");
}

static bool isForbiddenTileOperation(StringRef name) {
  return name == "pto.alloc_tile" || name == "pto.alloc_multi_tile" ||
         name == "pto.materialize_tile" || name == "pto.reserve_buffer" ||
         (name.starts_with("pto.t") && name != "pto.tile_buf_addr" &&
          name != "pto.tile_valid_rows" && name != "pto.tile_valid_cols");
}

static std::optional<unsigned> traceToFunctionArgument(Value value,
                                                       func::FuncOp function) {
  for (unsigned depth = 0; value && depth < 64; ++depth) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (argument.getOwner() == &function.front())
        return argument.getArgNumber();
      return std::nullopt;
    }

    Operation *def = value.getDefiningOp();
    if (!def)
      return std::nullopt;

    Value next;
    if (auto tileAddr = dyn_cast<TileBufAddrOp>(def)) {
      next = tileAddr.getSrc();
    } else if (auto addPtr = dyn_cast<AddPtrOp>(def)) {
      next = addPtr.getPtr();
    } else if (auto castPtr = dyn_cast<CastPtrOp>(def)) {
      next = castPtr.getInput();
    } else if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(def)) {
      auto result = dyn_cast<OpResult>(value);
      if (result && result.getResultNumber() < unrealized.getNumOperands())
        next = unrealized.getOperand(result.getResultNumber());
    } else if (auto alias = getOperationAliasInfo(def)) {
      if (alias->first == value)
        next = alias->second;
    }

    if (!next || next == value)
      return std::nullopt;
    value = next;
  }
  return std::nullopt;
}

static void applyEffectToAllTileArguments(func::FuncOp function, uint8_t effect,
                                          SmallVectorImpl<uint8_t> &effects) {
  for (auto [index, type] : llvm::enumerate(function.getArgumentTypes()))
    if (isa<TileBufType>(type))
      effects[index] |= effect;
}

static SmallVector<uint8_t>
collectDirectArgumentEffects(func::FuncOp function) {
  SmallVector<uint8_t> effects(function.getNumArguments(), NoEffect);
  function.walk([&](Operation *op) {
    auto memoryEffects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memoryEffects)
      return WalkResult::advance();

    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 8>
        instances;
    memoryEffects.getEffects(instances);
    for (const auto &instance : instances) {
      uint8_t effect = NoEffect;
      if (isa<MemoryEffects::Read>(instance.getEffect()))
        effect = ReadEffect;
      else if (isa<MemoryEffects::Write>(instance.getEffect()))
        effect = WriteEffect;
      if (effect == NoEffect || !instance.getValue())
        continue;

      if (auto argument =
              traceToFunctionArgument(instance.getValue(), function))
        effects[*argument] |= effect;
      else if (isMemoryReferenceType(instance.getValue().getType()))
        applyEffectToAllTileArguments(function, effect, effects);
    }
    return WalkResult::advance();
  });
  return effects;
}

static void summarizeSimtLaunchEffects(func::FuncOp helper, SimtLaunchOp launch,
                                       SmallVectorImpl<uint8_t> &effects) {
  auto module = helper->getParentOfType<ModuleOp>();
  auto callee = module ? module.lookupSymbol<func::FuncOp>(launch.getCallee())
                       : func::FuncOp();
  if (!callee || callee.isDeclaration()) {
    applyEffectToAllTileArguments(helper, ReadEffect | WriteEffect, effects);
    return;
  }

  SmallVector<uint8_t> calleeEffects = collectDirectArgumentEffects(callee);
  for (auto [argument, calleeEffect] :
       llvm::zip_equal(launch.getArgs(), calleeEffects)) {
    if (calleeEffect == NoEffect)
      continue;
    if (auto helperArgument = traceToFunctionArgument(argument, helper))
      effects[*helperArgument] |= calleeEffect;
    else
      applyEffectToAllTileArguments(helper, calleeEffect, effects);
  }
}

static void summarizeTileOpEffects(func::FuncOp helper) {
  SmallVector<uint8_t> effects = collectDirectArgumentEffects(helper);
  helper.walk([&](SimtLaunchOp launch) {
    summarizeSimtLaunchEffects(helper, launch, effects);
  });

  SmallVector<Attribute> effectAttrs;
  effectAttrs.reserve(effects.size());
  for (auto [type, effect] :
       llvm::zip_equal(helper.getArgumentTypes(), effects)) {
    StringRef effectName = "none";
    if (isa<TileBufType>(type)) {
      if (effect == (ReadEffect | WriteEffect))
        effectName = "readwrite";
      else if (effect == ReadEffect)
        effectName = "read";
      else if (effect == WriteEffect)
        effectName = "write";
    }
    effectAttrs.push_back(StringAttr::get(helper.getContext(), effectName));
  }
  helper->setAttr(kTileOpEffectsAttr,
                  ArrayAttr::get(helper.getContext(), effectAttrs));
}

static bool addValidShapeRequirement(ValidShapeRequirements &requirements,
                                     func::FuncOp function,
                                     unsigned argumentIndex) {
  auto &indices = requirements[function.getOperation()];
  if (llvm::is_contained(indices, argumentIndex))
    return false;
  indices.push_back(argumentIndex);
  llvm::sort(indices);
  return true;
}

static LogicalResult
collectTileOpValidShapeRequirements(func::FuncOp helper,
                                    ValidShapeRequirements &requirements) {
  LogicalResult status = success();
  helper.walk([&](Operation *op) {
    if (failed(status) || !isa<TileValidRowsOp, TileValidColsOp>(op))
      return WalkResult::advance();

    Value source = op->getOperand(0);
    auto argument = dyn_cast<BlockArgument>(source);
    if (!argument || argument.getOwner() != &helper.front()) {
      status = op->emitError(
          "tileop valid-shape metadata must be read directly from a Tile "
          "argument");
      return WalkResult::interrupt();
    }

    auto tileType = dyn_cast<TileBufType>(argument.getType());
    if (!tileType || tileType.getValidShape().size() != 2) {
      status = op->emitError(
          "tileop valid-shape metadata requires a rank-2 Tile argument");
      return WalkResult::interrupt();
    }

    unsigned dimension = isa<TileValidRowsOp>(op) ? 0 : 1;
    if (tileType.getValidShape()[dimension] < 0)
      addValidShapeRequirement(requirements, helper, argument.getArgNumber());
    return WalkResult::advance();
  });
  return status;
}

static LogicalResult
propagateValidShapeRequirements(ModuleOp module,
                                ValidShapeRequirements &requirements) {
  SmallVector<func::CallOp> calls;
  module.walk([&](func::CallOp call) { calls.push_back(call); });

  bool changed = true;
  while (changed) {
    changed = false;
    for (func::CallOp call : calls) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call.getOperation(), call.getCalleeAttr());
      if (!callee)
        continue;
      auto required = requirements.find(callee.getOperation());
      if (required == requirements.end())
        continue;
      SmallVector<unsigned, 2> requiredArguments(required->second);

      auto caller = call->getParentOfType<func::FuncOp>();
      if (!caller)
        return call.emitOpError(
            "cannot propagate Tile valid-shape metadata without a caller "
            "function");
      for (unsigned calleeArgument : requiredArguments) {
        if (calleeArgument >= call.getNumOperands())
          return call.emitOpError(
              "TileOp call has fewer operands than its helper ABI");
        auto callerArgument =
            traceToFunctionArgument(call.getOperand(calleeArgument), caller);
        if (!callerArgument)
          continue;
        auto tileType = dyn_cast<TileBufType>(
            caller.getArgument(*callerArgument).getType());
        if (!tileType || !tileType.hasDynamicValid())
          continue;
        changed |=
            addValidShapeRequirement(requirements, caller, *callerArgument);
      }
    }
  }
  return success();
}

static LogicalResult expandValidShapeFunctionArguments(
    ValidShapeRequirements &requirements,
    ExpandedValidShapeArguments &expandedArguments) {
  for (auto &[functionOperation, argumentIndices] : requirements) {
    auto function = cast<func::FuncOp>(functionOperation);
    if (function.isExternal())
      return function.emitOpError(
          "cannot propagate dynamic Tile valid-shape metadata through an "
          "external function");

    unsigned originalArgumentCount = function.getNumArguments();
    SmallVector<unsigned> insertionIndices(argumentIndices.size() * 2,
                                           originalArgumentCount);
    SmallVector<Type> argumentTypes(argumentIndices.size() * 2,
                                    IndexType::get(function.getContext()));
    SmallVector<DictionaryAttr> argumentAttrs(
        argumentIndices.size() * 2, DictionaryAttr::get(function.getContext()));
    SmallVector<Location> argumentLocations(argumentIndices.size() * 2,
                                            function.getLoc());
    if (failed(function.insertArguments(insertionIndices, argumentTypes,
                                        argumentAttrs, argumentLocations)))
      return function.emitOpError(
          "failed to append internal Tile valid-shape ABI arguments");

    auto &functionArguments = expandedArguments[functionOperation];
    for (auto [position, originalIndex] : llvm::enumerate(argumentIndices)) {
      unsigned rowIndex = originalArgumentCount + position * 2;
      functionArguments[originalIndex] = std::make_pair(
          function.getArgument(rowIndex), function.getArgument(rowIndex + 1));
    }

    if (auto effects = function->getAttrOfType<ArrayAttr>(kTileOpEffectsAttr)) {
      SmallVector<Attribute> effectAttrs(effects.begin(), effects.end());
      auto none = StringAttr::get(function.getContext(), "none");
      effectAttrs.append(argumentIndices.size() * 2, none);
      function->setAttr(kTileOpEffectsAttr,
                        ArrayAttr::get(function.getContext(), effectAttrs));
    }
  }
  return success();
}

static std::pair<Value, Value> findValidShapeOverride(Value tile,
                                                      Operation *anchor) {
  std::pair<Value, Value> result;
  for (Operation *user : tile.getUsers()) {
    auto setValidShape = dyn_cast<SetValidShapeOp>(user);
    if (!setValidShape || setValidShape.getSource() != tile)
      continue;
    if (user->getBlock() != anchor->getBlock() ||
        !user->isBeforeInBlock(anchor))
      continue;
    result = {setValidShape.getValidRow(), setValidShape.getValidCol()};
  }
  return result;
}

static std::optional<std::pair<Value, Value>>
resolveCallValidShape(Value tile, Operation *anchor, func::FuncOp caller,
                      const ExpandedValidShapeArguments &expandedArguments,
                      OpBuilder &builder, unsigned depth = 0) {
  if (!tile || depth >= 64)
    return std::nullopt;

  auto tileType = dyn_cast<TileBufType>(tile.getType());
  if (tileType && tileType.getValidShape().size() == 2 &&
      tileType.getValidShape()[0] >= 0 && tileType.getValidShape()[1] >= 0) {
    Value row = builder.create<arith::ConstantIndexOp>(
        anchor->getLoc(), tileType.getValidShape()[0]);
    Value col = builder.create<arith::ConstantIndexOp>(
        anchor->getLoc(), tileType.getValidShape()[1]);
    return std::make_pair(row, col);
  }

  auto override = findValidShapeOverride(tile, anchor);
  if (override.first && override.second)
    return override;

  if (auto alloc = tile.getDefiningOp<AllocTileOp>()) {
    if (alloc.getValidRow() && alloc.getValidCol())
      return std::make_pair(alloc.getValidRow(), alloc.getValidCol());
  }
  if (auto materialize = tile.getDefiningOp<MaterializeTileOp>()) {
    if (materialize.getValidRow() && materialize.getValidCol())
      return std::make_pair(materialize.getValidRow(),
                            materialize.getValidCol());
  }

  if (auto callerArgument = traceToFunctionArgument(tile, caller)) {
    auto function = expandedArguments.find(caller.getOperation());
    if (function != expandedArguments.end()) {
      auto metadata = function->second.find(*callerArgument);
      if (metadata != function->second.end())
        return metadata->second;
    }
  }

  Operation *definingOperation = tile.getDefiningOp();
  if (!definingOperation)
    return std::nullopt;
  if (auto alias = getOperationAliasInfo(definingOperation)) {
    if (alias->first == tile)
      return resolveCallValidShape(alias->second, anchor, caller,
                                   expandedArguments, builder, depth + 1);
  }
  return std::nullopt;
}

static LogicalResult expandValidShapeCallOperands(
    ModuleOp module, const ValidShapeRequirements &requirements,
    const ExpandedValidShapeArguments &expandedArguments) {
  SmallVector<func::CallOp> calls;
  module.walk([&](func::CallOp call) { calls.push_back(call); });

  for (func::CallOp call : calls) {
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call.getOperation(), call.getCalleeAttr());
    if (!callee)
      continue;
    auto required = requirements.find(callee.getOperation());
    if (required == requirements.end())
      continue;

    auto caller = call->getParentOfType<func::FuncOp>();
    OpBuilder builder(call);
    SmallVector<Value> metadataOperands;
    for (unsigned argumentIndex : required->second) {
      if (argumentIndex >= call.getNumOperands())
        return call.emitOpError(
            "TileOp call has fewer operands than its helper ABI");
      auto metadata =
          resolveCallValidShape(call.getOperand(argumentIndex), call, caller,
                                expandedArguments, builder);
      if (!metadata)
        return call.emitOpError()
               << "cannot resolve dynamic valid_row/valid_col for Tile "
                  "operand #"
               << argumentIndex
               << "; pass a Tile produced by alloc_tile/materialize_tile or "
                  "forward its valid-shape metadata through the caller ABI";
      metadataOperands.push_back(metadata->first);
      metadataOperands.push_back(metadata->second);
    }
    call.getArgOperandsMutable().append(metadataOperands);
  }
  return success();
}

static LogicalResult replaceTileOpValidShapeReads(
    ArrayRef<func::FuncOp> helpers,
    const ExpandedValidShapeArguments &expandedArguments) {
  for (func::FuncOp helper : helpers) {
    SmallVector<Operation *> reads;
    helper.walk([&](Operation *op) {
      if (isa<TileValidRowsOp, TileValidColsOp>(op))
        reads.push_back(op);
    });

    for (Operation *read : reads) {
      auto argument = dyn_cast<BlockArgument>(read->getOperand(0));
      auto tileType =
          argument ? dyn_cast<TileBufType>(argument.getType()) : TileBufType();
      if (!argument || !tileType || tileType.getValidShape().size() != 2)
        return read->emitError(
            "tileop valid-shape metadata must be read directly from a rank-2 "
            "Tile argument");

      unsigned dimension = isa<TileValidRowsOp>(read) ? 0 : 1;
      OpBuilder builder(read);
      Value replacement;
      if (tileType.getValidShape()[dimension] >= 0) {
        replacement = builder.create<arith::ConstantIndexOp>(
            read->getLoc(), tileType.getValidShape()[dimension]);
      } else {
        auto function = expandedArguments.find(helper.getOperation());
        if (function == expandedArguments.end())
          return read->emitError(
              "missing internal Tile valid-shape ABI arguments");
        auto metadata = function->second.find(argument.getArgNumber());
        if (metadata == function->second.end())
          return read->emitError(
              "missing internal Tile valid-shape ABI arguments");
        replacement =
            dimension == 0 ? metadata->second.first : metadata->second.second;
      }
      read->getResult(0).replaceAllUsesWith(replacement);
      read->erase();
    }
  }
  return success();
}

static LogicalResult
materializeTileOpValidShapeABI(ModuleOp module,
                               ArrayRef<func::FuncOp> helpers) {
  ValidShapeRequirements requirements;
  for (func::FuncOp helper : helpers)
    if (failed(collectTileOpValidShapeRequirements(helper, requirements)))
      return failure();
  if (requirements.empty()) {
    ExpandedValidShapeArguments emptyArguments;
    return replaceTileOpValidShapeReads(helpers, emptyArguments);
  }

  if (failed(propagateValidShapeRequirements(module, requirements)))
    return failure();
  ExpandedValidShapeArguments expandedArguments;
  if (failed(
          expandValidShapeFunctionArguments(requirements, expandedArguments)) ||
      failed(expandValidShapeCallOperands(module, requirements,
                                          expandedArguments)) ||
      failed(replaceTileOpValidShapeReads(helpers, expandedArguments)))
    return failure();
  return success();
}

static LogicalResult verifyTileOpABI(func::FuncOp helper) {
  for (auto [index, type] : llvm::enumerate(helper.getArgumentTypes())) {
    if (!isTileOrScalarType(type))
      return helper.emitOpError()
             << "tileop argument #" << index
             << " must be !pto.tile_buf or a PTO scalar, got " << type;
  }
  if (helper.getNumResults() != 0)
    return helper.emitOpError("tileop helpers must not return values; write "
                              "results through mutable Tile parameters");
  return success();
}

static LogicalResult inferTileOpKind(func::FuncOp helper, TileOpKind &kind) {
  Operation *firstVector = nullptr;
  Operation *firstCube = nullptr;
  LogicalResult status = success();

  helper.walk([&](Operation *op) {
    if (failed(status) || op == helper.getOperation() ||
        isa<func::ReturnOp>(op))
      return WalkResult::advance();

    StringRef name = op->getName().getStringRef();
    if (isa<SectionCubeOp, SectionVectorOp>(op)) {
      status = op->emitError(
          "tileop helpers must not contain pre-existing sections");
      return WalkResult::interrupt();
    }
    if (isMteDataMovementOperation(name)) {
      status =
          op->emitError("tileop helpers must not contain MTE data movement");
      return WalkResult::interrupt();
    }
    if (isForbiddenPipeOperation(op)) {
      status =
          op->emitError("tileop helpers must not contain pipe synchronization");
      return WalkResult::interrupt();
    }
    if (isDirectSimtOperation(name)) {
      status = op->emitError("tileop helpers must launch a @pto.simt helper "
                             "instead of containing SIMT operations directly");
      return WalkResult::interrupt();
    }
    if (isForbiddenTileOperation(name)) {
      status = op->emitError("tileop helpers must not contain Tile allocation "
                             "or high-level TileOps");
      return WalkResult::interrupt();
    }
    if (isa<func::CallOp>(op)) {
      status = op->emitError("tileop helpers must not call another helper; use "
                             "pto.simt_launch for SIMT work");
      return WalkResult::interrupt();
    }
    if (isa<SimtLaunchOp>(op)) {
      firstVector = firstVector ? firstVector : op;
      return WalkResult::advance();
    }
    if (isVectorMicroOperation(name)) {
      firstVector = firstVector ? firstVector : op;
      return WalkResult::advance();
    }
    if (isCubeMicroOperation(name)) {
      firstCube = firstCube ? firstCube : op;
      return WalkResult::advance();
    }
    if (auto pipeOp = dyn_cast<OpPipeInterface>(op)) {
      switch (pipeOp.getPipe()) {
      case PIPE::PIPE_V:
        firstVector = firstVector ? firstVector : op;
        break;
      case PIPE::PIPE_M:
        firstCube = firstCube ? firstCube : op;
        break;
      default:
        status = op->emitError("tileop helpers may only contain Vector or Cube "
                               "compute operations");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (failed(status))
    return failure();
  if (firstVector && firstCube) {
    InFlightDiagnostic diag = helper.emitOpError(
        "mixes Vector and Cube compute operations in one tileop helper");
    diag.attachNote(firstVector->getLoc()) << "first Vector operation is here";
    diag.attachNote(firstCube->getLoc()) << "first Cube operation is here";
    return failure();
  }
  if (!firstVector && !firstCube)
    return helper.emitOpError("contains no Vector or Cube compute operation "
                              "from which to infer tileop kind");
  kind = firstVector ? TileOpKind::Vector : TileOpKind::Cube;
  return success();
}

static LogicalResult materializeTileOpSection(func::FuncOp helper,
                                              TileOpKind kind) {
  if (helper.empty() || !helper.getBody().hasOneBlock())
    return helper.emitOpError("requires a single-block helper body");
  Block &entry = helper.front();
  auto returnOp = dyn_cast<func::ReturnOp>(entry.getTerminator());
  if (!returnOp)
    return helper.emitOpError("requires a func.return terminator");

  SmallVector<Operation *> roots;
  for (Operation &op : entry.without_terminator())
    roots.push_back(&op);
  if (roots.empty())
    return helper.emitOpError("contains no materializable compute body");

  OpBuilder builder(roots.front());
  Operation *sectionOperation =
      kind == TileOpKind::Vector
          ? builder.create<SectionVectorOp>(roots.front()->getLoc())
                .getOperation()
          : builder.create<SectionCubeOp>(roots.front()->getLoc())
                .getOperation();
  Region &sectionBody = sectionOperation->getRegion(0);
  auto *sectionBlock = new Block();
  sectionBody.push_back(sectionBlock);

  for (Operation *root : roots)
    root->moveBefore(sectionBlock, sectionBlock->end());
  helper->setAttr(
      kTileOpKindAttr,
      StringAttr::get(helper.getContext(),
                      kind == TileOpKind::Vector ? "vector" : "cube"));
  return success();
}

static LogicalResult materializeTileOpHelper(func::FuncOp helper) {
  if (failed(verifyTileOpABI(helper)))
    return failure();
  TileOpKind kind;
  if (failed(inferTileOpKind(helper, kind)))
    return failure();
  summarizeTileOpEffects(helper);
  return materializeTileOpSection(helper, kind);
}

struct PTOMaterializeTileOpSectionsPass
    : public mlir::pto::impl::PTOMaterializeTileOpSectionsBase<
          PTOMaterializeTileOpSectionsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    LogicalResult status = success();
    SmallVector<func::FuncOp> helpers;
    module.walk([&](func::FuncOp helper) {
      if (!helper->hasAttr(kTileOpHelperAttr))
        return WalkResult::advance();
      helpers.push_back(helper);
      if (failed(materializeTileOpHelper(helper))) {
        status = failure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failed(status) ||
        failed(materializeTileOpValidShapeABI(module, helpers)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOMaterializeTileOpSectionsPass() {
  return std::make_unique<PTOMaterializeTileOpSectionsPass>();
}
