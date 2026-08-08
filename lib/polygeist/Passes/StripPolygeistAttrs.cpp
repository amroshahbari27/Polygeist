#include "PassDetails.h"
#include "polygeist/Passes/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace polygeist;

namespace {
static void keepTTLAttrs(Operation *op) {
  SmallVector<NamedAttribute> keep;
  for (NamedAttribute attr : op->getAttrs())
    if (attr.getName().strref().starts_with("ttl."))
      keep.push_back(attr);
  op->setAttrs(DictionaryAttr::get(op->getContext(), keep));
}

static LogicalResult validateTTLKernel(func::FuncOp function) {
  llvm::DenseSet<StringRef> tensorNames;
  for (unsigned i = 0; i < function.getNumArguments(); ++i) {
    Attribute provenance = function.getArgAttr(i, "ttl.frontend_tensor");
    Type argumentType = function.getArgument(i).getType();
    if (isa<UnrankedMemRefType>(argumentType))
      return function.emitError()
             << "TTL frontend: tensor argument " << i
             << " must be a ranked memref declared with TTL_TENSOR_1D/2D/3D";
    auto type = dyn_cast<MemRefType>(argumentType);
    if (!type)
      return function.emitError()
             << "TTL frontend: kernel argument " << i
             << " must be a ranked memref declared with "
                "TTL_TENSOR_1D/2D/3D";
    if (!isa_and_nonnull<UnitAttr>(provenance))
      return function.emitError()
             << "TTL frontend: tensor argument " << i
             << " must be declared with TTL_TENSOR_1D/2D/3D";
    if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
        type.getRank() < 1 || type.getRank() > 3)
      return function.emitError()
             << "TTL frontend: tensor argument " << i
             << " must lower to a static rank-1/2/3 identity memref";

    auto argName = function.getArgAttrOfType<StringAttr>(i, "ttl.argname");
    if (!argName || argName.getValue().empty())
      return function.emitError()
             << "TTL frontend: tensor argument " << i
             << " has no source tensor name";
    for (unsigned j = 0; j < i; ++j) {
      auto previous =
          function.getArgAttrOfType<StringAttr>(j, "ttl.argname");
      if (previous && previous.getValue() == argName.getValue())
        return function.emitError()
               << "TTL frontend: duplicate tensor name '"
               << argName.getValue() << "'";
    }
    tensorNames.insert(argName.getValue());
    Attribute readonly = function.getArgAttr(i, "ttl.readonly");
    if (readonly && !isa<UnitAttr>(readonly))
      return function.emitError()
             << "TTL frontend: ttl.readonly must be a unit attribute";
  }

  bool scheduled = false;
  bool invalid = false;
  function.walk([&](Operation *operation) {
    if (invalid)
      return;

    StringRef dialect = operation->getName().getDialectNamespace();
    if (dialect == "scf" || dialect == "cf" || dialect == "gpu" ||
        dialect == "vector" || dialect == "polygeist") {
      operation->emitError()
          << "TTL frontend: unsupported control/data operation '"
          << operation->getName() << "'; use affine loops and accesses";
      invalid = true;
      return;
    }
    if (dialect == "memref" && !isa<memref::AllocaOp>(operation)) {
      operation->emitError()
          << "TTL frontend: non-affine memory operation '"
          << operation->getName()
          << "'; tensor accesses must lower to affine.load/affine.store";
      invalid = true;
      return;
    }
    if (auto alloca = dyn_cast<memref::AllocaOp>(operation)) {
      auto type = alloca.getType();
      if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
          type.getRank() > 3) {
        alloca.emitError(
            "TTL frontend: local storage must be a static rank-0/1/2/3 "
            "identity memref");
        invalid = true;
        return;
      }
    }
    if (dialect == "llvm" &&
        operation->getName().getStringRef() != "llvm.mlir.undef") {
      operation->emitError()
          << "TTL frontend: unsupported LLVM operation '"
          << operation->getName() << "'";
      invalid = true;
      return;
    }
    if (dialect != "affine" && dialect != "arith" && dialect != "func" &&
        dialect != "math" && dialect != "memref" && dialect != "llvm") {
      operation->emitError()
          << "TTL frontend: operation '" << operation->getName()
          << "' is outside the affine/memref source contract";
      invalid = true;
      return;
    }
    if (isa<func::CallOp>(operation)) {
      operation->emitError(
          "TTL frontend: helper calls must inline before structured lowering");
      invalid = true;
      return;
    }

    if (auto loop = dyn_cast<affine::AffineForOp>(operation)) {
      if (!isa_and_nonnull<UnitAttr>(
              loop->getAttr("ttl.frontend_loop"))) {
        loop.emitError(
            "TTL frontend: affine.for did not originate from "
            "TTL_LOOP_1D/2D/3D");
        invalid = true;
        return;
      }
    } else if (operation->hasAttr("ttl.frontend_loop")) {
      operation->emitError(
          "TTL frontend: ttl.frontend_loop is only valid on affine.for");
      invalid = true;
      return;
    }

    bool hasTile = operation->hasAttr("ttl.tile");
    bool hasSelections = operation->hasAttr("ttl.promote") ||
                         operation->hasAttr("ttl.double_buffer");
    if (!hasTile && !hasSelections)
      return;
    auto loop = dyn_cast<affine::AffineForOp>(operation);
    if (!loop) {
      operation->emitError(
          "TTL frontend: scheduling metadata requires affine.for");
      invalid = true;
      return;
    }
    if (!hasTile) {
      loop.emitError(
          "TTL frontend: tensor selections require a nonempty tile request");
      invalid = true;
      return;
    }
    llvm::DenseSet<StringRef> selectedNames;
    for (StringRef attribute : {"ttl.promote", "ttl.double_buffer"}) {
      Attribute raw = operation->getAttr(attribute);
      if (!raw)
        continue;
      auto names = dyn_cast<ArrayAttr>(raw);
      if (!names) {
        loop.emitError("TTL frontend: ")
            << attribute << " must be an array of tensor names";
        invalid = true;
        return;
      }
      for (Attribute entry : names) {
        auto name = dyn_cast<StringAttr>(entry);
        if (!name || name.getValue().empty()) {
          loop.emitError("TTL frontend: ")
              << attribute << " entries must be nonempty tensor names";
          invalid = true;
          return;
        }
        if (!tensorNames.contains(name.getValue())) {
          loop.emitError("TTL frontend: unknown selected tensor '")
              << name.getValue() << "'";
          invalid = true;
          return;
        }
        if (!selectedNames.insert(name.getValue()).second) {
          loop.emitError("TTL frontend: tensor '")
              << name.getValue()
              << "' is repeated or selected for both promotion modes";
          invalid = true;
          return;
        }
      }
    }
    auto tile = dyn_cast<ArrayAttr>(operation->getAttr("ttl.tile"));
    if (!tile || tile.empty() || tile.size() > 3 ||
        llvm::any_of(tile, [](Attribute entry) {
          auto integer = dyn_cast<IntegerAttr>(entry);
          return !integer || integer.getInt() <= 0;
        })) {
      loop.emitError(
          "TTL frontend: expected between one and three positive tile sizes");
      invalid = true;
      return;
    }
    scheduled = true;
  });

  if (invalid)
    return failure();
  if (!scheduled)
    return function.emitError(
        "TTL frontend: kernel must contain a structured TTL_LOOP_1D/2D/3D");
  return success();
}

static void removeFrontendProvenance(func::FuncOp function) {
  for (unsigned i = 0; i < function.getNumArguments(); ++i)
    function.removeArgAttr(i, "ttl.frontend_tensor");
  function.walk(
      [](affine::AffineForOp loop) { loop->removeAttr("ttl.frontend_loop"); });
}

struct StripPolygeistAttrsPass
    : public StripPolygeistAttrsPassBase<StripPolygeistAttrsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    keepTTLAttrs(module);

    // Identify the one requested kernel before erasing helper instantiations.
    SmallVector<func::FuncOp> kernels;
    SmallVector<func::FuncOp> helpers;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func->hasAttr("ttl.kernel"))
        kernels.push_back(func);
      else
        helpers.push_back(func);
    }

    if (kernels.size() != 1) {
      module.emitError() << "Expected exactly one ttl.kernel function in the "
                             "module, found " << kernels.size();
      signalPassFailure();
      return;
    }

    func::FuncOp mainFunc = kernels.front();
    if (failed(validateTTLKernel(mainFunc))) {
      signalPassFailure();
      return;
    }

    removeFrontendProvenance(mainFunc);
    for (func::FuncOp helper : helpers)
      helper.erase();
    mainFunc.setPrivate();

    keepTTLAttrs(mainFunc);
    mainFunc.setName(mainFunc.getName().str() + "_TTL_optimized");
  }
};
} // namespace

std::unique_ptr<Pass> polygeist::createStripPolygeistAttrsPass() {
  return std::make_unique<StripPolygeistAttrsPass>();
}
