//===- pragmaHandler.h - Pragmas used to emit MLIR---------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRCLANG_LIB_PRAGMAHANDLER_H
#define MLIR_TOOLS_MLIRCLANG_LIB_PRAGMAHANDLER_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"
#include "clang/Lex/Pragma.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

/// POD holds information processed from the lower_to pragma.
struct LowerToInfo {
  llvm::StringMap<std::string> SymbolTable;
  llvm::SmallVector<llvm::StringRef, 2> InputSymbol;
  llvm::SmallVector<llvm::StringRef, 2> OutputSymbol;
};

/// The location of the scop, as delimited by scop and endscop
/// pragmas by the user.
struct ScopLoc {
  ScopLoc() : end(0) {}

  clang::SourceLocation scop;
  clang::SourceLocation endscop;
  unsigned startLine;
  unsigned start;
  unsigned end;
};

/// List of pairs of #pragma scop and #pragma endscop locations.
struct ScopLocList {
  std::vector<ScopLoc> list;

  void addStart(clang::SourceManager &SM, clang::SourceLocation start) {
    ScopLoc loc;
    loc.scop = start;
    int line = SM.getExpansionLineNumber(start);
    start = SM.translateLineCol(SM.getFileID(start), line, 1);
    loc.startLine = line;
    loc.start = SM.getFileOffset(start);
    if (list.size() == 0 || list[list.size() - 1].end != 0)
      list.push_back(loc);
    else
      list[list.size() - 1] = loc;
  }

  void addEnd(clang::SourceManager &SM, clang::SourceLocation end) {
    if (list.size() == 0 || list[list.size() - 1].end != 0)
      return;
    list[list.size() - 1].endscop = end;
    int line = SM.getExpansionLineNumber(end);
    end = SM.translateLineCol(SM.getFileID(end), line + 1, 1);
    list[list.size() - 1].end = SM.getFileOffset(end);
  }

  bool isInScop(clang::SourceLocation target) {
    if (!list.size())
      return false;
    for (auto &scopLoc : list)
      if ((target >= scopLoc.scop) && (target <= scopLoc.endscop))
        return true;
    return false;
  }
};

//===----------------------------------------------------------------------===//
// TTL annotation structures
//===----------------------------------------------------------------------===//

/// The three attributes transported by one TTL_LOOP_*D invocation.
struct TTLLoopAttrs {
  /// Macro-expansion provenance used to bind this record to its physical loop.
  clang::SourceLocation sourceLocation;
  bool consumed = false;
  llvm::SmallVector<int64_t, 3> tileSizes;
  llvm::SmallVector<std::string, 4> promoteTensorNames;
  llvm::SmallVector<std::string, 4> doubleBufferTensorNames;
};

/// Carries loop attributes from preprocessing to affine-loop construction.
struct TTLAnnotationList {
  llvm::SmallVector<TTLLoopAttrs, 8> schedulingQueue;
};

/// Find the first requested macro in a source location's expansion chain, but
/// only when that invocation used the project definition from the same TTL.h
/// as the internal _TTL_NAME sentinel. This rejects simple user redefinitions
/// while still allowing harmless wrapper macros around the DSL spelling.
llvm::StringRef findProjectTTLMacroExpansion(
    clang::Preprocessor &PP, clang::SourceLocation location,
    llvm::ArrayRef<llvm::StringRef> requestedNames);

/// Return true when a token was spelled in the same pinned TTL.h that defines
/// the internal _TTL_NAME sentinel. Macro arguments spelled by the user do not
/// satisfy this predicate, even while they expand inside a TTL macro.
bool isProjectTTLDefinitionSpelling(clang::Preprocessor &PP,
                                    clang::SourceLocation location);

//===----------------------------------------------------------------------===//
// Handler registration
//===----------------------------------------------------------------------===//

void addPragmaLowerToHandlers(clang::Preprocessor &PP, LowerToInfo &LTInfo);
void addPragmaScopHandlers(clang::Preprocessor &PP, ScopLocList &scopLocList);
void addPragmaEndScopHandlers(clang::Preprocessor &PP,
                              ScopLocList &scopLocList);
void addPragmaTTLHandlers(clang::Preprocessor &PP,
                          TTLAnnotationList &annotations);

#endif
