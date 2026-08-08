//===- pragmaHandler.cc - Pragmas used to emit MLIR--------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pragmaHandler.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang;
using namespace llvm;

static const MacroInfo *getMacroDefinitionAtExpansion(
    Preprocessor &PP, StringRef macroName, SourceLocation location) {
  if (location.isInvalid())
    return nullptr;
  SourceLocation useLocation =
      PP.getSourceManager().getExpansionLoc(location);
  return PP.getMacroDefinitionAtLoc(PP.getIdentifierInfo(macroName),
                                    useLocation)
      .getMacroInfo();
}

static bool isProjectTTLMacroDefinition(Preprocessor &PP, StringRef macroName,
                                        SourceLocation location) {
  const MacroInfo *definition =
      getMacroDefinitionAtExpansion(PP, macroName, location);
  const MacroInfo *sentinel =
      getMacroDefinitionAtExpansion(PP, "_TTL_NAME", location);
  if (!definition || !sentinel)
    return false;

  SourceManager &sourceManager = PP.getSourceManager();
  SourceLocation definitionLocation =
      sourceManager.getSpellingLoc(definition->getDefinitionLoc());
  SourceLocation sentinelLocation =
      sourceManager.getSpellingLoc(sentinel->getDefinitionLoc());
  if (definitionLocation.isInvalid() || sentinelLocation.isInvalid() ||
      sourceManager.isWrittenInMainFile(sentinelLocation))
    return false;
  return sourceManager.getFileID(definitionLocation) ==
         sourceManager.getFileID(sentinelLocation);
}

bool isProjectTTLDefinitionSpelling(Preprocessor &PP,
                                    SourceLocation location) {
  const MacroInfo *sentinel =
      getMacroDefinitionAtExpansion(PP, "_TTL_NAME", location);
  if (!sentinel)
    return false;

  SourceManager &sourceManager = PP.getSourceManager();
  SourceLocation spellingLocation = sourceManager.getSpellingLoc(location);
  SourceLocation sentinelLocation =
      sourceManager.getSpellingLoc(sentinel->getDefinitionLoc());
  if (spellingLocation.isInvalid() || sentinelLocation.isInvalid() ||
      sourceManager.isWrittenInMainFile(sentinelLocation))
    return false;
  return sourceManager.getFileID(spellingLocation) ==
         sourceManager.getFileID(sentinelLocation);
}

StringRef findProjectTTLMacroExpansion(
    Preprocessor &PP, SourceLocation location,
    ArrayRef<StringRef> requestedNames) {
  SourceManager &sourceManager = PP.getSourceManager();
  while (location.isMacroID()) {
    StringRef macroName = Lexer::getImmediateMacroName(
        location, sourceManager, PP.getLangOpts());
    for (StringRef requestedName : requestedNames)
      if (macroName == requestedName &&
          isProjectTTLMacroDefinition(PP, macroName, location))
        return macroName;
    SourceLocation caller =
        sourceManager.getImmediateMacroCallerLoc(location);
    if (caller == location)
      break;
    location = caller;
  }
  return {};
}

namespace {

//===----------------------------------------------------------------------===//
// #pragma lower_to handler (existing)
//===----------------------------------------------------------------------===//

class PragmaLowerToHandler : public PragmaHandler {
  LowerToInfo &Info;

public:
  PragmaLowerToHandler(LowerToInfo &Info)
      : PragmaHandler("lower_to"), Info(Info) {}

  bool ReadInputOrOutput(Preprocessor &PP, Token &CurrentTok,
                         SmallVectorImpl<StringRef> &Ids) {
    PP.Lex(CurrentTok);
    if (CurrentTok.isNot(tok::l_paren)) {
      PP.Diag(CurrentTok.getLocation(), diag::warn_pragma_expected_lparen)
          << "lower_to";
      return false;
    }
    PP.Lex(CurrentTok);
    while (CurrentTok.isNot(tok::r_paren) && CurrentTok.isNot(tok::eod)) {
      if (CurrentTok.isNot(tok::identifier)) {
        PP.Diag(CurrentTok.getLocation(), diag::warn_pragma_expected_identifier)
            << "lower_to";
        return false;
      } else {
        StringRef Id = CurrentTok.getIdentifierInfo()->getName();
        Ids.push_back(Id);
      }
      PP.Lex(CurrentTok);
      if (CurrentTok.is(tok::r_paren))
        break;
      if (CurrentTok.isNot(tok::comma)) {
        PP.Diag(CurrentTok.getLocation(), diag::warn_pragma_expected_comma)
            << "lower_to";
        return false;
      }
      PP.Lex(CurrentTok);
    }
    PP.Lex(CurrentTok);
    return true;
  }

  bool HandleoptionalInputAndOutput(Preprocessor &PP, Token &PragmaTok,
                                    SmallVectorImpl<StringRef> &Inputs,
                                    SmallVectorImpl<StringRef> &Outputs) {
    Token CurrentTok;
    PP.Lex(CurrentTok);
    if (CurrentTok.is(tok::eod))
      return true;
    if (CurrentTok.isNot(tok::string_literal) ||
        (StringRef(CurrentTok.getLiteralData(), CurrentTok.getLength())
             .compare("input") == 0)) {
      PP.Diag(CurrentTok.getLocation(),
              diag::warn_pragma_expected_section_label_or_name)
          << "";
      return false;
    } else {
      if (!ReadInputOrOutput(PP, CurrentTok, Inputs))
        return false;
    }
    if (CurrentTok.isNot(tok::comma)) {
      PP.Diag(CurrentTok.getLocation(), diag::warn_pragma_expected_comma)
          << "lower_to";
      return false;
    }
    PP.Lex(CurrentTok);
    if (CurrentTok.isNot(tok::string_literal) ||
        (StringRef(CurrentTok.getLiteralData(), CurrentTok.getLength())
             .compare("output") == 0)) {
      PP.Diag(CurrentTok.getLocation(),
              diag::warn_pragma_expected_section_label_or_name)
          << "expect 'output' for lower to";
      return false;
    } else {
      if (!ReadInputOrOutput(PP, CurrentTok, Outputs))
        return false;
    }
    if (CurrentTok.isNot(tok::eod)) {
      PP.Diag(CurrentTok.getLocation(), diag::warn_pragma_extra_tokens_at_eol)
          << "lower_to";
      return false;
    }
    return true;
  }

  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &PragmaTok) override {
    Token Tok;
    PP.Lex(Tok);
    if (Tok.isNot(tok::l_paren)) {
      PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_lparen)
          << "lower_to";
      return;
    }
    Token PrevTok = Tok;
    llvm::StringRef FuncId = llvm::StringRef();
    llvm::StringRef SymbolName = llvm::StringRef();
    while (Tok.isNot(tok::eod)) {
      Token CurrentTok;
      PP.Lex(CurrentTok);
      if (PrevTok.is(tok::string_literal)) {
        if (CurrentTok.isNot(tok::r_paren)) {
          PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_rparen)
              << "lower_to";
          return;
        } else {
          if (!HandleoptionalInputAndOutput(PP, CurrentTok, Info.InputSymbol,
                                            Info.OutputSymbol))
            return;
          else
            break;
        }
      }
      if (PrevTok.is(tok::l_paren)) {
        if (CurrentTok.isNot(tok::identifier)) {
          PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_identifier)
              << "lower_to";
          return;
        } else {
          FuncId = CurrentTok.getIdentifierInfo()->getName();
        }
      }
      if (PrevTok.is(tok::identifier)) {
        if (CurrentTok.isNot(tok::comma)) {
          PP.Diag(Tok.getLocation(), diag::warn_pragma_expected_comma)
              << "lower_to";
          return;
        }
      }
      if (PrevTok.is(tok::comma)) {
        if (CurrentTok.isNot(tok::string_literal)) {
          PP.Diag(CurrentTok.getLocation(),
                  diag::warn_pragma_expected_section_name)
              << "lower to";
          return;
        } else {
          SmallVector<Token, 1> SymbolToks;
          SymbolToks.push_back(CurrentTok);
          SymbolName = StringLiteralParser(SymbolToks, PP).GetString();
        }
      }
      PrevTok = CurrentTok;
    }
    auto result = Info.SymbolTable.try_emplace(FuncId, SymbolName);
    assert(result.second &&
           "Shouldn't define lower_to over the same func id more than once.");
  }
};

//===----------------------------------------------------------------------===//
// #pragma scop / endscop handlers (existing)
//===----------------------------------------------------------------------===//

struct PragmaScopHandler : public PragmaHandler {
  ScopLocList &scops;
  PragmaScopHandler(ScopLocList &scops) : PragmaHandler("scop"), scops(scops) {}
  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &scopTok) override {
    scops.addStart(PP.getSourceManager(), scopTok.getLocation());
  }
};

struct PragmaEndScopHandler : public PragmaHandler {
  ScopLocList &scops;
  PragmaEndScopHandler(ScopLocList &scops)
      : PragmaHandler("endscop"), scops(scops) {}
  void HandlePragma(Preprocessor &PP, PragmaIntroducer introducer,
                    Token &endScopTok) override {
    scops.addEnd(PP.getSourceManager(), endScopTok.getLocation());
  }
};

//===----------------------------------------------------------------------===//
// #pragma ttl handler. The supported scheduling protocol is deliberately
// limited to tile sizes plus two explicit tensor-selection lists:
//   #pragma ttl tiled_pipeline_mb(s0, s1 [, s2])(promote_only0, ...)(promote_db0, ...)
//===----------------------------------------------------------------------===//

static void ttlError(Preprocessor &PP, SourceLocation location,
                     const Twine &message) {
  unsigned id = PP.getDiagnostics().getCustomDiagID(
      DiagnosticsEngine::Error, "TTL structured loop: %0");
  PP.Diag(location, id) << message.str();
}

/// Parse one strict parenthesized comma-separated list.
static bool parseParen(Preprocessor &PP, SmallVectorImpl<std::string> &args,
                       bool numbers, bool allowEmpty, StringRef description) {
  Token T;
  PP.Lex(T);
  if (T.isNot(tok::l_paren)) {
    ttlError(PP, T.getLocation(), "expected '(' before " + description);
    return false;
  }

  bool expectValue = true;
  bool sawValue = false;
  while (true) {
    PP.Lex(T);
    if (T.is(tok::r_paren)) {
      if (expectValue && sawValue) {
        ttlError(PP, T.getLocation(),
                 "trailing comma in " + description);
        return false;
      }
      if (!allowEmpty && !sawValue) {
        ttlError(PP, T.getLocation(), description + " cannot be empty");
        return false;
      }
      break;
    }
    if (T.is(tok::eod)) {
      ttlError(PP, T.getLocation(), "expected ')' after " + description);
      return false;
    }

    if (!expectValue) {
      if (T.isNot(tok::comma)) {
        ttlError(PP, T.getLocation(),
                 "expected ',' between entries in " + description);
        return false;
      }
      expectValue = true;
      continue;
    }

    if ((numbers && T.is(tok::numeric_constant)) ||
        (!numbers && T.is(tok::identifier))) {
      if (numbers)
        args.push_back(PP.getSpelling(T));
      else
        args.push_back(T.getIdentifierInfo()->getName().str());
    } else {
      ttlError(PP, T.getLocation(),
               (numbers ? "expected a decimal integer in "
                        : "expected a tensor name in ") +
                   description);
      return false;
    }
    sawValue = true;
    expectValue = false;
  }
  return true;
}

struct PragmaTTLHandler : public PragmaHandler {
  TTLAnnotationList &annotations;

  PragmaTTLHandler(TTLAnnotationList &annotations)
      : PragmaHandler("ttl"), annotations(annotations) {}

  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &PragmaTok) override {
    Token Tok;
    PP.Lex(Tok);

    if (Tok.isNot(tok::identifier) ||
        Tok.getIdentifierInfo()->getName() != "tiled_pipeline_mb") {
      PP.Diag(Tok.getLocation(), diag::err_expected) << "tiled_pipeline_mb";
      return;
    }
    handleTiledPipelineMB(PP, PragmaTok);
  }

private:
  /// #pragma ttl tiled_pipeline_mb(s0, s1 [, s2])(promote_only0, ...)(promote_db0, ...)
  ///
  /// One record carries tile sizes, promote names, and double-buffer names.
  void handleTiledPipelineMB(Preprocessor &PP, Token &PragmaTok) {
    TTLLoopAttrs attrs;
    attrs.sourceLocation = PragmaTok.getLocation();
    SmallVector<std::string, 3> tileArgs;
    if (!parseParen(PP, tileArgs, /*numbers=*/true, /*allowEmpty=*/false,
                    "tile-size list"))
      return;
    if (!parseParen(PP, attrs.promoteTensorNames, /*numbers=*/false,
                    /*allowEmpty=*/true, "promote list"))
      return;
    if (!parseParen(PP, attrs.doubleBufferTensorNames, /*numbers=*/false,
                    /*allowEmpty=*/true, "double-buffer list"))
      return;

    Token end;
    PP.Lex(end);
    if (end.isNot(tok::eod)) {
      ttlError(PP, end.getLocation(),
               "unexpected tokens after double-buffer list");
      return;
    }

    if (tileArgs.size() > 3) {
      ttlError(PP, PragmaTok.getLocation(),
               "expected between one and three tile sizes");
      return;
    }

    for (const auto &s : tileArgs) {
      int64_t v;
      if (StringRef(s).getAsInteger(10, v) || v <= 0) {
        ttlError(PP, PragmaTok.getLocation(),
                 "tile size '" + s + "' must be a positive decimal integer");
        return;
      }
      attrs.tileSizes.push_back(v);
    }
    annotations.schedulingQueue.push_back(std::move(attrs));
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Registration functions
//===----------------------------------------------------------------------===//

void addPragmaLowerToHandlers(Preprocessor &PP, LowerToInfo &LTInfo) {
  PP.AddPragmaHandler(new PragmaLowerToHandler(LTInfo));
}

void addPragmaScopHandlers(Preprocessor &PP, ScopLocList &scopLocList) {
  PP.AddPragmaHandler(new PragmaScopHandler(scopLocList));
}

void addPragmaEndScopHandlers(Preprocessor &PP, ScopLocList &scopLocList) {
  PP.AddPragmaHandler(new PragmaEndScopHandler(scopLocList));
}

void addPragmaTTLHandlers(Preprocessor &PP, TTLAnnotationList &annotations) {
  PP.AddPragmaHandler(new PragmaTTLHandler(annotations));
}
