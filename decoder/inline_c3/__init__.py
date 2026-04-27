"""In-line C3 decoding: coupled (CFG state × SSA symbol table) token-level
masking without post-hoc rejection sampling.

Public API (Day 22):
  - CFGAutomaton: compile a LARK grammar to an interactive-parser wrapper
    exposing {legal_next_terminals(), advance(token_type, value)}.

Coming next:
  - CoupledState (Day 23): CFGAutomaton × symbol-table transitions, with
    SSA_DEF / SSA_USE differentiation.
  - BPEVocabIndex (Day 24): maps regex-per-terminal to BPE token-id masks,
    with a trie for in-scope-SSA completion.
  - MLXSamplingLoop (Day 25): integrates CoupledState + BPEVocabIndex into
    the MLX sampling loop, bypassing Outlines.
"""
from .cfg_automaton import CFGAutomaton, CFGState, TerminalSpec
from .coupled_state import (
    CoupledState, SymbolTable, SSARole, OpPhase, ScopeViolation,
)
from .bpe_masks import InScopeNameTrie, regex_prefix_mask
from .generate import generate_text_mock, InlineC3Result

try:
    from .mlx_generate import (
        mlx_generate_coupled,
        mlx_generate_coupled_annealed,
        MLXGenerateResult,
    )
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

__all__ = [
    "CFGAutomaton", "CFGState", "TerminalSpec",
    "CoupledState", "SymbolTable", "SSARole", "OpPhase", "ScopeViolation",
    "InScopeNameTrie", "regex_prefix_mask",
    "generate_text_mock", "InlineC3Result",
]
if _HAS_MLX:
    __all__ += ["mlx_generate_coupled", "mlx_generate_coupled_annealed",
                "MLXGenerateResult"]
