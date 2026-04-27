"""BPE-aware token masks for in-line C3 decoding (Day 24 of ADR-0008).

Two mask constructions:

1. `regex_prefix_mask(tokenizer, pattern, prefix_text) -> set[int]`
   Token ids that when appended to `prefix_text` keep the concatenation a
   valid prefix of `pattern`. Used for fresh SSA names and other
   regex-matched terminals at each decode step.

2. `InScopeNameTrie(tokenizer, names)` — a trie over the BPE tokenizations
   of `names`. `.legal_next_ids(path_so_far) -> set[int]` returns the
   token ids that extend `path_so_far` toward at least one name in the
   set; `.is_name_boundary(path_so_far) -> bool` reports whether the path
   completes a full name and is a natural break point.

Design notes:
  - Tokenizer agnostic: accepts any object with `.encode(text) -> list[int]`
    and `.decode([id]) -> str`.
  - Vocab scans are cached per (tokenizer-id, pattern) so repeated queries
    during decoding are O(1) lookups.
  - We deliberately do not integrate MLX here — this module returns plain
    Python set[int], and Day 25 builds the mx.array mask.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

try:
    import regex as _regex_mod  # partial-match-capable regex engine
    _HAS_REGEX_MOD = True
except ImportError:
    _HAS_REGEX_MOD = False


def _vocab_pairs(tokenizer: Any) -> list[tuple[int, str]]:
    """Return (id, decoded_text) pairs for every vocab id. Cached per tokenizer."""
    n = getattr(tokenizer, "vocab_size", None)
    if n is None:
        vocab = tokenizer.get_vocab()
        return [(i, s) for s, i in vocab.items()]
    out: list[tuple[int, str]] = []
    for i in range(n):
        try:
            s = tokenizer.decode([i])
        except Exception:
            s = ""
        out.append((i, s))
    return out


_VOCAB_CACHE: dict[int, list[tuple[int, str]]] = {}


def _get_vocab(tokenizer: Any) -> list[tuple[int, str]]:
    key = id(tokenizer)
    if key not in _VOCAB_CACHE:
        _VOCAB_CACHE[key] = _vocab_pairs(tokenizer)
    return _VOCAB_CACHE[key]


def regex_prefix_mask(tokenizer: Any, pattern: str, prefix_text: str) -> set[int]:
    """Token ids that keep the concatenation `prefix_text + decode(id)` a
    valid prefix of strings matched by `pattern`.

    `pattern` is a regex source string. We wrap it with `\\A(?:pattern).*\\Z`
    semantics: the candidate must be extendable to a full match. Concretely:
    for each vocab id, we test whether `prefix_text + tok_text` is the
    prefix of SOME string that would match the full pattern.

    Approximation: we use `re.match(partial_regex, prefix+tok)` where
    `partial_regex` is built greedily. Python's `re` doesn't expose a
    partial-match flag, so we implement a best-effort via regex constructor
    that allows the match to be incomplete.
    """
    full = re.compile(rf"\A(?:{pattern})\Z")
    # Build an "acceptable as prefix" regex by making the overall pattern optional
    # beyond the matched portion — simple approximation: require every prefix of
    # the candidate to match a prefix of the pattern.
    # For most practical grammar patterns (identifier-like), this is handled by
    # regex_prefix_accept below.
    out: set[int] = set()
    vocab = _get_vocab(tokenizer)
    for tid, text in vocab:
        if not text:
            continue
        candidate = prefix_text + text
        if full.match(candidate):
            out.add(tid)
            continue
        # Partial: try to see if *any* extension could match. Heuristic: if
        # the candidate matches an anchored prefix of a more permissive form.
        if _regex_prefix_accept(pattern, candidate):
            out.add(tid)
    return out


def _regex_prefix_accept(pattern: str, candidate: str) -> bool:
    """True iff `candidate` could be the start of a string matching `pattern`.

    Uses the third-party `regex` module's `partial=True` flag when
    available (exact for any regex shape). Falls back to a conservative
    per-shape handler if `regex` isn't installed.

    DEFAULT: conservative. If we don't know the pattern shape, return
    False — over-permissive fallback (returning True) was the root of
    the Day-39 'modulemodule' bug: it made the WS mask include every
    token, defeating constraint masking after any terminal closure.
    """
    # Use \A...\Z anchors (not ^...$) — $ matches before a trailing
    # newline in Python's re and regex modules, which falsely accepts
    # over-length whitespace partials against bounded patterns like
    # [ \t\n]{1,8}. \Z is strictly end-of-string.
    if _HAS_REGEX_MOD:
        try:
            m = _regex_mod.compile(rf"\A(?:{pattern})\Z")
            return bool(m.match(candidate, partial=True))
        except _regex_mod.error:
            pass
    try:
        if re.compile(rf"\A(?:{pattern})\Z").match(candidate):
            return True
    except re.error:
        return False
    return _progressive_regex_accept(pattern, candidate)


def _progressive_regex_accept(pattern: str, candidate: str) -> bool:
    """Conservative partial-prefix matcher for the specific regex shapes
    used in our grammars. Returns False for unrecognized shapes (NOT
    pessimistically True — see Day 39 notes).
    """
    m = re.match(r"%\[([^\]]+)\]\[([^\]]+)\]\{0,(\d+)\}", pattern)
    if m:
        first_class = m.group(1)
        rest_class = m.group(2)
        maxlen = int(m.group(3)) + 1
        if not candidate.startswith("%"):
            return False
        body = candidate[1:]
        if len(body) > maxlen:
            return False
        if body and not re.fullmatch(f"[{first_class}][{rest_class}]*", body):
            return False
        return True
    m = re.match(r"@\[([^\]]+)\]\[([^\]]+)\]\{0,(\d+)\}", pattern)
    if m:
        first_class = m.group(1)
        rest_class = m.group(2)
        maxlen = int(m.group(3)) + 1
        if not candidate.startswith("@"):
            return False
        body = candidate[1:]
        if len(body) > maxlen:
            return False
        if body and not re.fullmatch(f"[{first_class}][{rest_class}]*", body):
            return False
        return True
    if pattern == r"\s+":
        return bool(candidate) and all(c.isspace() for c in candidate)
    if pattern in (r"\d+", r"-?\d+"):
        body = candidate[1:] if candidate.startswith("-") and pattern.startswith("-?") else candidate
        return bool(body) and body.isdigit()
    return False


@dataclass
class InScopeNameTrie:
    """Trie over BPE tokenizations of a set of names.

    Usage (Day 25 sampling loop):
      trie = InScopeNameTrie(tokenizer, {"%a", "%b", "%0"})
      # path_so_far is the list of token ids already sampled for this SSA
      legal = trie.legal_next_ids(path_so_far)
      # sample an id from (legal ∩ logits_mask_from_grammar)
    """
    tokenizer: Any
    names: frozenset[str]
    _paths: list[tuple[int, ...]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_paths", [
            tuple(self.tokenizer.encode(n, add_special_tokens=False))
            for n in sorted(self.names)
        ])

    def legal_next_ids(self, path_so_far: tuple[int, ...]) -> set[int]:
        out: set[int] = set()
        for p in self._paths:
            if len(p) <= len(path_so_far):
                continue
            if p[: len(path_so_far)] == path_so_far:
                out.add(p[len(path_so_far)])
        return out

    def is_name_boundary(self, path_so_far: tuple[int, ...]) -> bool:
        """True if `path_so_far` is exactly the tokenization of some name."""
        return any(p == path_so_far for p in self._paths)

    def completable_names(self, path_so_far: tuple[int, ...]) -> set[str]:
        """Names whose tokenization is extended by `path_so_far`."""
        out: set[str] = set()
        names_sorted = sorted(self.names)
        for i, p in enumerate(self._paths):
            if len(p) < len(path_so_far):
                continue
            if p[: len(path_so_far)] == path_so_far:
                out.add(names_sorted[i])
        return out
