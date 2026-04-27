"""Tests for BPE-aware in-scope name masks (Day 24)."""
from __future__ import annotations

import pytest

from decoder.inline_c3.bpe_masks import InScopeNameTrie, regex_prefix_mask


class _StubTokenizer:
    """Minimal tokenizer stub — character-level for predictability.

    vocab: [' ', ',', ':', '%', '0'..'9', 'a'..'z', 'A'..'Z', '_']. Each
    character is its own token at id = list index.
    """

    def __init__(self) -> None:
        chars = [" ", ",", ":", "%", "=", "_", "{", "}", ".", "(", ")",
                 "@", "-", ">", "<", "?", "[", "]"] + \
                [chr(c) for c in range(ord("0"), ord("9") + 1)] + \
                [chr(c) for c in range(ord("a"), ord("z") + 1)] + \
                [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        self._chars = chars
        self._map = {c: i for i, c in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self._chars)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._map[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._chars[i] for i in ids)


def test_trie_next_ids_from_empty() -> None:
    tok = _StubTokenizer()
    trie = InScopeNameTrie(tokenizer=tok, names=frozenset({"%a", "%b"}))
    # Nothing sampled yet — both names start with '%'
    legal = trie.legal_next_ids(())
    assert legal == {tok._map["%"]}


def test_trie_diverges_after_percent() -> None:
    tok = _StubTokenizer()
    trie = InScopeNameTrie(tokenizer=tok, names=frozenset({"%a", "%b"}))
    pct = (tok._map["%"],)
    legal = trie.legal_next_ids(pct)
    assert legal == {tok._map["a"], tok._map["b"]}


def test_trie_name_boundary() -> None:
    tok = _StubTokenizer()
    trie = InScopeNameTrie(tokenizer=tok, names=frozenset({"%a"}))
    assert not trie.is_name_boundary(())
    assert not trie.is_name_boundary((tok._map["%"],))
    assert trie.is_name_boundary((tok._map["%"], tok._map["a"]))


def test_trie_completable_names() -> None:
    tok = _StubTokenizer()
    trie = InScopeNameTrie(tokenizer=tok, names=frozenset({"%a", "%ab", "%b"}))
    # After '%a' two names remain completable
    path = (tok._map["%"], tok._map["a"])
    assert trie.completable_names(path) == {"%a", "%ab"}


def test_regex_prefix_mask_ssa_from_empty() -> None:
    """For the SSA regex, from empty text, only '%' should be a legal first
    token with our char-level tokenizer."""
    tok = _StubTokenizer()
    ssa_pattern = r"%[a-zA-Z0-9_][a-zA-Z0-9_]{0,31}"
    mask = regex_prefix_mask(tok, ssa_pattern, "")
    assert tok._map["%"] in mask
    # 'a' alone doesn't start a valid SSA name
    assert tok._map["a"] not in mask


def test_regex_prefix_mask_ssa_after_percent() -> None:
    tok = _StubTokenizer()
    ssa_pattern = r"%[a-zA-Z0-9_][a-zA-Z0-9_]{0,31}"
    mask = regex_prefix_mask(tok, ssa_pattern, "%")
    # Any alnum or '_' should now be legal
    assert tok._map["a"] in mask
    assert tok._map["Z"] in mask
    assert tok._map["0"] in mask
    assert tok._map["_"] in mask
    # A space does not continue a valid SSA name
    assert tok._map[" "] not in mask
