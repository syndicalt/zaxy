"""Unit + fuzz tests for the declarative notation codec."""

from __future__ import annotations

import random
import string

import pytest

from notation import MemoryFact, decode, encode


def _rt(fact: MemoryFact) -> MemoryFact:
    return decode(encode(fact))


def test_minimal():
    f = MemoryFact(domain="zaxy-default", etype="goal.created", actor="claude")
    assert _rt(f) == f


def test_scalar_types_preserved():
    f = MemoryFact(
        domain="d", etype="t", actor="a",
        attrs={"i": 63, "f": 3.14, "b": True, "bf": False, "z": None, "s": "hi"},
    )
    g = _rt(f)
    assert g == f
    # type identity, not just equality (True != 1 must hold structurally)
    assert isinstance(g.attrs["i"], int) and not isinstance(g.attrs["i"], bool)
    assert isinstance(g.attrs["b"], bool)
    assert g.attrs["z"] is None


def test_nested_json_value():
    f = MemoryFact(
        domain="d", etype="task.completed", actor="codex",
        attrs={"commits": ["a1", "b2"], "meta": {"k": [1, 2, {"x": True}]}},
    )
    assert _rt(f) == f


def test_special_chars_everywhere():
    nasty = "[]«»@{}():+,!Ø>>\\ /\n\t"
    f = MemoryFact(
        domain=nasty, etype=nasty, actor=nasty,
        entity=nasty,
        attrs={nasty: nasty, "k2": "v+w:z)]"},
        relations=[nasty, "x>>y"],
        invalidations=[nasty, "a,b"],
    )
    assert _rt(f) == f


def test_entity_relations_invalidations():
    f = MemoryFact(
        domain="zaxy-default", etype="memory.checkout.completed", actor="zaxy",
        entity="checkout@63899",
        attrs={"stale": True, "since_ev": 13917},
        relations=["memory_checkout"],
        invalidations=["session:zaxy"],
    )
    assert _rt(f) == f


def test_key_order_preserved():
    f = MemoryFact(
        domain="d", etype="t", actor="a",
        attrs={"z": 1, "a": 2, "m": 3},
    )
    assert list(_rt(f).attrs.keys()) == ["z", "a", "m"]


def test_unicode_value():
    f = MemoryFact(domain="d", etype="t", actor="a", attrs={"k": "héllo → wörld ✓ 日本語"})
    assert _rt(f) == f


def test_empty_string_fields():
    f = MemoryFact(domain="", etype="", actor="", entity="", attrs={"": ""})
    assert _rt(f) == f


def _rand_str(rng: random.Random) -> str:
    alphabet = string.ascii_letters + string.digits + "[]«»@{}():+,!Ø>\\ _.-/\n"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))


def _rand_value(rng: random.Random, depth: int = 0):
    choice = rng.randint(0, 6 if depth < 2 else 4)
    if choice == 0:
        return rng.randint(-(10**6), 10**6)
    if choice == 1:
        return rng.choice([True, False])
    if choice == 2:
        return None
    if choice == 3:
        return _rand_str(rng)
    if choice == 4:
        return round(rng.uniform(-1e3, 1e3), 6)
    if choice == 5:
        return [_rand_value(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return {_rand_str(rng): _rand_value(rng, depth + 1) for _ in range(rng.randint(0, 3))}


def test_fuzz_roundtrip():
    rng = random.Random(20260613)
    for _ in range(2000):
        attrs = {_rand_str(rng): _rand_value(rng) for _ in range(rng.randint(0, 5))}
        f = MemoryFact(
            domain=_rand_str(rng),
            etype=_rand_str(rng),
            actor=_rand_str(rng),
            entity=_rand_str(rng) if rng.random() < 0.5 else None,
            attrs=attrs,
            relations=[_rand_str(rng) for _ in range(rng.randint(0, 3))],
            invalidations=[_rand_str(rng) for _ in range(rng.randint(0, 3))],
        )
        assert _rt(f) == f


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
