"""Declarative compact-notation codec for Zaxy memory events (experiment).

Hypothesis under test: declarative event STATE can be encoded losslessly into a
dense symbolic notation iff (a) closed vocabulary, (b) bijective token<->field,
(c) read-not-generate. This module implements the codec and the event<->fact
projection so rungs 1 (token density) and 2 (round-trip fidelity) can measure it.

Scope decision (important, and itself a finding): the notation encodes the
*declarative content* of an event -- type, actor, thread/domain, and payload --
NOT the immutable integrity envelope (seq, timestamp, prev_hash, hash, security,
envelope_version). Those are cryptographic metadata, not facts to re-author, so a
human-facing notation should not own them. Round-trip fidelity is therefore
defined over the declarative content, and full-envelope reproduction is reported
separately as out-of-scope-by-design.

Grammar (canonical, compact form):

    [domain]«etype»@actor{entity}(k:Tv+k:Tv) >>rel !Ø(inv,inv)

  [domain]   thread/domain scope        (always present)
  «etype»    event type                 (always present)
  @actor     emitting actor             (always present)
  {entity}   optional identity label    (omitted if None)
  (...)      attributes, order-preserved (omitted if empty)
  >>rel      relation: causes/precedes   (zero or more)
  !Ø(...)    invalidations/exclusions    (omitted if empty)

Value type codes (preserve scalar identity through the round-trip):
  s string  i int  f float  b bool(T/F)  z null  j json(list/dict)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Structural metacharacters. Any occurrence inside leaf text (domain, etype,
# actor, entity, keys, values, relation targets, invalidations) is backslash-
# escaped so the scanner can treat unescaped occurrences as delimiters.
_SPECIAL = set("\\[]«»@{}():+,!Ø>")


def _esc(text: str) -> str:
    out = []
    for ch in text:
        if ch in _SPECIAL:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _scan_leaf(s: str, i: int, stops: set[str]) -> tuple[str, int]:
    """Read an escaped leaf starting at i until an unescaped char in `stops`.

    Returns (decoded_text, index_of_stop_char).
    """
    out = []
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\":
            if i + 1 >= n:
                raise ValueError("dangling escape")
            out.append(s[i + 1])
            i += 2
            continue
        if ch in stops:
            break
        out.append(ch)
        i += 1
    return "".join(out), i


# ---- scalar value codec ---------------------------------------------------

def _enc_value(v: Any) -> str:
    if isinstance(v, bool):  # bool before int -- bool is a subclass of int
        return "b" + ("T" if v else "F")
    if isinstance(v, int):
        return "i" + str(v)
    if isinstance(v, float):
        return "f" + repr(v)
    if v is None:
        return "z"
    if isinstance(v, str):
        return "s" + _esc(v)
    if isinstance(v, (list, dict)):
        return "j" + _esc(json.dumps(v, ensure_ascii=False, separators=(",", ":")))
    raise TypeError(f"unsupported value type: {type(v).__name__}")


def _dec_value(code: str, text: str) -> Any:
    if code == "b":
        return text == "T"
    if code == "i":
        return int(text)
    if code == "f":
        return float(text)
    if code == "z":
        return None
    if code == "s":
        return text
    if code == "j":
        return json.loads(text)
    raise ValueError(f"unknown value code: {code!r}")


# ---- MemoryFact -----------------------------------------------------------

@dataclass
class MemoryFact:
    domain: str
    etype: str
    actor: str
    attrs: dict[str, Any] = field(default_factory=dict)
    entity: str | None = None
    relations: list[str] = field(default_factory=list)
    invalidations: list[str] = field(default_factory=list)


def encode(fact: MemoryFact) -> str:
    parts = [f"[{_esc(fact.domain)}]«{_esc(fact.etype)}»@{_esc(fact.actor)}"]
    if fact.entity is not None:
        parts.append("{" + _esc(fact.entity) + "}")
    if fact.attrs:
        inner = "+".join(
            f"{_esc(k)}:{_enc_value(v)}" for k, v in fact.attrs.items()
        )
        parts.append("(" + inner + ")")
    for rel in fact.relations:
        parts.append(">>" + _esc(rel))
    if fact.invalidations:
        parts.append("!Ø(" + ",".join(_esc(x) for x in fact.invalidations) + ")")
    return "".join(parts)


def decode(s: str) -> MemoryFact:
    n = len(s)
    if not s.startswith("["):
        raise ValueError("expected '['")
    domain, i = _scan_leaf(s, 1, {"]"})
    if i >= n or s[i] != "]":
        raise ValueError("expected ']'")
    i += 1
    if i >= n or s[i] != "«":
        raise ValueError("expected '«'")
    etype, i = _scan_leaf(s, i + 1, {"»"})
    if i >= n or s[i] != "»":
        raise ValueError("expected '»'")
    i += 1
    if i >= n or s[i] != "@":
        raise ValueError("expected '@'")
    actor, i = _scan_leaf(s, i + 1, {"{", "(", ">", "!"})

    entity: str | None = None
    if i < n and s[i] == "{":
        entity, i = _scan_leaf(s, i + 1, {"}"})
        if i >= n or s[i] != "}":
            raise ValueError("expected '}'")
        i += 1

    attrs: dict[str, Any] = {}
    if i < n and s[i] == "(":
        i += 1
        if i < n and s[i] == ")":
            i += 1  # empty () -- shouldn't happen (omitted when empty) but tolerate
        else:
            while True:
                key, i = _scan_leaf(s, i, {":"})
                if i >= n or s[i] != ":":
                    raise ValueError("expected ':' after key")
                i += 1
                if i >= n:
                    raise ValueError("expected value code")
                code = s[i]
                i += 1
                text, i = _scan_leaf(s, i, {"+", ")"})
                attrs[key] = _dec_value(code, text)
                if i >= n:
                    raise ValueError("unterminated attrs")
                if s[i] == "+":
                    i += 1
                    continue
                if s[i] == ")":
                    i += 1
                    break
                raise ValueError("expected '+' or ')'")

    relations: list[str] = []
    while i < n and s[i] == ">":
        if i + 1 >= n or s[i + 1] != ">":
            raise ValueError("expected '>>'")
        rel, i = _scan_leaf(s, i + 2, {">", "!"})
        relations.append(rel)

    invalidations: list[str] = []
    if i < n and s[i] == "!":
        if s[i : i + 2] != "!Ø":
            raise ValueError("expected '!Ø'")
        i += 2
        if i >= n or s[i] != "(":
            raise ValueError("expected '(' after !Ø")
        i += 1
        # `!Ø(...)` is only emitted for a non-empty list, so always parse at
        # least one element: `!Ø()` round-trips to [''] (empty list is omitted).
        while True:
            inv, i = _scan_leaf(s, i, {",", ")"})
            invalidations.append(inv)
            if i >= n:
                raise ValueError("unterminated !Ø")
            if s[i] == ",":
                i += 1
                continue
            if s[i] == ")":
                i += 1
                break
    if i != n:
        raise ValueError(f"trailing input at {i}: {s[i:]!r}")
    return MemoryFact(
        domain=domain,
        etype=etype,
        actor=actor,
        attrs=attrs,
        entity=entity,
        relations=relations,
        invalidations=invalidations,
    )


# ---- event <-> fact projection -------------------------------------------

# Declarative content fields. The integrity envelope below is excluded by design.
ENVELOPE_FIELDS = {
    "seq", "timestamp", "prev_hash", "hash", "id", "parent_event_id",
    "caused_by", "envelope_version", "security",
}


def event_to_fact(event: dict[str, Any]) -> MemoryFact:
    """Project the declarative content of a raw Zaxy event into a MemoryFact.

    Lossless over: type, actor, thread, payload (full, order-preserved).
    """
    return MemoryFact(
        domain=event.get("thread", ""),
        etype=event.get("type", ""),
        actor=event.get("actor", ""),
        attrs=dict(event.get("payload", {})),
    )


def fact_to_content(fact: MemoryFact) -> dict[str, Any]:
    """Reconstruct the declarative-content view of an event from a fact."""
    return {
        "type": fact.etype,
        "actor": fact.actor,
        "thread": fact.domain,
        "payload": fact.attrs,
    }


def event_content(event: dict[str, Any]) -> dict[str, Any]:
    """The declarative-content view of a raw event, for comparison."""
    return {
        "type": event.get("type", ""),
        "actor": event.get("actor", ""),
        "thread": event.get("thread", ""),
        "payload": event.get("payload", {}),
    }
