"""Quality-controlled token-reduction methodology: structured cited memory vs chunk-RAG.

Dev target #2: the AGENTS.md Performance-Targets table states "Token reduction vs
chunk RAG: 70-90% (structured paths vs raw text)". That is an *unvalidated target*.
This module is the reproducible methodology to validate (or correct) it.

The trap this avoids: a raw token ratio is meaningless, because you can always
"save tokens" by retrieving less and answering worse. So every comparison is
**quality-controlled** — measured at EQUAL ANSWER-BEARING RECALL: both arms must
surface the gold answer span, and we count the tokens each needs to get there.

  reduction = 1 - structured_tokens / chunkrag_tokens   (over cases where BOTH reach gold)

The chunk-RAG baseline is pinned and reported (chunk size, ranking) so the ratio
is not gameable by quietly weakening the baseline.

Run `python3 scripts/chunk_rag_token_compare.py` for a self-test on a worked
example. To produce a *publishable* number, drive `compare_corpus()` with a real
QA dataset (e.g. LongMemEval) through the gated benchmark apparatus; the headline
number depends on workload and must be reported with both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
DEFAULT_CHUNK_TOKENS = 512


def tokens(text: str) -> int:
    return len(ENC.encode(text))


def chunk_text(text: str, chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> list[str]:
    toks = ENC.encode(text)
    return [ENC.decode(toks[i : i + chunk_tokens]) for i in range(0, len(toks), chunk_tokens)]


def _kw(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _score(passage: str, query: str) -> int:
    q = set(_kw(query))
    return sum(1 for w in _kw(passage) if w in q)


def _tokens_to_gold(passages: list[str], query: str, gold: str) -> tuple[int, bool]:
    """Greedily accumulate the highest-scoring passages until gold appears.

    Returns (tokens_used, reached_gold). Pure keyword ranking — a standard,
    embedding-free chunk-RAG baseline (note: vector-RAG could differ).
    """
    ranked = sorted(passages, key=lambda p: -_score(p, query))
    used: list[str] = []
    total = 0
    gold_l = gold.lower()
    for p in ranked:
        used.append(p)
        total += tokens(p)
        if gold_l in " ".join(used).lower():
            return total, True
    return total, False


@dataclass
class CaseResult:
    query: str
    structured_tokens: int
    chunkrag_tokens: int
    both_reached: bool

    @property
    def reduction(self) -> float | None:
        if not self.both_reached or self.chunkrag_tokens == 0:
            return None
        return 1 - self.structured_tokens / self.chunkrag_tokens


def compare_case(
    *, query: str, gold: str, facts: list[str], raw_text: str, chunk_tokens: int = DEFAULT_CHUNK_TOKENS
) -> CaseResult:
    """One query: structured-cited retrieval vs chunk-RAG over raw_text, at equal recall."""
    s_tok, s_ok = _tokens_to_gold(facts, query, gold)
    c_tok, c_ok = _tokens_to_gold(chunk_text(raw_text, chunk_tokens), query, gold)
    return CaseResult(query, s_tok, c_tok, s_ok and c_ok)


def compare_corpus(cases: list[dict], *, chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> dict:
    """Run a corpus of {query, gold, facts, raw_text} cases. Returns aggregate stats.

    Only cases where BOTH arms reach gold count toward the reduction (equal recall).
    """
    results = [compare_case(chunk_tokens=chunk_tokens, **c) for c in cases]
    scored = [r for r in results if r.reduction is not None]
    reductions = [r.reduction for r in scored]
    return {
        "baseline": {"chunk_tokens": chunk_tokens, "ranking": "keyword-overlap (embedding-free)"},
        "cases_total": len(results),
        "cases_equal_recall": len(scored),
        "median_reduction": (sorted(reductions)[len(reductions) // 2] if reductions else None),
        "mean_reduction": (sum(reductions) / len(reductions) if reductions else None),
        "per_case": [
            {"query": r.query, "structured_tok": r.structured_tokens,
             "chunkrag_tok": r.chunkrag_tokens, "reduction": r.reduction}
            for r in results
        ],
    }


def _self_test_corpus() -> list[dict]:
    """A worked example: a fact buried in verbose tool/dialogue text vs a compact card.

    Illustrative only — the magnitude depends on raw verbosity, which is exactly why
    a real run needs a QA dataset + the gated apparatus.
    """
    noise = ("INFO build step ok; cache warm; 0 warnings; retrying; " * 60)
    raw = (
        noise
        + " The team decided the canonical session id is zaxy-default and froze the legacy zaxy session. "
        + noise
        + " Unrelated: the default graph backend for beta is Neo4j. "
        + noise
    )
    facts = [
        "[decision] canonical session id = zaxy-default; legacy 'zaxy' frozen. (eventloom://zaxy-default/events/77848)",
        "[decision] default graph backend (beta) = Neo4j. (eventloom://zaxy-default/events/1200)",
        "[goal] ship per-turn memory injection. (eventloom://zaxy-default/events/77889)",
    ]
    return [
        {"query": "which session id is canonical", "gold": "zaxy-default", "facts": facts, "raw_text": raw},
        {"query": "default graph backend for beta", "gold": "Neo4j", "facts": facts, "raw_text": raw},
    ]


if __name__ == "__main__":
    import json

    out = compare_corpus(_self_test_corpus())
    print(json.dumps(out, indent=2))
    print("\nNOTE: self-test is illustrative. A publishable 'X% vs chunk-RAG' number "
          "requires driving compare_corpus() with a real QA dataset (gold-labeled) "
          "through the freeze/gate/audit apparatus, reported with workload + baseline.")
