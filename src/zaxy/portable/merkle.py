"""Binary Merkle tree with domain separation, for verifiable subset disclosure.

Leaves and internal nodes use distinct hash prefixes (second-preimage resistance:
an internal node digest can never be mistaken for a leaf). Unpaired nodes are
promoted ("carried up") rather than duplicated, avoiding the duplicate-leaf
ambiguity class.
"""

from __future__ import annotations

import hashlib

_LEAF = b"\x00"
_NODE = b"\x01"


def _sha(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def leaf_hash(content: bytes) -> bytes:
    return _sha(_LEAF, content)


def _node_hash(left: bytes, right: bytes) -> bytes:
    return _sha(_NODE, left, right)


def merkle_root(leaves: list[bytes]) -> bytes:
    if not leaves:
        raise ValueError("merkle_root requires at least one leaf")
    layer = list(leaves)
    while len(layer) > 1:
        nxt: list[bytes] = []
        for i in range(0, len(layer), 2):
            if i + 1 < len(layer):
                nxt.append(_node_hash(layer[i], layer[i + 1]))
            else:
                nxt.append(layer[i])  # promote unpaired node
        layer = nxt
    return layer[0]


def inclusion_proof(leaves: list[bytes], index: int) -> list[tuple[bytes, str]]:
    """Return [(sibling_hash, side)] from leaf up to root; side is the SIBLING side."""
    if not 0 <= index < len(leaves):
        raise IndexError("index out of range")
    proof: list[tuple[bytes, str]] = []
    idx = index
    layer = list(leaves)
    while len(layer) > 1:
        nxt: list[bytes] = []
        for i in range(0, len(layer), 2):
            if i + 1 < len(layer):
                left, right = layer[i], layer[i + 1]
                if i == idx - (idx % 2):  # the pair containing idx
                    proof.append((right, "R") if idx % 2 == 0 else (left, "L"))
                nxt.append(_node_hash(left, right))
            else:
                nxt.append(layer[i])  # idx (if this lone node) carries up with no sibling
        idx //= 2
        layer = nxt
    return proof


def verify_inclusion(leaf: bytes, proof: list[tuple[bytes, str]], root: bytes) -> bool:
    node = leaf
    for sibling, side in proof:
        node = _node_hash(sibling, node) if side == "L" else _node_hash(node, sibling)
    return node == root
