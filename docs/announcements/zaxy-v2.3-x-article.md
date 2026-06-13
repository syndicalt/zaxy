# X Article Draft: Zaxy 2.3.0

![Zaxy 2.3 release header](../assets/zaxy-v2.3-header.png)

Zaxy 2.3.0 is out: the embedded engine moves off archived infrastructure, and
nothing you rely on moves with it.

The short version: Zaxy's default local graph runtime was built on Kuzu, which
shipped its final release and archived its repository — no more fixes, no more
wheels for new Python versions. 2.3 moves the embedded backend to LadybugDB,
the maintained fork of the same engine lineage, and keeps the
`PROJECTION_BACKEND=embedded` contract, the schema, query semantics, and the
ANN engagement defaults exactly where 2.2 left them.

Depending on a dead dependency is a slow-motion outage. But swapping the engine
under a memory system is also exactly how you lose someone's data if you are
careless about it. So the rule for this release was: the source of truth never
moves, and the migration deletes nothing.

How the move works:

- Eventloom — the append-only, hash-chained log — is untouched. It was always
  the source of truth; projections were always rebuildable from it. That design
  is what makes an engine swap a projection-layer change instead of a data
  migration.
- On first open of a pre-fork projection file LadybugDB cannot read, Zaxy moves
  it (and its WAL) aside to `<path>.pre-ladybug.bak` — **no data deleted** —
  opens a fresh store, and rebuilds from the log via `zaxy reproject`. `zaxy
  doctor` reports the leftover backup until you have verified the rebuild and
  chosen to delete it.
- The move also collects a dividend: LadybugDB fixed the upstream
  `DROP_VECTOR_INDEX` corruption defect 2.2 had to work around, so superseded
  ANN index generations are now dropped for full space reclaim.

Now the part that is the reason to trust the rest of it.

The release branch arrived with a confident comment in the code: the vector
index extension was "statically linked, offline-safe, no INSTALL needed." It
was wrong. LadybugDB ships the vector index as a downloadable extension, not a
bundled one. It happened to work on the machine that built the release —
because that machine had the extension cached from earlier use — and it would
have silently failed on every fresh install, disabling approximate search for
new users while swallowing the error.

Nothing in a human read of the diff would have caught that. What caught it was
the coverage ratchet: on a clean CI runner the extension was absent, the native
vector tests skipped, and total coverage slipped 0.02% under the floor. A
two-hundredths-of-a-percent tripwire is the only thing standing between "looks
done" and a marquee feature that is dead on arrival. We traced the skip to the
missing extension, reproduced the fresh-machine condition locally, and fixed
the cause: the store now installs-then-loads the extension, cached after first
use, and degrades to exact search with a clear warning when it genuinely
cannot.

A note on framing, because we got it wrong first and corrected it: this is
*local-first*, not offline-only. A one-time extension fetch that then runs
entirely on-box is no more a network dependency than `pip install` itself — the
data, the engine, and the compute all stay local. The default exact-search path
is pure NumPy and fetches nothing. Air-gapped deployments that want approximate
search pre-seed the extension cache; everyone else never notices.

The claim boundary, as always: the engine swap was gated on a readiness-research
pass and lane reruns that reproduced 2.2's deterministic evidence — identical
corpus hashes and recall — with raw artifacts versioned in the repo. The fork is
single-maintainer and the dependency pin is exact on purpose; that is what
depending on infrastructure honestly looks like, archived or forked.

Release: https://github.com/syndicalt/zaxy/releases/tag/v2.3.0
Migration: https://docs.zaxy.io/docs/migration.html
Install: `pip install zaxy-memory`

Zaxy is event-sourced memory for agent work: a hash-chained append-only log as
the source of truth, cited Memory Checkout as the trust contract, and a runtime
where the engine is replaceable because the log is authoritative.
https://docs.zaxy.io
