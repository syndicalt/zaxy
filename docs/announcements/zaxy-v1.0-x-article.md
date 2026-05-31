# X Article Draft: Zaxy 1.0.0

Zaxy 1.0.0 is out: Coordinator Memory for Agent Teams.

The short version: Zaxy is memory infrastructure for agent work that needs
provenance, replay, and coordination. It is built around Eventloom append-only
logs, graph-backed retrieval, Memory Bootstrap, Memory Checkout, and Zaxy
Coordinate workflows.

Most agent memory systems flatten context into notes or chunks. That is useful
until multiple agents are exploring in parallel and nobody can tell which
worker-local claim became project truth.

Zaxy 1.0.0 focuses on that gap.

What ships:

- local-first Eventloom memory logs with hash-chain provenance;
- embedded Kuzu graph projection as the default no-sidecar runtime;
- MCP tools for memory append, query, replay, feedback, bootstrap, checkout,
  and coordination;
- Memory Checkout with cited current state, diagnostics, answerability, and
  feedback guidance;
- Zaxy Coordinate missions with isolated workers, cited findings, review,
  approval packets, promoted state, and handoff;
- docs, release gates, benchmark guardrails, package artifacts, and clean-repo
  UAT wired into the release check.

The release gate passed end to end locally:

- 1972 tests passed, 1 skipped, 11 deselected;
- 92.04% coverage against a 92.00% ratchet;
- package build and twine check passed;
- docs validation passed;
- MCP smoke, public examples, LangGraph smoke, Coordinate mission smoke, and
  benchmark guardrails passed;
- clean-repo beta UAT passed across the local onboarding paths.

External verification request:

If you did not work on this implementation session, please try the first-run
path or the Coordinate example and report what happens. Successes are useful,
but friction is more useful.

First-run path:

```bash
zaxy init
zaxy memory bootstrap --eventloom-path .eventloom
zaxy memory checkout "current project memory and next useful action" --eventloom-path .eventloom
zaxy doctor --beta-readiness
```

Coordinate path:

```bash
python examples/coordinate_three_worker_project.py
```

Please file an "External validation" issue with OS, shell, Python version,
install source, exact commands, time to first useful checkout or handoff, what
worked, what failed, and whether the release should pass, pass with follow-up,
or fail until fixed.

External validation is optional post-release evidence for 1.0.0. We are not
pretending it already happened. We are asking for it publicly.

Links:

- Release announcement: `docs/announcements/zaxy-v1.0.md`
- External validation guide: `docs/external-validation.md`
- Getting started: `docs/getting-started.md`
- Coordinate quickstart: `docs/coordinate-quickstart.md`
- GitHub issue template: `.github/ISSUE_TEMPLATE/external_validation.md`
