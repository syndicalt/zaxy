# Zaxy v1 Release Docs and Media Design

## Goal

Make the public documentation, static site, release article, and launch media
consistent with a Zaxy 1.0.0 stable release while publicly asking for external
verification after release.

## Scope

- Refresh release-facing docs and site copy from release-candidate/beta posture
  to stable 1.0.0 posture.
- Preserve the truth that external validation is optional for this release, but
  make the public request for outside verification visible and actionable.
- Add a polished header image for the 1.0.0 announcement.
- Add a short scripted demo asset for Zaxy Coordinate/Collaborate that can be
  generated locally from static frames and `ffmpeg`.
- Add an X-ready release article/thread draft with exact copy and links to the
  verification request.

## Non-Goals

- No live account screenshare or hosted social posting.
- No new product behavior.
- No benchmark claim changes unless correcting stale public proof text.
- No external validation report fabrication.

## Design

The release docs will use "Zaxy 1.0.0" as the stable launch label. Historical
0.4.0 docs remain available as archive material, but the README, homepage, and
v1.0 announcement should lead with the stable release.

The demo media will be a scripted product walkthrough, not a terminal recording.
It will show the coordination flow in five frames: launch, mission creation,
worker findings, coordinator review, and Memory Checkout. This keeps the asset
credible and reproducible without depending on a browser recorder or external
accounts.

The X article will be saved as documentation so it is reviewable and versioned.
It will include a public call for external verification: run the first-run or
Coordinate path, file the external validation issue, and share both successes
and friction.

## Validation

- Regenerate rendered docs with `python scripts/build-site-docs.py`.
- Run `python scripts/build-site-docs.py --check`.
- Run `scripts/validate-docs.sh --root .`.
- Run focused documentation/package tests affected by version and site text.
