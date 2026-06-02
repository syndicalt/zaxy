# Public-Derived Purpose Holdout Pack

This pack is a diagnostic holdout fixture for purpose-conditioned memory. It is
not an internal release gate, external validation report, or same-harness
competitor result.

The canonical fingerprint is computed over `holdout-pack.json` with the
`fingerprint` field removed. Cases are frozen and must not be tuned on expected
answers. Report these holdouts separately from deterministic `purpose-v1`
lanes.
