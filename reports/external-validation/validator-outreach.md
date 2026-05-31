# Zaxy v1.0 Validator Outreach Tracker

Use this file to track the 1-3 external validators requested after the v1.0
release. Do not mark a validator complete until their evidence is posted to a
reviewable issue or equivalent artifact. Convert successful evidence into the
machine-checkable JSON report only after it is real and reviewable.

Reviewable validation request:

```text
https://github.com/syndicalt/zaxy/issues/17
```

## Candidate Slots

### Validator 1

- Candidate:
- Relationship to project:
- External to implementation session: yes/no
- Public material sent:
  - README or docs site
  - `docs/external-validation.md`
  - install instructions
- Selected validation path:
- Evidence URL:
- Release decision:
- Status: not contacted

### Validator 2

- Candidate:
- Relationship to project:
- External to implementation session: yes/no
- Public material sent:
  - README or docs site
  - `docs/external-validation.md`
  - install instructions
- Selected validation path:
- Evidence URL:
- Release decision:
- Status: not contacted

### Validator 3

- Candidate:
- Relationship to project:
- External to implementation session: yes/no
- Public material sent:
  - README or docs site
  - `docs/external-validation.md`
  - install instructions
- Selected validation path:
- Evidence URL:
- Release decision:
- Status: not contacted

## Completion Rule

The default v1.0 release gate does not require external validation. This
outreach is optional post-release evidence. A strict local policy can require
external validation after at least one external validator posts reviewable
evidence with a `pass` or `pass_with_follow_up` decision, the evidence is copied
into
`reports/external-validation/external-validation-report.json`, and both local
commands pass:

```bash
python scripts/check-external-validation.py reports/external-validation/external-validation-report.json
zaxy doctor --beta-readiness --require-external-validation --external-validation-report reports/external-validation/external-validation-report.json
```
