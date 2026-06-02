# CoordinationBench Adapter Contract Kit

This kit defines the `coordinationbench-v1` external adapter contract.

Adapter authors should:

1. Use `coordination-workload.json` as the frozen workload.
2. Copy the relevant runner manifest template.
3. Replace every placeholder with a real pinned adapter version, source ref,
   install command, dataset contract, result schema, and argv-only runner
   command.
4. Produce a result file that matches `schemas/result.schema.json`.
5. Validate locally before publishing:

```bash
zaxy coordinate benchmark-adapter validate-manifest mem0=templates/mem0.runner-manifest.json --workload coordination-workload.json
zaxy coordinate benchmark-adapter validate-result mem0=templates/mem0-result.json --workload coordination-workload.json
zaxy coordinate benchmark --output-dir report --competitor-runner mem0=templates/mem0.runner-manifest.json
```

Packaged disclosure templates currently cover `mem0`, `agent_memory`,
`activegraph`, `quarq`, and `hybi`. Each adapter name is valid only when the
manifest and result file match the frozen workload fingerprint.

The packaged `quarq` and `hybi` templates include pinned public metadata and an
explicit unsupported runner command. They are not same-harness results. Remove
`template: true` only after replacing the unsupported runner with a real adapter
that replays the workload and writes `schemas/result.schema.json`.

Template manifests are not executable claims. Zaxy refuses to execute manifests
with `template: true` or placeholder `run_command` values.

Result cases may also report optional synthesis proof fields such as
`answer_candidate`, `synthesis_artifact`, `ledger_rows`, `support_source_ids`,
`excluded_source_ids`, `promotion_source_ids`, `answerability`, and
`non_authoritative_rows_injected`. Use these fields to disclose whether a
system produced a cited accepted-state answer, excluded duplicate or stale rows,
and avoided leaking pending worker-local evidence into authoritative output.
`accepted_state_synthesis_quality` is scored from these proof-backed synthesis
fields; `returned_text` alone is not sufficient for synthesis-quality credit.
