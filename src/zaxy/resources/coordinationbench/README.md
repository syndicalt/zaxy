# CoordinationBench Adapter Contract Kit

This kit defines the `coordinationbench-v1` external adapter contract.

Adapter authors should:

1. Use `coordination-workload.json` as the frozen workload.
2. Copy the relevant runner manifest template.
3. Replace every placeholder with a real pinned adapter version, source ref,
   install command, and argv-only runner command.
4. Produce a result file that matches `schemas/result.schema.json`.
5. Validate locally before publishing:

```bash
zaxy coordinate benchmark-adapter validate-manifest mem0=templates/mem0.runner-manifest.json --workload coordination-workload.json
zaxy coordinate benchmark-adapter validate-result mem0=templates/mem0-result.json --workload coordination-workload.json
zaxy coordinate benchmark --output-dir report --competitor-runner mem0=templates/mem0.runner-manifest.json
```

Template manifests are not executable claims. Zaxy refuses to execute manifests
with `template: true` or placeholder `run_command` values.
