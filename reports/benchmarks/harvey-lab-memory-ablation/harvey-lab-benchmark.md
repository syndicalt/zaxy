# Harvey LAB External Memory Benchmark

- Generated: `2026-06-06T19:16:09.593828+00:00`
- Status: `complete`
- External suite: [Harvey LAB memory retrieval ablation](https://github.com/rushilchugh01/harvey-labs-ablations-and-benchmarks)
- Article: [https://rushilchugh.substack.com/p/what-agent-memory-actually-fixes](https://rushilchugh.substack.com/p/what-agent-memory-actually-fixes)
- Zaxy tasks scored: `10/10`

## Result Provenance

- Source: `harvey-lab-import`
- Harvey git commit: `29748828133dff83ad2263af353fb035504f8f77`
- Provenance roots: `2` (exact paths are stored in the JSON report)
- Normalized result artifacts: `10`
- External baseline reports: `comparison-gpt54mini-flashlite.json, comparison-zaxy.json, comparison.json`
- External run manifests: `harvey-lab-external-run.json`
- External readiness reports: `harvey-lab-ready.json`
- External status reports: `harvey-lab-status.json`

## Summary

| System | Tasks | Mean criterion pass rate | Delta vs regular | Delta vs article best | Wins vs article best |
|--------|------:|--------------------------:|-----------------:|----------------------:|---------------------:|
| Zaxy | 10 | 0.788 | +0.184 | +0.081 | 9 |

## Zaxy Runtime And Usage

| Mean total seconds | Total tokens | Memory search calls | Memory read calls |
|-------------------:|-------------:|--------------------:|------------------:|
| 138.786 | 5951174 | 30 | 10 |

## Task Comparison

| Task | Shape | Regular | Article best | Zaxy | Delta vs regular | Delta vs best | Winner |
|------|-------|--------:|--------------|------:|-----------------:|--------------:|--------|
| FTC noncompete | Compact legal-risk synthesis | 0.807 | Graphiti 0.790 | 0.895 | +0.088 | +0.105 | Zaxy |
| Change-of-control | Sparse clause hunt | 0.667 | GBrain keyword 0.737 | 0.860 | +0.193 | +0.123 | Zaxy |
| Acquisition diligence | Broad diligence sweep | 0.469 | raw-rg 0.641 | 0.797 | +0.328 | +0.156 | Zaxy |
| Data-room red flags | Red-flag spotting | 0.520 | LightRAG 0.600 | 0.660 | +0.140 | +0.060 | Zaxy |
| Privacy program | Compliance mapping | 0.532 | ActiveGraph 0.661 | 0.790 | +0.258 | +0.129 | Zaxy |
| Litigation timeline | Event reconstruction | 0.652 | GBrain keyword 0.758 | 0.894 | +0.242 | +0.136 | Zaxy |
| Relevance / privilege | Document-by-document coding | 0.701 | GBrain keyword 0.791 | 0.687 | -0.014 | -0.104 | GBrain keyword |
| Attorney production review | Production-set classification | 0.583 | GBrain Gemma / LightRAG 0.708 | 0.750 | +0.167 | +0.042 | Zaxy |
| Privilege log | Large log-heavy classification | 0.402 | GBrain keyword 0.598 | 0.634 | +0.232 | +0.036 | Zaxy |
| Subpoena comparison | Request matching | 0.702 | raw-rg 0.790 | 0.912 | +0.210 | +0.122 | Zaxy |

## Framework Scorecard

| Framework | Evidence scope | Tasks | Mean score | Delta vs regular | Zaxy overlap | Zaxy delta on overlap |
|-----------|----------------|------:|-----------:|-----------------:|-------------:|----------------------:|
| regular no-memory | article regular baseline across all ten tasks | 10 | 0.604 | +0.000 | 10 | +0.184 |
| article best observed | best published memory/search row per article task | 10 | 0.707 | +0.104 | 10 | +0.081 |
| raw-rg | article task-winner matrix only | 2 | 0.716 | +0.130 | 2 | +0.139 |
| GBrain keyword | article task-winner matrix only | 4 | 0.721 | +0.116 | 4 | +0.048 |
| GBrain Gemma | article task-winner matrix only | 1 | 0.708 | +0.125 | 1 | +0.042 |
| LightRAG | article task-winner matrix only | 2 | 0.654 | +0.103 | 2 | +0.051 |
| ActiveGraph | article task-winner matrix only | 1 | 0.661 | +0.129 | 1 | +0.129 |
| Graphiti | article task-winner matrix only | 1 | 0.790 | -0.017 | 1 | +0.105 |
| Mem0 | framework fit only; no published task-winning score | 0 |  |  | 0 |  |
| Zaxy | same-harness external Zaxy normalized results | 10 | 0.788 | +0.184 | 10 |  |

## External Baseline Aggregate

| Framework | Evidence scope | Runs | Mean score | Delta vs source raw-rg | Mean seconds | Source report |
|-----------|----------------|-----:|-----------:|----------------:|-------------:|---------------|
| activegraph | Harvey-native comparison artifact | 1 | 0.544 | +0.158 | 108.140 | comparison.json |
| mem0-keyword | Harvey-native comparison artifact | 1 | 0.544 | +0.158 | 107.740 | comparison.json |
| cognee | Harvey-native comparison artifact | 1 | 0.509 | +0.123 | 86.130 | comparison.json |
| mem0 | Harvey-native comparison artifact | 1 | 0.439 | +0.053 | 114.950 | comparison.json |
| llm-wiki | Harvey-native comparison artifact | 1 | 0.404 | +0.018 | 99.800 | comparison.json |
| gbrain-keyword | Harvey-native comparison artifact | 1 | 0.386 | +0.000 | 83.410 | comparison.json |
| raw-rg | Harvey-native comparison artifact | 2 | 0.380 | +0.000 | 149.590 | comparison-gpt54mini-flashlite.json |
| lightrag | Harvey-native comparison artifact | 1 | 0.368 | -0.018 | 157.800 | comparison.json |
| lightrag-keyword | Harvey-native comparison artifact | 1 | 0.333 | -0.053 | 83.380 | comparison.json |
| gbrain-gemma | Harvey-native comparison artifact | 1 | 0.281 | -0.105 | 115.780 | comparison.json |
| graphiti | Harvey-native comparison artifact | 1 | 0.123 | -0.263 | 111.600 | comparison.json |

## Zaxy vs External Scored Systems

| Framework | Evidence scope | Runs | Mean score | Delta vs source raw-rg | Delta vs best external | Mean seconds | Rank |
|-----------|----------------|-----:|-----------:|-----------------------:|-----------------------:|-------------:|-----:|
| Zaxy | same-harness external Zaxy normalized results | 10 | 0.788 | +0.408 | +0.244 | 138.786 | 1 |
| activegraph | Harvey-native comparison artifact | 1 | 0.544 | +0.158 | +0.000 | 108.140 | 2 |
| mem0-keyword | Harvey-native comparison artifact | 1 | 0.544 | +0.158 | +0.000 | 107.740 | 3 |
| cognee | Harvey-native comparison artifact | 1 | 0.509 | +0.123 | -0.035 | 86.130 | 4 |
| mem0 | Harvey-native comparison artifact | 1 | 0.439 | +0.053 | -0.105 | 114.950 | 5 |
| llm-wiki | Harvey-native comparison artifact | 1 | 0.404 | +0.018 | -0.140 | 99.800 | 6 |
| gbrain-keyword | Harvey-native comparison artifact | 1 | 0.386 | +0.000 | -0.158 | 83.410 | 7 |
| raw-rg | Harvey-native comparison artifact | 2 | 0.380 | +0.000 | -0.164 | 149.590 | 8 |
| lightrag | Harvey-native comparison artifact | 1 | 0.368 | -0.018 | -0.176 | 157.800 | 9 |
| lightrag-keyword | Harvey-native comparison artifact | 1 | 0.333 | -0.053 | -0.211 | 83.380 | 10 |
| gbrain-gemma | Harvey-native comparison artifact | 1 | 0.281 | -0.105 | -0.263 | 115.780 | 11 |
| graphiti | Harvey-native comparison artifact | 1 | 0.123 | -0.263 | -0.421 | 111.600 | 12 |

## Framework Fit

| Framework | Where strongest | Example tasks | Interpretation |
|-----------|-----------------|---------------|----------------|
| raw-rg | Literal evidence finding | Acquisition diligence; subpoena comparison | raw-rg is a retrieval/search baseline, not a no-memory baseline; it was strongest when direct lexical matches were enough. |
| GBrain keyword | Clause hunts and classification-heavy tasks | Change-of-control; litigation timeline; relevance / privilege; privilege log | Keyword retrieval preserved high-precision hooks from task prompts and rubrics. |
| GBrain Gemma | Production-review style classification | Attorney production review | The query layer appeared to bring useful classification cues into final review. |
| LightRAG | Red-flag spotting after full document coverage | Data-room red flags; attorney production review | Graph/vector retrieval looked useful when selection and focus mattered. |
| ActiveGraph | Compliance/state mapping | Privacy program | Structured state helped organize controls, obligations, and gaps. |
| Graphiti | Compact legal synthesis | FTC noncompete | Episode/graph memory may organize actors and relationships, but regular already scored higher. |
| Mem0 | Native stored source chunk memory | Article comparison set | Included as a native memory-search system in the article, but not a task winner in the published matrix. |
| Zaxy | Compact legal-risk synthesis; Sparse clause hunt; Broad diligence sweep; Red-flag spotting; Compliance mapping; Event reconstruction; Production-set classification; Large log-heavy classification; Request matching | FTC noncompete; Change-of-control; Acquisition diligence; Data-room red flags; Privacy program; Litigation timeline; Attorney production review; Privilege log; Subpoena comparison | Zaxy is ahead of the published article-best rows on the scored subset. |

## Caveats

- Scores are criterion pass rates from Harvey LAB judging, not binary all-pass task success.
- Article comparison rows are published external disclosures, not Zaxy same-process reruns.
- Framework scorecard rows for article systems use published task-winner coverage, not full hidden per-framework averages.
- raw-rg is a retrieval/search baseline, not the no-memory baseline.
- Zaxy rows must come from Harvey normalized result artifacts; internal LongMemEval results are rejected.
