"""Built-in Coordinate mission templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zaxy.coordination import CoordinationManager


@dataclass(frozen=True)
class WorkerTemplate:
    """Worker role and assignment included in a mission template."""

    worker_id: str
    role: str
    assignment: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "role": self.role,
            "assignment": self.assignment,
        }


@dataclass(frozen=True)
class MissionTemplate:
    """Reusable Coordinate mission shape for a common workflow."""

    name: str
    title: str
    objective: str
    description: str
    workers: tuple[WorkerTemplate, ...]
    acceptance_criteria: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "objective": self.objective,
            "description": self.description,
            "workers": [worker.to_dict() for worker in self.workers],
            "acceptance_criteria": list(self.acceptance_criteria),
        }


MISSION_TEMPLATES: tuple[MissionTemplate, ...] = (
    MissionTemplate(
        name="software-delivery",
        title="Software Delivery",
        objective="Ship a production change with implementation, verification, and review evidence.",
        description="Coordinate a code change from implementation through independent verification.",
        workers=(
            WorkerTemplate(
                worker_id="implementation",
                role="Implementation",
                assignment="Implement the scoped change and report code, migration, and risk evidence.",
            ),
            WorkerTemplate(
                worker_id="verification",
                role="Verification",
                assignment="Run focused tests and relevant gates, then report exact commands and outcomes.",
            ),
            WorkerTemplate(
                worker_id="review",
                role="Review",
                assignment="Inspect the diff for correctness, regressions, missing tests, and release risk.",
            ),
        ),
        acceptance_criteria=(
            "Implementation evidence cites changed files or commits.",
            "Verification evidence cites commands and results.",
            "Review findings are accepted or explicitly deferred before handoff.",
        ),
    ),
    MissionTemplate(
        name="research-review",
        title="Research Review",
        objective="Review a research question with source-grounded claims and explicit uncertainty.",
        description="Split source collection, synthesis, and critique into auditable worker lanes.",
        workers=(
            WorkerTemplate(
                worker_id="source-survey",
                role="Source Survey",
                assignment="Collect primary sources and report citation-backed facts with dates.",
            ),
            WorkerTemplate(
                worker_id="synthesis",
                role="Synthesis",
                assignment="Compare source claims, identify consensus, and summarize open questions.",
            ),
            WorkerTemplate(
                worker_id="critique",
                role="Critique",
                assignment="Check for unsupported claims, stale evidence, and methodological limits.",
            ),
        ),
        acceptance_criteria=(
            "Accepted claims cite source references.",
            "Known conflicts or stale claims are reviewed before promotion.",
            "The handoff includes limitations and unresolved questions.",
        ),
    ),
    MissionTemplate(
        name="benchmark-investigation",
        title="Benchmark Investigation",
        objective="Investigate a benchmark result with reproducible inputs, baselines, and limitations.",
        description="Separate workload verification, result analysis, and disclosure review.",
        workers=(
            WorkerTemplate(
                worker_id="workload-audit",
                role="Workload Audit",
                assignment="Verify tracked benchmark inputs, fingerprints, commands, and environment.",
            ),
            WorkerTemplate(
                worker_id="result-analysis",
                role="Result Analysis",
                assignment="Analyze scores, latency, regressions, and baseline comparisons.",
            ),
            WorkerTemplate(
                worker_id="disclosure-review",
                role="Disclosure Review",
                assignment="Review claims for reproducibility, caveats, and competitor-comparison limits.",
            ),
        ),
        acceptance_criteria=(
            "Benchmark inputs and commands are reproducible from tracked artifacts.",
            "Baselines and regression thresholds are reported.",
            "Limitations are included with accepted benchmark claims.",
        ),
    ),
    MissionTemplate(
        name="release-validation",
        title="Release Validation",
        objective="Validate a release candidate with gates, packaging, runtime smoke, and risk review.",
        description="Coordinate final release checks before tagging or publishing a candidate.",
        workers=(
            WorkerTemplate(
                worker_id="release-gates",
                role="Release Gates",
                assignment="Run and record required tests, lint, type checks, docs checks, and release smoke.",
            ),
            WorkerTemplate(
                worker_id="docs-packaging",
                role="Docs and Packaging",
                assignment="Validate docs, changelog, package metadata, and distribution artifacts.",
            ),
            WorkerTemplate(
                worker_id="runtime-smoke",
                role="Runtime Smoke",
                assignment="Exercise install, init, MCP, native adapter, and Coordinate smoke paths.",
            ),
            WorkerTemplate(
                worker_id="risk-audit",
                role="Risk Audit",
                assignment="Review release blockers, known limitations, migrations, and rollback posture.",
            ),
        ),
        acceptance_criteria=(
            "Release gates pass or have explicit reviewed exceptions.",
            "Docs and packaging artifacts match the release scope.",
            "Runtime smoke covers the public adoption paths.",
            "Known risks are captured in the handoff before release.",
        ),
    ),
)


def list_mission_templates() -> list[MissionTemplate]:
    """Return built-in mission templates in stable CLI display order."""
    return list(MISSION_TEMPLATES)


def get_mission_template(name: str) -> MissionTemplate:
    """Return a built-in mission template by name."""
    for template in MISSION_TEMPLATES:
        if template.name == name:
            return template
    known = ", ".join(template.name for template in MISSION_TEMPLATES)
    raise KeyError(f"Unknown mission template '{name}'. Available templates: {known}")


def apply_mission_template(
    manager: CoordinationManager,
    template_name: str,
    *,
    mission_id: str,
    actor: str = "coordinator",
) -> dict[str, Any]:
    """Create a mission, workers, and assignments from a built-in template."""
    template = get_mission_template(template_name)
    events: list[dict[str, Any]] = []

    mission_result = manager.start_mission(mission_id, objective=template.objective, actor=actor)
    events.append(_event_payload("mission", mission_result.event))

    for worker in template.workers:
        worker_result = manager.create_worker(mission_id, worker.worker_id, actor=actor)
        events.append(_event_payload("worker", worker_result.event))
        assignment_result = manager.assign(mission_id, worker.worker_id, worker.assignment, actor=actor)
        events.append(_event_payload("assignment", assignment_result.event))

    return {
        "template": template.name,
        "mission_id": mission_id,
        "objective": template.objective,
        "worker_count": len(template.workers),
        "assignment_count": len(template.workers),
        "workers": [worker.to_dict() for worker in template.workers],
        "acceptance_criteria": list(template.acceptance_criteria),
        "events": events,
    }


def _event_payload(kind: str, event: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "event_seq": event.seq,
        "event_hash": event.hash,
        "event_type": event.type,
    }
