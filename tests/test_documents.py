"""Tests for filesystem document ingestion."""

from __future__ import annotations

from pathlib import Path

from zaxy.documents import collect_document_events


class TestCollectDocumentEvents:
    """Tests for turning local files into durable document events."""

    def test_collects_markdown_chunks_with_line_citations(self, tmp_path: Path) -> None:
        doc = tmp_path / "docs" / "guide.md"
        doc.parent.mkdir()
        doc.write_text("# Guide\n\nAlpha context\nBeta context\nGamma context\n", encoding="utf-8")

        events = collect_document_events(tmp_path, max_lines=3)

        assert [event["event_type"] for event in events] == ["document.indexed", "document.indexed"]
        assert events[0]["payload"] == {
            "path": "docs/guide.md",
            "start_line": 1,
            "end_line": 3,
            "content": "# Guide\n\nAlpha context",
            "sha256": events[0]["payload"]["sha256"],
        }
        assert "labels" not in events[0]["payload"]
        assert "purpose_label" not in events[0]["payload"]
        assert "entity_type" not in events[0]["payload"]
        assert events[1]["payload"]["start_line"] == 4
        assert events[1]["payload"]["end_line"] == 5
        assert events[1]["payload"]["content"] == "Beta context\nGamma context"

    def test_skips_hidden_directories_and_unsupported_files(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("secret-ish config", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
        (tmp_path / "README.txt").write_text("Index me", encoding="utf-8")

        events = collect_document_events(tmp_path)

        assert len(events) == 1
        assert events[0]["payload"]["path"] == "README.txt"
