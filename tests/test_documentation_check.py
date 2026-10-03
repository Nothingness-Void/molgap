from __future__ import annotations

import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.documentation_check import (
    ENTRYPOINT_LINE_LIMITS,
    audit_repository,
    main,
    markdown_anchors,
)


def test_markdown_anchors_handle_duplicates_and_explicit_ids() -> None:
    anchors = markdown_anchors(
        "# API Guide\n\n# API Guide\n\n<span id=\"custom-entry\"></span>\n"
    )
    assert {"api-guide", "api-guide-1", "custom-entry"} <= anchors


def test_audit_checks_local_links_fragments_and_code_pointers(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("# API\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Entry\n"
        "[good](guide.md#api)\n"
        "[bad anchor](guide.md#missing)\n"
        "[bad path](missing.md)\n"
        "`guide.md`\n"
        "```md\n[ignored](missing-in-fence.md)\n```\n"
        "[remote](https://example.invalid/missing.md)\n",
        encoding="utf-8",
    )

    result = audit_repository(tmp_path, documents=[readme])

    assert result.documents_scanned == 1
    assert result.pointers_checked == 4
    assert result.anchors_checked == 2
    assert [(issue.kind, issue.pointer) for issue in result.issues] == [
        ("missing_anchor", "guide.md#missing"),
        ("missing_path", "missing.md"),
    ]


def test_audit_rejects_escaping_pointer_and_enforces_optional_line_limit(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Entry\n[escape](../outside.md)\n", encoding="utf-8")

    result = audit_repository(
        tmp_path,
        documents=[readme],
        line_limits={"README.md": 1},
    )

    assert {issue.kind for issue in result.issues} == {"unsafe_path", "line_limit"}


def test_cli_is_machine_readable_and_read_only(tmp_path: Path, capsys) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Entry\n[good](guide.md)\n", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Guide\n", encoding="utf-8")

    assert main(["--repo-root", str(tmp_path), "--document", "README.md"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["documents_scanned"] == 1
    assert readme.read_text(encoding="utf-8") == "# Entry\n[good](guide.md)\n"


def test_cli_budget_guard_returns_nonzero_for_an_oversized_entrypoint(
    tmp_path: Path, capsys,
) -> None:
    document = tmp_path / "AGENTS.md"
    document.write_text("line\n" * (ENTRYPOINT_LINE_LIMITS["AGENTS.md"] + 1), encoding="utf-8")

    assert main([
        "--repo-root", str(tmp_path), "--document", "AGENTS.md",
        "--check-entrypoint-budgets",
    ]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["issues"][0]["kind"] == "line_limit"


def test_frozen_documents_are_skipped_unless_requested(tmp_path: Path) -> None:
    frozen = tmp_path / "docs" / "archive" / "old.md"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("[missing](gone.md)\n", encoding="utf-8")

    skipped = audit_repository(tmp_path, documents=[frozen])
    included = audit_repository(tmp_path, documents=[frozen], include_frozen=True)

    assert skipped.documents_scanned == 0
    assert skipped.skipped_documents == ("docs/archive/old.md",)
    assert [issue.kind for issue in included.issues] == ["missing_path"]


def test_entrypoint_budgets_are_stable_and_do_not_bound_conditional_indexes() -> None:
    assert ENTRYPOINT_LINE_LIMITS["AGENTS.md"] == 170
    assert ENTRYPOINT_LINE_LIMITS["CURRENT_STATE.md"] == 120
    assert ENTRYPOINT_LINE_LIMITS["ROADMAP.md"] == 120
    assert ENTRYPOINT_LINE_LIMITS["ARCHITECTURE.md"] == 140
    assert ENTRYPOINT_LINE_LIMITS["experiments/README.md"] == 90
    assert ENTRYPOINT_LINE_LIMITS["docs/operations/EXPERIMENT_QUICKSTART.md"] == 160
    assert "experiments/DIRECTORY_INDEX.md" not in ENTRYPOINT_LINE_LIMITS
    assert "experiments/EVIDENCE_INDEX.md" not in ENTRYPOINT_LINE_LIMITS


def test_tracked_active_repository_docs_pass_navigation_and_budget_guard() -> None:
    result = audit_repository(REPO_ROOT, line_limits=ENTRYPOINT_LINE_LIMITS)

    assert result.ok, "active documentation issues:\n" + "\n".join(
        f"{issue.document}:{issue.line} {issue.kind} {issue.pointer}"
        for issue in result.issues
    )
    assert result.documents_scanned >= 700
    assert result.anchors_checked >= 10


def test_graph_screen_extension_is_auditable_when_supplied_explicitly() -> None:
    document = REPO_ROOT / "docs" / "operations" / "GRAPH_SCREEN_EXTENSION.md"
    assert document.is_file()
    result = audit_repository(
        REPO_ROOT,
        documents=[document],
        line_limits=ENTRYPOINT_LINE_LIMITS,
    )
    assert result.ok, "graph screen contract issues: " + "; ".join(
        f"{issue.line}:{issue.kind}:{issue.pointer}" for issue in result.issues
    )
