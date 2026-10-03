from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

from conftest import CLI_MODULES
from molgap.constants import REPO_ROOT

ACTIVE_ROOTS = ("production", "experiments", "platforms")
EXCLUDED_PARTS = {
    "history",       # frozen phase 1-7 reproduction
    "_closed",       # settled experiments
    "_retired",      # replaced production evidence
    "_records",      # downloaded remote payloads
    "_staging",      # generated source snapshots, not active implementation
    "packages",      # generated Kaggle copies, not source entrypoints
    "bundle",        # frozen remote wheel/notebook payloads
    "__pycache__",
}
STALE_PATH_SNIPPETS = (
    '"results/phase8',
    '"scripts/phase8',
    "'results/phase8",
    "'scripts/phase8",
)
MODEL_RESULT_NAME = re.compile(
    r"^(?:routed_)?[a-z0-9]+(?:_[a-z0-9]+)*_(?:\d+k|\d+m)_v\d+$"
)
BANNED_ACTIVE_DIRECTORY = re.compile(
    r"(^|_)(new|latest|final|best|tmp|temp|misc|phase\d+)(_|$)"
)
CALENDAR_DATE = re.compile(r"(?:19|20)\d{6}")
ROOT_POINTER = re.compile(
    r"`((?:(?:production|experiments|platforms|models|src|tests|data)/[^`\n]+)"
    r"|(?:AGENTS|CURRENT_STATE|ROADMAP|TRACKS|ARCHITECTURE|NAMING|README)\.md)`"
)
LOCAL_ARTIFACT_POINTER = re.compile(r"`([^`\n]+\.(?:md|json))`")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\n]+)\)")
REPO_POINTER_ROOTS = {
    "data",
    "experiments",
    "models",
    "platforms",
    "production",
    "src",
    "tests",
}
REPO_CONTROL_DOCS = {
    "AGENTS.md",
    "ARCHITECTURE.md",
    "CURRENT_STATE.md",
    "NAMING.md",
    "README.md",
    "ROADMAP.md",
    "TRACKS.md",
}
LOCAL_LINK_SUFFIXES = (
    ".csv",
    ".ipynb",
    ".jpeg",
    ".jpg",
    ".json",
    ".md",
    ".parquet",
    ".pdf",
    ".png",
    ".pt",
    ".py",
    ".svg",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
)
CONTROL_DOC_LIMITS = {
    "CURRENT_STATE.md": 120,
    "ROADMAP.md": 120,
}
TOP_LEVEL_POINTER_DOCS = (
    "AGENTS.md",
    "CURRENT_STATE.md",
    "ROADMAP.md",
    "TRACKS.md",
    "ARCHITECTURE.md",
    "NAMING.md",
    "README.md",
    "experiments/README.md",
    "production/README.md",
    "platforms/README.md",
)

ORACLE_DIR = "experiments/pcqm_expert_oracle_feasibility"
ORACLE_FINALIZATION = f"{ORACLE_DIR}/rml/rml_finalized/finalization.json"
SLOT_DIR = "experiments/pcqm_k1_slot_readout_diagnostic"
SLOT_FINALIZATION = f"{SLOT_DIR}/rml/rml_finalized/finalization.json"
# Only these named historical artifacts can receive the two allowances below.
FROZEN_ARTIFACT_DIGESTS = {
    f"{ORACLE_DIR}/analyze.py": "8b43cf0f050bbc3bd801a2befe2ed782d742d3a1871da4cc06da2e190c467a30",
    f"{ORACLE_DIR}/decision.md": "81dae767d7d1b7fabcc6f9d5c362bccac64feb007e84b83574e7e35432ca7ac1",
    f"{ORACLE_DIR}/analysis.json": "58bbd71c011ceb3bd57a696736198e18eea5a52dd6fff05c8de4f74a3428ab3d",
    f"{SLOT_DIR}/prepare.py": "822053385b584ff424b358f55915fd92ccadce82bba35b570572f2df3f1863e6",
    f"{SLOT_DIR}/finalize_analysis.py": "b600ae7e9cb99de9f09720fc4febab38e98d6eff948a085a3b4be4357c6f62bd",
    f"{SLOT_DIR}/terminal_decision.md": "db39710b327f5b2ec0713e07e82b7790dd052cbc0e1e4f9c42e4396b2e463dd5",
    f"{SLOT_DIR}/results/analysis.json": "d0c7544ec13a1316c31264175b0259e25818cbd41e504581e52e96b7118952e2",
    f"{SLOT_DIR}/input_authority/terminal_decision.md": "7e040041be9d1f9fcd768ded2612f87330793ea243f8b55349a02c82a2199772",
    f"{SLOT_DIR}/input_authority/scientific_metrics.json": "b6babc6b8eb954d69cea6aea1a14377f4a4d1ed8bbd5328d992f3ef7a3c7c4b9",
    f"{SLOT_DIR}/inputs.json": "2aa4c07861260336205b139b7b988026ba95698644f3e8c9ab45c19e16b0183f",
}
FROZEN_ROOT_DEPTH_SCRIPTS = {
    f"{ORACLE_DIR}/analyze.py", f"{SLOT_DIR}/prepare.py", f"{SLOT_DIR}/finalize_analysis.py",
}
# This exact Windows checkout variant predates maintenance. Do not normalize
# arbitrary sources: both its raw bytes and the receipt-bound LF bytes must match.
FROZEN_CRLF_DIGESTS = {
    f"{ORACLE_DIR}/analyze.py": "1d077d1194f372e7816f24605c5b119ea49ea17d984b62ca06273c1bae946d8f",
}
IMMUTABLE_DECISION_NAVIGATION = {
    f"{ORACLE_DIR}/decision.md": f"{ORACLE_DIR}/analysis.json",
    f"{SLOT_DIR}/terminal_decision.md": f"{SLOT_DIR}/results/analysis.json",
}


def frozen_receipt(relative: str) -> str:
    return SLOT_FINALIZATION if relative.startswith(SLOT_DIR + "/") else ORACLE_FINALIZATION


def verified_frozen_artifact(relative: str) -> bool:
    expected = FROZEN_ARTIFACT_DIGESTS.get(relative)
    if expected is None:
        return False
    try:
        receipt = json.loads((REPO_ROOT / frozen_receipt(relative)).read_text(encoding="utf-8"))
        if receipt.get("input_artifact_hashes", {}).get(relative) != expected:
            return False
        payload = (REPO_ROOT / relative).read_bytes()
    except (OSError, ValueError):
        return False
    actual = hashlib.sha256(payload).hexdigest()
    return actual == expected or (
        actual == FROZEN_CRLF_DIGESTS.get(relative)
        and hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest() == expected
    )


def frozen_root_depth_snapshot(path: Path) -> bool:
    relative = path.relative_to(REPO_ROOT).as_posix()
    return relative in FROZEN_ROOT_DEPTH_SCRIPTS and verified_frozen_artifact(relative)


def supplementary_machine_evidence(decision: Path, readme: Path) -> bool:
    relative = decision.relative_to(REPO_ROOT).as_posix()
    evidence = IMMUTABLE_DECISION_NAVIGATION.get(relative)
    if evidence is None or not all(
        verified_frozen_artifact(item) for item in (relative, evidence)
    ):
        return False
    pointers = MARKDOWN_LINK.findall(readme.read_text(encoding="utf-8"))
    resolved = {resolve_doc_pointer(readme, pointer) for pointer in pointers}
    return decision in resolved and REPO_ROOT / evidence in resolved


def frozen_snapshot_navigation(document: Path) -> bool:
    """Only the named receipt-bound copy may keep its owner's relative links."""
    relative = document.relative_to(REPO_ROOT).as_posix()
    snapshot = f"{SLOT_DIR}/input_authority/terminal_decision.md"
    required = (snapshot, f"{SLOT_DIR}/input_authority/scientific_metrics.json", f"{SLOT_DIR}/inputs.json")
    if relative != snapshot or not all(verified_frozen_artifact(item) for item in required):
        return False
    readme = REPO_ROOT / SLOT_DIR / "README.md"
    resolved = {resolve_doc_pointer(readme, p) for p in MARKDOWN_LINK.findall(readme.read_text(encoding="utf-8"))}
    return all(REPO_ROOT / item in resolved for item in required)


def experiment_is_indexed(index: str, experiment: Path) -> bool:
    if f"`{experiment.name}/`" in index:
        return True
    owner = REPO_ROOT / "experiments/README.md"
    return any(
        target is not None and target.resolve().is_relative_to(experiment.resolve())
        for target in (resolve_doc_pointer(owner, p) for p in MARKDOWN_LINK.findall(index))
    )


def has_main_guard(source: str) -> bool:
    for node in ast.parse(source).body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        left, right = test.left, test.comparators[0]
        if isinstance(right, ast.Name):
            left, right = right, left
        if (isinstance(left, ast.Name) and left.id == "__name__"
                and isinstance(right, ast.Constant) and right.value == "__main__"):
            return True
    return False


def active_python_files() -> list[Path]:
    files: list[Path] = []
    for root in ACTIVE_ROOTS:
        for path in (REPO_ROOT / root).rglob("*.py"):
            if EXCLUDED_PARTS.intersection(path.relative_to(REPO_ROOT).parts):
                continue
            files.append(path)
    return sorted(files)


def active_markdown_files() -> list[Path]:
    files = [
        REPO_ROOT / name
        for name in (
            "AGENTS.md",
            "ARCHITECTURE.md",
            "CURRENT_STATE.md",
            "NAMING.md",
            "README.md",
            "ROADMAP.md",
            "TRACKS.md",
        )
    ]
    for root in ACTIVE_ROOTS:
        for path in (REPO_ROOT / root).rglob("*.md"):
            if EXCLUDED_PARTS.intersection(path.relative_to(REPO_ROOT).parts):
                continue
            files.append(path)
    return sorted(set(files))


def argparse_clis() -> list[Path]:
    return [
        path
        for path in active_python_files()
        if "argparse" in path.read_text(encoding="utf-8", errors="ignore")
    ]


def active_experiment_dirs() -> list[Path]:
    return sorted(
        path
        for path in (REPO_ROOT / "experiments").iterdir()
        if path.is_dir() and not path.name.startswith("_")
        # Import caches can remain after a tracked experiment is moved.
        and any(
            item.is_file()
            and not EXCLUDED_PARTS.intersection(item.relative_to(path).parts)
            for item in path.rglob("*")
        )
    )


def resolve_doc_pointer(document: Path, raw_pointer: str) -> Path | None:
    pointer = raw_pointer.split("#", maxsplit=1)[0]
    if any(token in pointer for token in ("*", "<", ">")):
        return None
    normalized = pointer.replace("\\", "/")
    first = normalized.split("/", maxsplit=1)[0]
    if first in REPO_POINTER_ROOTS or normalized in REPO_CONTROL_DOCS:
        return REPO_ROOT / normalized
    return document.parent / normalized


def test_active_experiments_have_readme_entrypoints() -> None:
    missing = [
        path.name
        for path in active_experiment_dirs()
        if not (path / "README.md").is_file()
    ]
    assert not missing, "active experiments missing README.md:\n" + "\n".join(missing)


def test_active_experiments_are_indexed_and_traceable() -> None:
    index = (REPO_ROOT / "experiments" / "README.md").read_text(encoding="utf-8")
    failures: list[str] = []
    for experiment in active_experiment_dirs():
        readme_path = experiment / "README.md"
        if not readme_path.is_file():
            continue
        readme = readme_path.read_text(encoding="utf-8", errors="ignore")
        decisions = sorted(experiment.rglob("*decision*.md"))
        evidence = sorted(experiment.rglob("*.json"))
        relative_decisions = [path.relative_to(experiment).as_posix() for path in decisions]

        if not experiment_is_indexed(index, experiment):
            failures.append(f"{experiment.name}: missing from experiments/README.md")
        if not decisions:
            failures.append(f"{experiment.name}: no *decision*.md")
        elif not any(relative in readme for relative in relative_decisions):
            failures.append(f"{experiment.name}: README does not point to a decision")
        if not evidence:
            failures.append(f"{experiment.name}: no JSON machine evidence")
        else:
            reachable_evidence = False
            for decision in decisions:
                decision_text = decision.read_text(encoding="utf-8", errors="ignore")
                for raw_pointer in (LOCAL_ARTIFACT_POINTER.findall(decision_text)
                                    + MARKDOWN_LINK.findall(decision_text)):
                    if not raw_pointer.endswith(".json"):
                        continue
                    resolved = resolve_doc_pointer(decision, raw_pointer)
                    if resolved is not None and resolved.is_file():
                        reachable_evidence = True
                        break
                if reachable_evidence:
                    break
            if not reachable_evidence:
                reachable_evidence = any(
                    supplementary_machine_evidence(decision, readme_path)
                    for decision in decisions
                )
            if not reachable_evidence:
                failures.append(
                    f"{experiment.name}: no decision points to reachable JSON evidence"
                )

    assert not failures, "experiment traceability failures:\n" + "\n".join(failures)


def test_live_control_documents_remain_bounded() -> None:
    failures: list[str] = []
    for relative, limit in CONTROL_DOC_LIMITS.items():
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        line_count = len(text.splitlines())
        if line_count > limit:
            failures.append(f"{relative}: {line_count} lines exceeds {limit}")
    assert not failures, "live control documents accumulated history:\n" + "\n".join(failures)


def test_top_level_document_pointers_exist() -> None:
    failures: list[str] = []
    for relative in TOP_LEVEL_POINTER_DOCS:
        document = REPO_ROOT / relative
        text = document.read_text(encoding="utf-8", errors="ignore")
        for match in ROOT_POINTER.finditer(text):
            pointer = match.group(1).split("#", maxsplit=1)[0].rstrip("/")
            if any(token in pointer for token in ("*", "<", ">")):
                continue
            if not (REPO_ROOT / pointer).exists():
                failures.append(f"{relative}: {match.group(1)}")
    assert not failures, "missing top-level document pointers:\n" + "\n".join(failures)


def test_active_experiment_entrypoint_pointers_exist() -> None:
    failures: list[str] = []
    for experiment in active_experiment_dirs():
        documents = [experiment / "README.md"]
        status = experiment / "STATUS.md"
        if status.is_file():
            documents.append(status)
        for document in documents:
            text = document.read_text(encoding="utf-8", errors="ignore")
            for raw_pointer in LOCAL_ARTIFACT_POINTER.findall(text):
                resolved = resolve_doc_pointer(document, raw_pointer)
                if resolved is not None and not resolved.exists():
                    failures.append(
                        f"{document.relative_to(REPO_ROOT)}: {raw_pointer}"
                    )
    assert not failures, "missing experiment entrypoint pointers:\n" + "\n".join(failures)


def test_active_markdown_links_resolve() -> None:
    failures: list[str] = []
    for document in active_markdown_files():
        if frozen_snapshot_navigation(document):
            continue
        text = document.read_text(encoding="utf-8", errors="ignore")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = target.split("#", maxsplit=1)[0]
            if not target:
                continue
            normalized = target.replace("\\", "/")
            first = normalized.split("/", maxsplit=1)[0]
            looks_local = (
                target.startswith(("./", "../"))
                or first in REPO_POINTER_ROOTS
                or normalized in REPO_CONTROL_DOCS
                or normalized.lower().endswith(LOCAL_LINK_SUFFIXES)
            )
            if not looks_local:
                continue
            resolved = resolve_doc_pointer(document, target)
            if resolved is not None and not resolved.exists():
                failures.append(f"{document.relative_to(REPO_ROOT)}: {raw_target}")
    assert not failures, "broken active Markdown links:\n" + "\n".join(failures)


def test_production_model_result_names_are_self_describing() -> None:
    train_root = REPO_ROOT / "production" / "03_train"
    failures = [
        path.name
        for path in train_root.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and path.name != "scripts"
        and MODEL_RESULT_NAME.fullmatch(path.name) is None
    ]
    assert not failures, (
        "production model directories must encode composition, training rows, "
        "and version:\n" + "\n".join(failures)
    )


def test_active_directories_use_role_or_question_names() -> None:
    failures: list[str] = []
    for root_name in ACTIVE_ROOTS:
        for path in (REPO_ROOT / root_name).rglob("*"):
            if not path.is_dir():
                continue
            relative = path.relative_to(REPO_ROOT)
            if EXCLUDED_PARTS.intersection(relative.parts):
                continue
            if BANNED_ACTIVE_DIRECTORY.search(path.name) or CALENDAR_DATE.search(path.name):
                failures.append(str(relative))
    assert not failures, (
        "active directories contain ambiguous, phase, or calendar names:\n"
        + "\n".join(failures)
    )


def test_cli_aliases_are_complete_and_importable() -> None:
    # conftest imports every alias during collection; this additionally makes a
    # missing/moved file failure explicit instead of silently skipping it.
    for alias, relative in CLI_MODULES.items():
        path = REPO_ROOT / relative
        assert path.is_file(), f"{alias} points to missing CLI: {relative}"
        spec = importlib.util.spec_from_file_location(f"probe_{alias}", path)
        assert spec is not None and spec.loader is not None


def test_active_sources_do_not_encode_removed_phase_trees() -> None:
    failures: list[str] = []
    for path in active_python_files():
        source = path.read_text(encoding="utf-8", errors="ignore")
        for snippet in STALE_PATH_SNIPPETS:
            if snippet in source:
                failures.append(f"{path.relative_to(REPO_ROOT)}: {snippet}")
    assert not failures, "removed tree references:\n" + "\n".join(failures)


def test_active_sources_do_not_infer_repo_root_from_parent_depth() -> None:
    failures: list[str] = []
    for path in active_python_files():
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(path))
        if frozen_root_depth_snapshot(path):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            text = ast.get_source_segment(source, node) or ""
            if "Path(__file__)" in text and ".parents[" in text:
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}:{getattr(node, 'lineno', '?')}"
                )
    assert not failures, (
        "active code derives repository roots from directory depth; "
        "import REPO_ROOT from molgap.constants instead:\n" + "\n".join(failures)
    )


def test_named_frozen_layout_allowances_are_verified() -> None:
    for relative in FROZEN_ARTIFACT_DIGESTS:
        assert verified_frozen_artifact(relative), relative
    assert frozen_root_depth_snapshot(REPO_ROOT / ORACLE_DIR / "analyze.py")
    assert not frozen_root_depth_snapshot(REPO_ROOT / ORACLE_DIR / "prepare.py")
    assert not frozen_root_depth_snapshot(REPO_ROOT / ORACLE_DIR / "finalize_diagnostic.py")
    assert frozen_root_depth_snapshot(REPO_ROOT / SLOT_DIR / "prepare.py")
    assert frozen_root_depth_snapshot(REPO_ROOT / SLOT_DIR / "finalize_analysis.py")
    assert not frozen_root_depth_snapshot(REPO_ROOT / SLOT_DIR / "hydrate_existing_bindings.py")
    assert frozen_snapshot_navigation(REPO_ROOT / SLOT_DIR / "input_authority/terminal_decision.md")
    assert supplementary_machine_evidence(
        REPO_ROOT / ORACLE_DIR / "decision.md", REPO_ROOT / ORACLE_DIR / "README.md"
    )


@pytest.mark.parametrize("relative", list(FROZEN_ARTIFACT_DIGESTS))
@pytest.mark.parametrize("mutation", ["artifact", "receipt"])
def test_changed_hash_invalidates_frozen_allowances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str, mutation: str
) -> None:
    original_root = REPO_ROOT
    for name in (*FROZEN_ARTIFACT_DIGESTS, ORACLE_FINALIZATION, SLOT_FINALIZATION,
                 f"{ORACLE_DIR}/README.md", f"{SLOT_DIR}/README.md"):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((original_root / name).read_bytes())
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    assert verified_frozen_artifact(relative)
    if mutation == "artifact":
        artifact = tmp_path / relative
        artifact.write_bytes(artifact.read_bytes() + b"\n# altered\n")
    else:
        receipt_path = tmp_path / frozen_receipt(relative)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["input_artifact_hashes"][relative] = "0" * 64
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert not verified_frozen_artifact(relative)
    if relative in FROZEN_ROOT_DEPTH_SCRIPTS:
        assert not frozen_root_depth_snapshot(tmp_path / relative)
    if relative in IMMUTABLE_DECISION_NAVIGATION:
        assert not supplementary_machine_evidence(
            tmp_path / relative, (tmp_path / relative).parent / "README.md"
        )
    if relative in (f"{SLOT_DIR}/input_authority/terminal_decision.md",
                    f"{SLOT_DIR}/input_authority/scientific_metrics.json", f"{SLOT_DIR}/inputs.json"):
        assert not frozen_snapshot_navigation(tmp_path / SLOT_DIR / "input_authority/terminal_decision.md")


def test_argparse_clis_have_a_guarded_entrypoint() -> None:
    # Parsers may be constructed inline inside main(), so only require a guarded
    # entrypoint here. Representative subprocess tests below prove --help.
    failures: list[str] = []
    for path in argparse_clis():
        source = path.read_text(encoding="utf-8", errors="ignore")
        if not has_main_guard(source):
            failures.append(str(path.relative_to(REPO_ROOT)))
    assert not failures, "argparse CLIs missing main guard:\n" + "\n".join(failures)


@pytest.mark.parametrize("source, expected", (
    ("if __name__=='__main__': main()", True),
    ('if "__main__" == __name__: main()', True),
    ('# if __name__ == "__main__": main()', False),
    ('text = "__name__ == __main__"', False),
    ('if __name__ != "__main__": main()', False),
))
def test_main_guard_check_is_syntax_aware(source, expected):
    assert has_main_guard(source) is expected


def test_experiment_index_accepts_markdown_links_without_prefix_collisions():
    experiment = REPO_ROOT / SLOT_DIR
    assert experiment_is_indexed("[entry](pcqm_k1_slot_readout_diagnostic/README.md)", experiment)
    assert not experiment_is_indexed("[entry](pcqm_k1_slot_readout_diagnostic_other/README.md)", experiment)
    assert not experiment_is_indexed("[entry](pcqm_k1_slot_readout_diagnostic/../other/README.md)", experiment)


@pytest.mark.parametrize(
    "relative",
    [
        "production/03_train/scripts/training/train_encoder.py",
        "production/04_evaluate/scripts/evaluation/build_model_report_table.py",
        "production/04_evaluate/scripts/evaluation/benchmark_routed_v4_latency.py",
        "production/04_evaluate/scripts/evaluation/benchmark_repaired_2m_latency.py",
        "production/04_evaluate/scripts/evaluation/smoke_test_repaired_2m_inference.py",
        "production/04_evaluate/scripts/evaluation/verify_repaired_2m_public_inference.py",
        "production/04_evaluate/scripts/evaluation/compare_dft_vs_ml_cost.py",
        "production/04_evaluate/scripts/evaluation/benchmark_experimental_values.py",
        "production/04_evaluate/scripts/evaluation/build_presentation_evidence.py",
        "production/07_database/scripts/build_database.py",
        "experiments/pcqm_route_b/build_pcqm_route_b_1m.py",
        "experiments/pcqm_gine_expert/continue_pcqm_gine_local.py",
        "experiments/_scripts/run_conformer_ab.py",
        "platforms/kaggle/organize_account.py",
    ],
)
def test_representative_cli_help(relative: str) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / relative), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"{relative}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "usage:" in result.stdout.lower()
