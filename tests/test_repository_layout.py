from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

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
    "frozen_source", "executed_source", "source_snapshots",
    "terminal_acceptance",  # retained terminal artifacts, not maintained code
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

@lru_cache(maxsize=1)
def tracked_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True,
    )
    return [REPO_ROOT / name.decode("utf-8") for name in result.stdout.split(b"\0") if name]


@lru_cache(maxsize=None)
def changed_paths(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "-z", "HEAD", "--"], cwd=root,
        capture_output=True, check=False,
    )
    if result.returncode:
        return {"*"}
    return {name.decode("utf-8") for name in result.stdout.split(b"\0") if name}


@lru_cache(maxsize=None)
def canonical_bytes(path: Path) -> bytes:
    """Compare tracked, unchanged evidence using Git's canonical bytes.

    Clean checkout filters (including Windows line endings) are Git's concern;
    changed/untracked files are checked byte-for-byte without normalization.
    """
    relative = path.relative_to(REPO_ROOT).as_posix()
    changed = changed_paths(REPO_ROOT)
    if "*" not in changed and relative not in changed:
        blob = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=REPO_ROOT,
            capture_output=True, check=False,
        )
        if blob.returncode == 0:
            return blob.stdout
    return path.read_bytes()


def receipt_artifact_matches(path: Path, expected: str) -> bool:
    try:
        return hashlib.sha256(canonical_bytes(path)).hexdigest() == expected
    except OSError:
        return False


@lru_cache(maxsize=1)
def closed_experiment_dirs() -> set[Path]:
    """Only an experiment-level, hash-bound terminal receipt retires its code.

    An arm receipt cannot retire an unresolved multi-arm question. Full scientific
    acceptance remains owned by RML; this only selects maintained layout scope.
    """
    closed = set()
    root_receipts = set()
    finalized_ids: dict[Path, set[str]] = {}
    for receipt_path in tracked_paths():
        if receipt_path.name != "finalization.json":
            continue
        relative = receipt_path.relative_to(REPO_ROOT)
        if (len(relative.parts) < 4 or relative.parts[0] != "experiments"
                or relative.parts[-2] != "rml_finalized"):
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            trajectory_path = receipt_path.parent / "trajectory.json"
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            outcome = receipt["outcome"]
            decision = trajectory["decision"]
            decision_path = REPO_ROOT / decision["decision_ref"]
            experiment = REPO_ROOT / "experiments" / relative.parts[1]
            if (receipt.get("format") != "molgap-rml-finalization-v1"
                    or outcome in ("ACTIVE", "PENDING", "", None)
                    or decision.get("outcome") != outcome
                    or receipt.get("trajectory_id") != trajectory.get("trajectory_id")
                    or not decision_path.resolve().is_relative_to(experiment.resolve())):
                continue
            bindings = {trajectory_path: receipt["published_hashes"]["trajectory.json"]}
            bindings.update({REPO_ROOT / name: digest for name, digest in
                             receipt["input_artifact_hashes"].items()
                             if name.startswith(experiment.relative_to(REPO_ROOT).as_posix() + "/")
                             and Path(name).suffix in (".py", ".md", ".json")})
            if decision_path not in bindings:
                continue
            if all(receipt_artifact_matches(path, digest) for path, digest in bindings.items()):
                finalized_ids.setdefault(experiment, set()).add(receipt["trajectory_id"])
                if relative.parts[2:] == ("rml", "rml_finalized", "finalization.json"):
                    root_receipts.add(experiment)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for experiment, terminal_ids in finalized_ids.items():
        originals = [path for path in tracked_paths()
                     if path.name == "trajectory.json" and path.is_relative_to(experiment)
                     and "rml_finalized" not in path.relative_to(experiment).parts
                     and not EXCLUDED_PARTS.intersection(path.relative_to(experiment).parts)
                     and "input_authority" not in path.relative_to(experiment).parts]
        try:
            ids = {json.loads(path.read_text(encoding="utf-8"))["trajectory_id"] for path in originals}
            if (ids and ids <= terminal_ids) or (not ids and experiment in root_receipts):
                closed.add(experiment)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return closed


@lru_cache(maxsize=1)
def active_files() -> list[Path]:
    closed = closed_experiment_dirs()
    return [path for path in tracked_paths()
            if path.is_file()
            and not EXCLUDED_PARTS.intersection(path.relative_to(REPO_ROOT).parts)
            and not any(path.is_relative_to(owner) for owner in closed)]


@lru_cache(maxsize=1)
def frozen_navigation_documents() -> set[Path]:
    """Frozen copied/context documents keep their original owner's links.

    Maintained entrypoints remain checked even if some run binds their bytes.
    Only explicit retained-reference inventories and trajectory source bindings
    grant this navigation scope; artifact integrity still uses the bound digest.
    """
    documents = set()
    for record_path in tracked_paths():
        relative = record_path.relative_to(REPO_ROOT)
        if (len(relative.parts) < 3 or relative.parts[0] != "experiments"
                or record_path.name not in ("manifest.json", "trajectory.json")):
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            bindings = {}
            if record.get("format") == "molgap-retained-reference-files-v1":
                bindings.update({item["path"]: item["sha256"] for item in record["files"]})
            if record_path.name == "trajectory.json":
                bindings.update(record.get("decision_state", {}).get("source_hashes", {}))
            owner = REPO_ROOT / "experiments" / relative.parts[1]
            for name, digest in bindings.items():
                document = REPO_ROOT / name.replace("\\", "/")
                if (document.suffix == ".md" and document.name not in ("README.md", "STATUS.md", "NAVIGATION.md")
                        and document.resolve().is_relative_to(owner.resolve())
                        and receipt_artifact_matches(document, digest)):
                    documents.add(document)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return documents


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
    return sorted(path for path in active_files()
                  if path.suffix == ".py" and path.relative_to(REPO_ROOT).parts[0] in ACTIVE_ROOTS)


def active_markdown_files() -> list[Path]:
    return sorted(path for path in active_files() if path.suffix == ".md"
                  and (path.relative_to(REPO_ROOT).parts[0] in ACTIVE_ROOTS
                       or path.name in REPO_CONTROL_DOCS and path.parent == REPO_ROOT))


def argparse_clis() -> list[Path]:
    return [
        path
        for path in active_python_files()
        if "argparse" in path.read_text(encoding="utf-8", errors="ignore")
    ]


def active_experiment_dirs() -> list[Path]:
    return sorted({REPO_ROOT / "experiments" / relative.parts[1]
                   for path in active_files()
                   if len((relative := path.relative_to(REPO_ROOT)).parts) > 2
                   and relative.parts[0] == "experiments"
                   and not relative.parts[1].startswith("_")})


def resolve_doc_pointer(document: Path, raw_pointer: str) -> Path | None:
    pointer = unquote(raw_pointer.strip().removeprefix("<").removesuffix(">")).split("#", 1)[0]
    if (not pointer or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", pointer)
            or pointer.startswith(("/", "\\"))
            or any(token in pointer for token in ("*", "<", ">"))):
        return None
    normalized = pointer.replace("\\", "/")
    # Explicit relative links are relative even when the basename is README.md.
    local = (document.parent / normalized).resolve()
    if pointer.startswith(("./", "../")) or local.exists():
        return local
    first = normalized.split("/", 1)[0]
    if first in REPO_POINTER_ROOTS or normalized in REPO_CONTROL_DOCS:
        return (REPO_ROOT / normalized).resolve()
    return local


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
                failures.append(
                    f"{experiment.name}: no decision points to reachable JSON evidence"
                )

    assert not failures, "experiment traceability failures:\n" + "\n".join(failures)


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
            if not document.is_file():
                continue  # The entrypoint test reports missing README files.
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
        if document in frozen_navigation_documents():
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
    names = {path.relative_to(train_root).parts[0] for path in active_files()
             if path.is_relative_to(train_root) and len(path.relative_to(train_root).parts) > 1}
    failures = sorted(name for name in names
                      if not name.startswith("_") and name != "scripts"
                      and MODEL_RESULT_NAME.fullmatch(name) is None)
    assert not failures, (
        "production model directories must encode composition, training rows, "
        "and version:\n" + "\n".join(failures)
    )


def test_active_directories_use_role_or_question_names() -> None:
    failures: list[str] = []
    directories = {parent for path in active_files() for parent in path.parents
                   if parent != REPO_ROOT and parent.is_relative_to(REPO_ROOT)
                   and parent.relative_to(REPO_ROOT).parts[0] in ACTIVE_ROOTS}
    for path in sorted(directories):
        if BANNED_ACTIVE_DIRECTORY.search(path.name) or CALENDAR_DATE.search(path.name):
            failures.append(str(path.relative_to(REPO_ROOT)))
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
    experiment = REPO_ROOT / "experiments/pcqm_k1_slot_readout_diagnostic"
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


@pytest.fixture
def layout_tree(tmp_path, monkeypatch):
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    paths = []
    monkeypatch.setitem(globals(), "tracked_paths", lambda: paths)
    monkeypatch.setitem(globals(), "canonical_bytes", lambda path: path.read_bytes())
    closed_experiment_dirs.cache_clear()
    active_files.cache_clear()
    frozen_navigation_documents.cache_clear()
    yield tmp_path, paths
    closed_experiment_dirs.cache_clear()
    active_files.cache_clear()
    frozen_navigation_documents.cache_clear()


def test_active_scope_uses_tracked_files_and_excludes_frozen_artifacts(layout_tree):
    root, tracked = layout_tree
    for name in ("experiments/live/run.py", "experiments/live/README.md",
                 "experiments/live/frozen_source/stale.py",
                 "experiments/live/executed_source/stale.md",
                 "experiments/live/source_snapshots/20261001/stale.py",
                 "experiments/live/terminal_acceptance/outputs/20261001/trace.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        tracked.append(path)
    untracked = root / "experiments/leftover/run.py"
    untracked.parent.mkdir(parents=True)
    untracked.write_text("", encoding="utf-8")
    assert set(active_files()) == {root / "experiments/live/run.py", root / "experiments/live/README.md"}
    assert active_experiment_dirs() == [root / "experiments/live"]


@pytest.mark.parametrize("mutation", (None, "source", "receipt", "trajectory", "arm_only", "active", "closed_arms", "unresolved_arm", "unresolved_root"))
def test_terminal_scope_requires_matching_receipt_bindings(layout_tree, mutation):
    root, tracked = layout_tree
    experiment = root / "experiments/question"
    final = experiment / ("rml/arm/rml_finalized" if mutation in ("arm_only", "closed_arms", "unresolved_arm") else "rml/rml_finalized")
    final.mkdir(parents=True)
    source = experiment / "analyze.py"
    source.write_bytes(b"# frozen historical source\n")
    decision = experiment / "decision.md"
    decision.write_bytes(b"# Terminal decision\n")
    trajectory = final / "trajectory.json"
    outcome = "ACTIVE" if mutation == "active" else "NO_TRAIN"
    trajectory.write_text(json.dumps({"trajectory_id": "synthetic", "decision": {
        "outcome": outcome, "decision_ref": "experiments/question/decision.md"}}), encoding="utf-8")
    receipt = final / "finalization.json"
    record = {"format": "molgap-rml-finalization-v1", "outcome": outcome,
              "trajectory_id": "synthetic", "published_hashes": {
                  "trajectory.json": hashlib.sha256(trajectory.read_bytes()).hexdigest()},
              "input_artifact_hashes": {
                  path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (source, decision)}}
    if mutation == "receipt":
        record["input_artifact_hashes"]["experiments/question/analyze.py"] = "0" * 64
    receipt.write_text(json.dumps(record), encoding="utf-8")
    if mutation == "source":
        source.write_bytes(b"# changed\n")
    if mutation == "trajectory":
        trajectory.write_bytes(trajectory.read_bytes() + b" ")
    tracked.extend((source, decision, trajectory, receipt))
    if mutation in ("closed_arms", "unresolved_arm"):
        original = final.parent / "trajectory.json"
        original.write_bytes(trajectory.read_bytes())
        tracked.append(original)
    if mutation in ("unresolved_arm", "unresolved_root"):
        pending = experiment / "rml/pending/trajectory.json"
        pending.parent.mkdir(parents=True)
        pending.write_text(json.dumps({"trajectory_id": "pending-arm"}), encoding="utf-8")
        tracked.append(pending)
    retired = mutation in (None, "closed_arms")
    assert (experiment in closed_experiment_dirs()) is retired
    assert (source in active_files()) is not retired


@pytest.mark.parametrize("pointer, relative", (
    ("README.md", "experiments/question/README.md"),
    ("./README.md#heading", "experiments/question/README.md"),
    ("../../README.md", "README.md"),
    ("src/example.py", "src/example.py"),
    ("<a%20file.md>", "experiments/question/a file.md"),
    ("D:/old-checkout/decision.md", None),
    (r"D:\old-checkout\decision.md", None),
    ("file:///D:/old-checkout/decision.md", None),
    ("https://example.org/README.md", None),
))
def test_document_pointer_resolution(layout_tree, pointer, relative):
    root, _ = layout_tree
    document = root / "experiments/question/README.md"
    document.parent.mkdir(parents=True)
    document.write_text("", encoding="utf-8")
    assert resolve_doc_pointer(document, pointer) == (root / relative if relative else None)



def test_receipt_hashes_use_git_checkout_filters_without_hash_allowlists(tmp_path, monkeypatch):
    for args in (("init",), ("config", "core.autocrlf", "true"),
                 ("config", "user.name", "Layout test"),
                 ("config", "user.email", "layout@example.invalid")):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    artifact = tmp_path / "source.py"
    original = b"# retained source\n"
    artifact.write_bytes(original)
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-m", "fixture"],
                   cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    expected = hashlib.sha256(original).hexdigest()
    artifact.write_bytes(original.replace(b"\n", b"\r\n"))
    assert receipt_artifact_matches(artifact, expected)
    artifact.write_bytes(b"# edited source\r\n")
    canonical_bytes.cache_clear()
    changed_paths.cache_clear()
    assert not receipt_artifact_matches(artifact, expected)


@pytest.mark.parametrize("binding", ("reference", "trajectory"))
@pytest.mark.parametrize("mutation", (None, "document", "digest"))
def test_frozen_navigation_requires_digest_and_keeps_entrypoints_live(layout_tree, binding, mutation):
    root, tracked = layout_tree
    owner = root / "experiments/question"
    owner.mkdir(parents=True)
    document = owner / "evidence_review.md"
    entrypoint = owner / "README.md"
    document.write_bytes(b"[original context](../missing.md)\n")
    entrypoint.write_bytes(b"# Maintained entrypoint\n")
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (document, entrypoint)}
    if mutation == "digest":
        hashes[document.relative_to(root).as_posix()] = "0" * 64
    record = owner / ("manifest.json" if binding == "reference" else "trajectory.json")
    payload = ({"format": "molgap-retained-reference-files-v1", "files": [
        {"path": name, "sha256": digest} for name, digest in hashes.items()]}
        if binding == "reference" else {"decision_state": {"source_hashes": hashes}})
    record.write_text(json.dumps(payload), encoding="utf-8")
    if mutation == "document":
        document.write_bytes(b"edited\n")
    tracked.extend((document, entrypoint, record))
    assert (document in frozen_navigation_documents()) is (mutation is None)
    assert entrypoint not in frozen_navigation_documents()
