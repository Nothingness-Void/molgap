"""Shared desktop portability cases copied without changing server corpus tests."""
import json
import subprocess

from molgap.research_memory.portability import (
    committed_head_differences,
    missing_locally_claimed_artifacts,
    uncommitted_closure_paths,
)


def test_portability_check_detects_uncommitted_pointer_closure(tmp_path):
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")

    source = tmp_path / "experiments" / "trace_manifest.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text("{}", encoding="utf-8")
    sidecar = source.parent / "trace_migration.json"
    sidecar.write_text(
        json.dumps({"source_trace_ref": "source_trace.json"}), encoding="utf-8"
    )
    (tmp_path / "source_trace.json").write_text("{}", encoding="utf-8")
    evidence = tmp_path / "v5_evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "authority": {"pointers": ["contract.json"]},
                "artifacts": [
                    {
                        "locator": "remote_only.pt",
                        "availability": "durable_remote_verified",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    git("add", "experiments/trace_manifest.json", "contract.json")
    git("commit", "-qm", "fixture")

    records = {
        "traces": [(source, {"contract_ref": "contract.json"})],
        "evidence": [
            (
                evidence,
                {
                    **json.loads(evidence.read_text(encoding="utf-8")),
                    "artifacts": [
                        {
                            "locator": "remote_only.pt",
                            "availability": "durable_remote_verified",
                        },
                        {
                            "locator": "locally_retained.pt",
                            "availability": "durable_local_verified",
                        },
                    ],
                },
            )
        ],
    }
    assert missing_locally_claimed_artifacts(tmp_path, records) == [
        f"{evidence}: locally_retained.pt"
    ]
    assert uncommitted_closure_paths(tmp_path, records) == [
        "experiments/trace_migration.json",
        "source_trace.json",
        "v5_evidence.json",
    ]

    git(
        "add",
        "experiments/trace_migration.json",
        "source_trace.json",
        "v5_evidence.json",
    )
    git("commit", "-qm", "close evidence paths")
    assert uncommitted_closure_paths(tmp_path, records) == []
    (tmp_path / "locally_retained.pt").write_bytes(b"artifact")
    assert missing_locally_claimed_artifacts(tmp_path, records) == []


def test_portability_check_rejects_dirty_runtime_and_derived_outputs(tmp_path):
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")

    paths = [
        "pyproject.toml",
        "src/molgap/__init__.py",
        "src/molgap/constants.py",
        "src/molgap/comparison_readiness.py",
        "src/molgap/evidence_pointers.py",
        "src/molgap/v5_common.py",
        "src/molgap/research_memory/__init__.py",
        "src/molgap/research_memory/cli.py",
        "research_memory/schemas/trajectory.json",
        "research_memory/policies/registry.json",
        "experiments/trajectory.json",
        "experiments/contract.json",
    ]
    from molgap.research_memory.compiler import DERIVED_FILENAMES

    paths.extend(f"research_memory/derived/{name}" for name in DERIVED_FILENAMES)
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "baseline")

    source = tmp_path / "experiments/trajectory.json"
    records = {
        "trajectories": [(source, {"contract_ref": "experiments/contract.json"})]
    }
    assert committed_head_differences(tmp_path, records) == {
        "missing_from_head": [],
        "changed_from_head": [],
    }

    (tmp_path / "src/molgap/research_memory/cli.py").write_text(
        "# local runtime change\n", encoding="utf-8"
    )
    (tmp_path / "research_memory/derived/research_summary.json").write_text(
        '{"stale": true}\n', encoding="utf-8"
    )
    untracked_module = tmp_path / "src/molgap/research_memory/new_module.py"
    untracked_module.write_text("# untracked runtime module\n", encoding="utf-8")

    assert committed_head_differences(tmp_path, records) == {
        "missing_from_head": ["src/molgap/research_memory/new_module.py"],
        "changed_from_head": [
            "research_memory/derived/research_summary.json",
            "src/molgap/research_memory/cli.py",
        ],
    }
