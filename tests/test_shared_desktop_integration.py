"""Server integration guards; no model execution, data access or remote work."""
import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest

from molgap import k1_pair_value, qm9_neural_atom
from molgap.research_memory import cli, terminal_wiring


def _git_blob(path):
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def test_pair_value_copy_preserves_desktop_bytes_and_server_history():
    root = Path(__file__).resolve().parents[1]
    # Pinned donor a47b945f; rename only, never overwrite frozen server modes.
    assert _git_blob(root / "src/molgap/k1_pair_value.py") == (
        "6c4801fdf60a933ddec04d06292585d10a6d3bf8"
    )
    assert _git_blob(root / "src/molgap/k1_pair_token.py") == (
        "05d16c7a9193f3fb6dcb4600ab741b9a76830897"
    )


def test_copied_value_mode_dispatch_without_real_model(monkeypatch):
    base, token = object(), object()
    base_factory = Mock(return_value=base)
    token_factory = Mock(return_value=token)
    monkeypatch.setattr(qm9_neural_atom, "make_encoder", base_factory)
    monkeypatch.setattr(k1_pair_value._PairTokenFactory, "make", token_factory)
    wrapper = k1_pair_value.make_encoder(k1_pair_value.VALUE_DECOUPLED_MODE)
    assert wrapper.base is base and wrapper.relation_token is token
    base_factory.assert_called_once_with("neural_atom_k1")
    token_factory.assert_called_once_with()


def test_terminal_cli_uses_shared_trace_wiring(tmp_path, monkeypatch):
    closer = Mock(return_value={"pipeline_status": "COMPLETE"})
    monkeypatch.setattr(terminal_wiring, "close_terminal_arm", closer)
    cli.main(["--repo-root", str(tmp_path), "terminal-pipeline",
              "--trajectory", "trajectory.json", "--terminal", "terminal.json",
              "--trace-source", "trace.json", "--arm", "candidate"])
    closer.assert_called_once_with(
        repo_root=tmp_path.resolve(), trajectory="trajectory.json",
        terminal="terminal.json", trace=None, arm_identifier="candidate",
        trace_source="trace.json",
    )


@pytest.mark.parametrize("problem", ["clean", "missing", "dirty", "artifact"])
def test_portable_cli_does_not_ignore_missing_evidence(tmp_path, monkeypatch, problem):
    monkeypatch.setattr(cli, "frozen_differences", lambda _: [])
    monkeypatch.setattr(cli, "validate_repository_records", lambda _: {"records": {}})
    monkeypatch.setattr(cli, "committed_head_differences", lambda *_: {
        "missing_from_head": ["missing.json"] if problem == "missing" else [],
        "changed_from_head": ["dirty.py"] if problem == "dirty" else [],
    })
    monkeypatch.setattr(cli, "missing_locally_claimed_artifacts",
                        lambda *_: ["model.pt"] if problem == "artifact" else [])
    args = ["--repo-root", str(tmp_path), "check", "--frozen", "--portable"]
    if problem == "clean":
        cli.main(args)
    else:
        with pytest.raises(SystemExit):
            cli.main(args)


def test_public_terminal_exports_are_same_shared_functions():
    import molgap.research_memory as rml

    for name in ("close_terminal_arm", "close_terminal_multi_arm",
                 "resolve_trace_for_terminal_arm", "build_default_trace_manifest"):
        assert getattr(rml, name) is getattr(terminal_wiring, name)
