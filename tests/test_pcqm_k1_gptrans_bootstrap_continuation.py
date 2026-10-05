"""Focused CPU regressions for the K1/GPTrans 500K bootstrap continuation."""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import ast

import pytest

from molgap.training_reproducibility import sha256_file


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "platforms/kaggle/run_legacy_500k_pair.py"
TRAINER_PATH = ROOT / "src/molgap/pcqm_500k_v4_evidence.py"

_RUNNER_SPEC = importlib.util.spec_from_file_location("_test_legacy_500k_pair", RUNNER_PATH)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(runner)


def _write_resume_case(root: Path):
    arm_id = "k1_pretrained_consistency"
    resume_root = root / "private-resume" / arm_id
    resume_root.mkdir(parents=True)
    payload = resume_root / "last_checkpoint.pt"
    payload.write_bytes(b"synthetic checkpoint bytes")
    manifest = {
        "arm": arm_id,
        "source_sha256": "a" * 64,
        "next_epoch": 7,
        "artifacts": {payload.name: hashlib.sha256(payload.read_bytes()).hexdigest()},
    }
    manifest_path = resume_root / "stage_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    arm = {
        "arm_id": arm_id,
        "resume": {
            "mount": "private-resume",
            "manifest_sha256": sha256_file(manifest_path),
            "source_sha256": manifest["source_sha256"],
            "next_epoch": manifest["next_epoch"],
        },
    }
    return root, arm, payload


def test_start_logged_forwards_flushed_output_before_exit_and_retains_failure(tmp_path, monkeypatch, capsys):
    first_line_forwarded = threading.Event()

    def observed_print(*args, **kwargs):
        builtins.print(*args, **kwargs)
        if args and "first" in str(args[0]):
            first_line_forwarded.set()

    monkeypatch.setattr(runner, "print", observed_print, raising=False)
    log_path = tmp_path / "worker.log"
    code = "import time; print('first', flush=True); time.sleep(1); print('last', flush=True); raise SystemExit(9)"
    worker, forwarder = runner.start_logged(
        [sys.executable, "-u", "-c", code], log_path, label="synthetic-arm"
    )

    assert first_line_forwarded.wait(timeout=3), "first flushed line was not forwarded promptly"
    assert worker.poll() is None, "forwarding happened only after worker exit"
    assert worker.wait(timeout=5) == 9
    forwarder.join(timeout=5)
    assert not forwarder.is_alive()
    assert log_path.read_text(encoding="utf-8") == "first\nlast\n"
    visible = capsys.readouterr().out
    assert "[synthetic-arm] first" in visible
    assert "[synthetic-arm] last" in visible


def test_resolve_resume_accepts_exact_manifest_source_cursor_and_artifact(tmp_path, monkeypatch, capsys):
    mounted, arm, _ = _write_resume_case(tmp_path)
    monkeypatch.setattr(runner, "digest", sha256_file)

    resolved = runner.resolve_resume(mounted, arm)

    assert resolved == mounted / "private-resume" / arm["arm_id"]
    assert f"RESUME VERIFIED {arm['arm_id']} completed_epochs=7" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("manifest", "Pinned resume manifest changed"),
        ("source", "Resume cursor/source changed"),
        ("cursor", "Resume cursor/source changed"),
        ("artifact", "Resume artifact changed"),
    ],
)
def test_resolve_resume_rejects_changed_manifest_source_cursor_or_artifact(
    tmp_path, monkeypatch, tamper, message
):
    mounted, arm, payload = _write_resume_case(tmp_path)
    monkeypatch.setattr(runner, "digest", sha256_file)
    if tamper == "manifest":
        arm["resume"]["manifest_sha256"] = "0" * 64
    elif tamper == "source":
        arm["resume"]["source_sha256"] = "b" * 64
    elif tamper == "cursor":
        arm["resume"]["next_epoch"] += 1
    else:
        payload.write_bytes(b"tampered checkpoint bytes")

    with pytest.raises(RuntimeError, match=message):
        runner.resolve_resume(mounted, arm)


def test_both_arm_resume_cli_pairs_are_appended_inside_both_worker_phases():
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_main")
    phase_loop = next(
        node for node in ast.walk(main)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == "phase"
        and ast.unparse(node.iter) == "('preflight', 'training')"
    )
    arm_loop = next(
        node for node in ast.walk(phase_loop)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "config['arms']"
    )
    resume_branch = next(
        node for node in arm_loop.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "resumes[aid] is not None"
    )
    preflight_branch = next(
        node for node in arm_loop.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "phase == 'preflight'"
    )
    resume_cli = resume_branch.body[0]
    assert isinstance(resume_cli, ast.AugAssign) and ast.unparse(resume_cli.target) == "command"
    assert isinstance(resume_cli.value, ast.List)
    assert [elt.value for elt in resume_cli.value.elts if isinstance(elt, ast.Constant)] == [
        "--resume", "--resume-source-sha"
    ]
    assert resume_branch.lineno < preflight_branch.lineno

    # Execute the production resume branch for two synthetic arms so each CLI
    # pair is checked against that arm's own resolved checkpoint and source pin.
    cli_by_arm = {}
    for aid, resume_path, source_sha in (
        ("k1_pretrained_consistency", "mount/k1", "1" * 64),
        ("gptrans_g1_bond_local_ema999", "mount/gp", "2" * 64),
    ):
        command = ["python", "--arm", aid]
        arm = {"resume": {"source_sha256": source_sha}}
        resumes = {aid: resume_path}
        scope = {"aid": aid, "arm": arm, "resumes": resumes, "command": command, "str": str}
        exec(compile(ast.Module(body=[resume_branch], type_ignores=[]), str(RUNNER_PATH), "exec"), scope)
        cli_by_arm[aid] = command

    assert cli_by_arm["k1_pretrained_consistency"][-4:] == [
        "--resume", "mount/k1", "--resume-source-sha", "1" * 64
    ]
    assert cli_by_arm["gptrans_g1_bond_local_ema999"][-4:] == [
        "--resume", "mount/gp", "--resume-source-sha", "2" * 64
    ]
