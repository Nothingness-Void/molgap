"""Producer-journal adversarial metadata tests use only synthetic CPU outputs."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from molgap.family_training_outputs import SessionOutputs, _recover_commit
from molgap.training_reproducibility import sha256_file
from test_family_recovery_guards import prepared
from molgap.family_training_runner import PROFILES


def tree(root):
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}


def test_wrong_context_cannot_delete_pending_scratch(tmp_path):
    session, out, _, record, ctx = prepared(tmp_path)
    scratch = out.pending / "selected_model.pt"
    scratch.write_bytes(b"unacknowledged synthetic scratch")
    before = tree(out.root)
    try:
        with pytest.raises(ValueError, match="context"):
            SessionOutputs(out.root, replace(ctx, arm_id="wrong"), contract=tmp_path / "contract.json",
                adapter=PROFILES[session.recipe.family], trajectory_id="synthetic-trajectory",
                train_role_identity=record["train_role_identity"])
        assert tree(out.root) == before
    finally:
        session.close()


@pytest.mark.parametrize("bad", ["name", "digest", "context", "missing_backup", "changed_backup"])
def test_unbound_recovery_fails_before_any_edit(tmp_path, bad):
    session, out, _, _, ctx = prepared(tmp_path)
    try:
        selected, checkpoint = out.root / "selected_model.pt", out.root / "last_checkpoint.pt"
        selected.write_bytes(b"next-selected")
        checkpoint.write_bytes(b"previous-checkpoint")
        backup = out.pending / "selected_model.pt.previous"
        backup.write_bytes(b"previous-selected")
        record = {"format": "molgap-session-commit-v1", "context": ctx.to_dict(), "entries": [
            {"name": selected.name, "previous_sha256": sha256_file(backup), "next_sha256": sha256_file(selected)},
            {"name": checkpoint.name, "previous_sha256": sha256_file(checkpoint), "next_sha256": "e" * 64}]}
        if bad == "name":
            record["entries"][0]["name"] = "../outside.pt"
        elif bad == "digest":
            record["entries"][0]["next_sha256"] = "invalid"
        elif bad == "context":
            record["context"]["arm_id"] = "wrong"
        elif bad == "missing_backup":
            backup.unlink()
        else:
            backup.write_bytes(b"changed-backup")
        (out.root / "session_commit.json").write_text(json.dumps(record))
        before = tree(out.root)
        with pytest.raises(ValueError):
            _recover_commit(out.root, ctx)
        assert tree(out.root) == before
    finally:
        session.close()


@pytest.mark.parametrize("bad", ["journal_directory", "missing_context", "changed_contract"])
def test_unidentified_scratch_or_nonfile_journal_is_preserved(tmp_path, bad):
    session, out, reopen, _, ctx = prepared(tmp_path)
    try:
        (out.pending / "selected_model.pt").write_bytes(b"scratch")
        if bad == "journal_directory":
            (out.root / "session_commit.json").mkdir()
        elif bad == "missing_context":
            (out.root / "canonical_trace.json.context.json").unlink()
        else:
            (out.root / "training_contract.json").write_bytes(b"changed-contract")
        before = tree(out.root)
        with pytest.raises(ValueError):
            reopen()
        assert tree(out.root) == before
    finally:
        session.close()


def test_recovery_reuses_each_verified_visible_hash_without_rereading_checkpoint(tmp_path, monkeypatch):
    import molgap.family_training_outputs as owner
    session, out, _, _, ctx = prepared(tmp_path)
    try:
        entries = []
        for name in ("selected_model.pt", "last_checkpoint.pt"):
            path = out.root / name
            path.write_bytes(name.encode("ascii"))
            entries.append({"name": name, "previous_sha256": None, "next_sha256": sha256_file(path)})
        (out.root / "session_commit.json").write_text(json.dumps({"format": "molgap-session-commit-v1",
            "context": ctx.to_dict(), "entries": entries}))
        calls, native = [], owner.file_digest
        def digest(path):
            calls.append(Path(path))
            return native(path)
        monkeypatch.setattr(owner, "file_digest", digest)
        _recover_commit(out.root, ctx)
        assert calls.count(out.root / "last_checkpoint.pt") == 1
        assert calls.count(out.root / "selected_model.pt") == 1
        assert not (out.root / "session_commit.json").exists()
    finally:
        session.close()
