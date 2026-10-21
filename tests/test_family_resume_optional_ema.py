"""Wire/owner resume compatibility; synthetic CPU only, no scientific inputs."""
from dataclasses import replace

import pytest
import torch

from molgap.family_training_loaders import FamilyLoaders
from molgap.family_training_outputs import SessionOutputs
from molgap.family_training_runner import PROFILES
from molgap.family_training_session import FamilyTrainingSession
from molgap.gptrans_execution_tuning import ExecutionTuning
from test_family_recovery_guards import bind
from test_family_training_runner import context, contract_file
from test_family_training_session import SyntheticKernel, events, make_session


@pytest.mark.parametrize(
    "ema_decay,corruption,error",
    [
        (None, None, None),
        (.999, None, None),
        (.999, "ema", "EMA policy"),
        (None, "scheduler", "Incomplete family resume state"),
        (None, "unknown", "Incomplete family resume state"),
    ],
)
def test_resume_hook_normalizes_only_disabled_wire_ema(
    tmp_path, monkeypatch, ema_decay, corruption, error
):
    import molgap.family_training_runner as owner

    template = make_session(
        "gptrans_t", development=True, schedule_kind="cosine_step", events=events()
    )
    recipe = replace(
        template.recipe, ema_decay=ema_decay,
        selection="best-development-live" if ema_decay is None else "best-development-ema",
    )
    data = template.loaders.data
    template.close()
    session = FamilyTrainingSession(
        SyntheticKernel(recipe),
        FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0)),
        checkpoint=lambda *args: None,
        selected=lambda *args: None,
        pass_finished=lambda *args: None,
    )
    try:
        path, record = contract_file(tmp_path, session)
        ctx = context("gptrans_t", path)
        out = SessionOutputs(
            tmp_path / "output", ctx, contract=path, adapter=PROFILES["gptrans_t"],
            trajectory_id="synthetic-trajectory",
            train_role_identity=record["train_role_identity"],
        )
        bind(session, out)
        session.run(should_pause=lambda: session.progress.optimizer_step >= 2)
        checkpoint = out.root / "last_checkpoint.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert ("ema" in saved) is (ema_decay is not None)
        if corruption:
            if corruption == "unknown":
                saved["unknown"] = True
            else:
                saved.pop(corruption)
            torch.save(saved, checkpoint)
        # This unit isolates the wire-to-session boundary. Real retained bundles
        # separately exercise the original immutable Spec/source/recipe gates.
        monkeypatch.setattr(owner, "read_recipe", lambda _: (record, recipe))
        monkeypatch.setattr(owner, "validate_screen_recipe", lambda *_: None)
        kwargs = dict(
            spec=object(), arm_id="reference", manifest={}, bundle_root=tmp_path,
            artifacts={"checkpoint": checkpoint, "trace": out.root / "canonical_trace.json"},
            sidecars={}, trajectory={"trajectory_id": "synthetic-trajectory"},
            context=ctx.to_dict(),
        )
        if error:
            with pytest.raises(ValueError, match=error):
                owner.validate_screen_resume(**kwargs)
        else:
            result = owner.validate_screen_resume(**kwargs)
            assert result["status"] == "SHARED_RESUME_VERIFIED"
            assert result["optimizer_steps"] == 2
    finally:
        session.close()
