"""CPU synthetic end-to-end using the existing family session test owner."""
from copy import deepcopy
from dataclasses import replace
import inspect
import json

import pytest
import torch

from cross_scale_fixtures import cohorts, plan_dict, curve, identity, sha
from test_family_training_session import make_session, events, assert_equal
from molgap.cross_scale_plan import DraftComparison
from molgap.cross_scale_inputs import validate_nested_cohorts, audit_fixed_manifests
from molgap.cross_scale_checkpoint import (
    capture_checkpoint, evaluate_readonly, EvaluationPermit, ExactStepSnapshotSink, state_digest,
)
from molgap.cross_scale_analysis import align_curves, interpret_run
from molgap.common.wire import canonical_fingerprint
from molgap.experiment_family_workflow import tensor_digest
from molgap.training_reproducibility import atomic_torch_save
from molgap.v4_runtime import torch_load_compat


def synthetic_session(log):
    session=make_session("gptrans_t",development=False,schedule_kind="cosine_step",events=log)
    recipe=replace(session.recipe,checkpoint_every=1)
    session.recipe=session.kernel.recipe=session.loaders.recipe=recipe
    # This adapts the existing elementary fixture's forward ABI only; the real
    # shared loop, optimizer, sampler, scheduler, EMA and recovery stay unchanged.
    kernel=session.kernel
    kernel.forward=lambda batch,model=None: (kernel.model if model is None else model)(
        batch.source_idx.float().reshape(-1,1)/257.).reshape(-1)
    return session


def permit_case():
    source=torch.arange(600,604); target=torch.arange(4).float()
    row_sha=tensor_digest(source,role="source_idx")
    authority={"role_identity":"internal_development_synthetic_only","row_sha256":row_sha,
        "operations":["prediction_input","labels_read","metric_computed"],"protected":False,"scope":"CPU_SYNTHETIC_ONLY"}
    history={"role_identity":authority["role_identity"],"row_sha256":row_sha,"usage":"selection-used"}
    permit=EvaluationPermit(authority["role_identity"],row_sha,tensor_digest(target,role="target"),
        canonical_fingerprint(authority),canonical_fingerprint(history),4,600)
    from types import SimpleNamespace
    batch=SimpleNamespace(source_idx=source,y=target)
    return permit,authority,history,batch


def test_synthetic_plan_snapshot_interrupt_resume_readonly_curve_no_remote_no_rml(tmp_path):
    plan=DraftComparison.from_dict(plan_dict())
    c,members=cohorts(); assert validate_nested_cohorts(*c,memberships=members)["nested"]
    baseline=synthetic_session(events()); baseline.run(); expected=deepcopy(baseline.state_dict())
    first=synthetic_session(events())
    original=first.checkpoint
    snapshots=ExactStepSnapshotSink(first.recipe,[1,3,5],lambda progress,state:
        atomic_torch_save(tmp_path/f"step_{progress.optimizer_step}.pt",state))
    def checkpoint(progress,state):
        original(progress,state); snapshots(progress,state)
    first.checkpoint=checkpoint
    paused=first.run(should_pause=lambda: True)
    assert (paused.optimizer_step,paused.pass_index,paused.next_batch)==(1,0,1)
    retained=capture_checkpoint(first)
    before=state_digest(first.state_dict()); modes=[m.training for m in first.kernel.model.modules()]
    permit,authority,history,batch=permit_case()
    result=evaluate_readonly(first,retained,permit=permit,authority=authority,role_history=history,
                             batches_factory=lambda:[batch],weights="ema")
    assert result["optimizer_step"]==1 and result["complete_passes"]==0 and result["next_batch"]==1
    assert not result["selection_updated"] and not result["rml_written"]
    assert state_digest(first.state_dict())==before
    assert modes==[m.training for m in first.kernel.model.modules()]
    # A separate process deserializes its own tensor storages. CPU AdamW may
    # retain moment references; do not alias the in-memory evidence fixture.
    resumed=synthetic_session(events()); resumed.load_state_dict(deepcopy(retained))
    original2=resumed.checkpoint
    def checkpoint2(progress,state):
        original2(progress,state); snapshots(progress,state)
    resumed.checkpoint=checkpoint2; resumed.run()
    assert_equal(expected,resumed.state_dict())
    assert state_digest(expected)==state_digest(resumed.state_dict())
    assert snapshots.report()["missing_steps"]==[]
    assert state_digest(torch_load_compat(tmp_path/"step_1.pt",map_location="cpu",weights_only=True))==state_digest(retained)
    panel=align_curves(curve([.2,.15,.1]),curve([.22,.14,.09]),left_identity=identity(),right_identity=identity())
    assert panel["status"]=="MATCHED_PREFIX"
    assert not (tmp_path/"research_memory").exists()
    assert plan.status=="DRAFT_ONLY"


@pytest.mark.parametrize("tamper",["authority","history","protected","target"])
def test_forbidden_or_tampered_roles_fail_before_opening_tensors(tamper):
    session=synthetic_session(events()); session.run(should_pause=lambda:True)
    retained=capture_checkpoint(session)
    permit,authority,history,batch=permit_case(); opened=[]
    if tamper=="authority": authority["operations"].append("selection_used")
    elif tamper=="history": history["usage"]="untouched"
    elif tamper=="protected":
        with pytest.raises(ValueError,match="ROLE_BLOCKED"):
            replace(permit,role="official_validation",protected=True)
        return
    else: authority["row_sha256"]=sha("wrong")
    with pytest.raises(ValueError):
        evaluate_readonly(session,retained,permit=permit,authority=authority,role_history=history,
                          batches_factory=lambda:opened.append(True) or [batch])
    assert opened==[]


def test_evaluation_exception_restores_rng_modes_and_all_training_state():
    session=synthetic_session(events()); session.run(should_pause=lambda:True)
    state=capture_checkpoint(session); before=state_digest(session.state_dict())
    permit,authority,history,batch=permit_case()
    def failing():
        import random
        import numpy as np
        random.random(); np.random.rand()
        torch.rand(7)
        raise RuntimeError("loader interrupted")
    with pytest.raises(RuntimeError,match="loader interrupted"):
        evaluate_readonly(session,state,permit=permit,authority=authority,role_history=history,batches_factory=failing)
    assert state_digest(session.state_dict())==before


def test_readonly_bn_buffers_optimizer_and_rng_are_unchanged():
    from molgap.family_training_kernel import make_family_optimizer, ContractSchedule
    from molgap.gptrans_execution_hotpath import BoundForeachEMA
    session=synthetic_session(events()); kernel=session.kernel
    kernel.model=torch.nn.Sequential(torch.nn.Linear(1,2),torch.nn.BatchNorm1d(2),
        torch.nn.Dropout(.2),torch.nn.Linear(2,1))
    kernel.optimizer=make_family_optimizer(session.recipe,kernel.model.parameters())
    session.schedule=ContractSchedule(kernel)
    kernel.ema=BoundForeachEMA(kernel.model,session.recipe.ema_decay)
    session.run(should_pause=lambda:True)
    checkpoint=capture_checkpoint(session); before=state_digest(session.state_dict())
    assert "1.running_mean" in checkpoint["model"]
    permit,authority,history,batch=permit_case()
    evaluate_readonly(session,checkpoint,permit=permit,authority=authority,role_history=history,
                      batches_factory=lambda:[batch],weights="live")
    assert state_digest(session.state_dict())==before
    assert torch.equal(checkpoint["model"]["1.running_mean"],kernel.model[1].running_mean)


@pytest.mark.parametrize("role",["official_validation","test_dev","internal_development_ood","internal_development_p8_hard","internal_development_external"])
def test_role_name_cannot_hide_protected_access_behind_a_false_flag(role):
    permit,*_=permit_case()
    with pytest.raises(ValueError,match="ROLE_BLOCKED"):
        replace(permit,role_identity=role,protected=False)


def test_snapshot_publication_failure_is_not_acknowledged(tmp_path):
    session=synthetic_session(events())
    def failed(*args): raise OSError("disk failure")
    sink=ExactStepSnapshotSink(session.recipe,[1],failed)
    session.checkpoint=sink
    with pytest.raises(OSError,match="disk failure"): session.run(should_pause=lambda:True)
    assert sink.report()["status"]=="INCOMPLETE" and session.progress.status=="persistence_failed"


def test_native_real_role_or_unavailable_clean_state_is_not_implicitly_enabled():
    session=synthetic_session(events()); session.run(should_pause=lambda:True)
    state=capture_checkpoint(session); permit,authority,history,batch=permit_case()
    with pytest.raises(ValueError,match="Clean-BN"):
        evaluate_readonly(session,state,permit=permit,authority=authority,role_history=history,
                          batches_factory=lambda:[batch],weights="clean-ema")
    authority["scope"]="NATIVE_GPU"
    permit=replace(permit,authority_sha256=canonical_fingerprint(authority))
    with pytest.raises(ValueError,match="no CPU"):
        evaluate_readonly(session,state,permit=permit,authority=authority,role_history=history,
                          batches_factory=lambda:[batch])


def test_retained_fixed_manifests_are_role_blocked_without_opening_graphs():
    from pathlib import Path
    from molgap.research_memory.trace import file_digest
    root=Path(__file__).resolve().parents[1]/"platforms/_records/ims/pcqm_fixed_datasets_v1"
    paths=[root/"ogb-train-100k.manifest.json",root/"ogb-train-500k-scnet-v1.manifest.json"]
    if not all(p.is_file() for p in paths):
        pytest.skip("Host fixed metadata not installed; no scientific cache substitute")
    report=audit_fixed_manifests(*[(json.loads(p.read_text()),file_digest(p)) for p in paths])
    assert report["status"]=="ROLE_BLOCKED" and report["training_shards_nested_bytewise"]
    assert report["small_development_overlaps_large_train"] and not report["graph_tensors_opened"]
    assert report["target_tensor_sha256"] is None


def test_analysis_imports_no_training_policy_or_remote_or_rml_writer():
    import ast
    import molgap.cross_scale_analysis as analysis
    import molgap.cross_scale_backtest as backtest
    for module in (analysis,backtest):
        tree=ast.parse(inspect.getsource(module))
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any(any(part in (name or "") for part in ("scale_transfer_diagnostic","futility_gate","research_memory.plan",
            "experiment_prospective","server_control","kaggle","research_memory.terminal")) for name in imports)


def test_future_model_fidelity_stub_neither_registers_nor_claims_scientific_failure():
    from molgap.model_fidelity import ModelFidelityEvidence
    fields={key:sha(key) for key in ModelFidelityEvidence.__dataclass_fields__ if key.endswith("sha256")}
    values={**fields,"family":"future-ppgt-stub","revision":"draft","paper_version":"doi:synthetic/fixture-v1",
        "upstream_commit":"1"*40,"license":"synthetic-only","parameters":3,"atol":1e-6,"rtol":1e-5,
        "max_absolute_error":.1,"max_relative_error":.1,"pure_2d":True,"upstream_executed":False,
        "permutation_invariant":True,"batch_isolated":True,"backward_finite":True}
    result=ModelFidelityEvidence.from_dict(values).assessment()
    assert result["status"]=="IMPLEMENTATION_UNVERIFIED" and not result["family_registered"]
    assert result["scientific_status"]=="NOT_EVALUATED" and not result["training_authorized"]
