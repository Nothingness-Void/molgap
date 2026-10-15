from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from cross_scale_fixtures import cohorts, plan_dict, sha
from molgap.cross_scale_plan import DraftComparison
from molgap.cross_scale_inputs import validate_nested_cohorts, verify_sidecar, validate_graph_fixture
from molgap.common.wire import canonical_fingerprint


def test_draft_is_typed_stable_and_never_compiles_to_executable_spec():
    p = DraftComparison.from_dict(plan_dict())
    assert p.identity == DraftComparison.from_dict(plan_dict()).identity
    with pytest.raises(ValueError, match="DRAFT_ONLY"):
        p.execution_request()
    from molgap.experiment_spec import ExperimentSpec
    with pytest.raises(ValueError):
        ExperimentSpec(plan_dict())


@pytest.mark.parametrize("change", ["unknown", "hash", "role", "target", "unit", "protected", "status", "family", "revision",
                                  "feature", "approval", "axis", "horizon", "transform", "score_stop", "stqs", "futility", "seed", "snapshot", "budget", "nested_unknown"])
def test_draft_rejects_confounds_or_authority_escalation(change):
    p = plan_dict()
    if change == "unknown": p["run_now"] = True
    elif change == "hash": del p["train100k"]["source_index_sha256"]
    elif change == "role": p["development"]["role_identity"] = p["train500k"]["role_identity"]
    elif change == "target": p["train100k"]["target"] = "homo"
    elif change == "unit": p["development"]["unit"] = "Hartree"
    elif change == "protected": p["development"]["protected"] = True
    elif change == "status": p["status"] = "READY"
    elif change == "family": p["models"][0]["family"] = "ppgt_not_implemented"
    elif change == "revision": p["models"][0]["revision"] = "999"
    elif change == "feature": p["models"][0]["feature_format"] = "RWSE-is-RRWP-v999"
    elif change == "approval": p["models"][0]["feature_approval_sha256"] = None
    elif change == "axis": p["recipes"][1]["schedule_axis"] = "complete_pass"
    elif change == "horizon": p["recipes"][0]["schedule_horizon_steps"] = 31240
    elif change == "transform": p["recipes"][0]["transform_train_source_sha256"] = sha("500k extra targets")
    elif change == "score_stop": p["stop_rules"]["quality_stop_on_100k_score"] = True
    elif change == "stqs": p["stop_rules"]["automatic_stqs"] = True
    elif change == "futility": p["stop_rules"]["automatic_futility"] = True
    elif change == "seed": p["seeds"] = [42,42]
    elif change == "snapshot": p["milestones"] = [10,5]
    elif change == "budget": p["native_budget"]["unit"] = "GPU-equiv-hours"
    else: p["train100k"]["unknown"] = "field"
    with pytest.raises((ValueError, TypeError)):
        DraftComparison.from_dict(p)


def test_nested_memberships_and_disjoint_common_development():
    c, members = cohorts()
    result = validate_nested_cohorts(*c, memberships=members)
    assert result["nested"] and not result["independent_holdout"]


@pytest.mark.parametrize("change", ["reorder", "overlap", "target", "missing", "not_nested", "official_boundary"])
def test_membership_and_target_tamper_fail(change):
    from molgap.cross_scale_inputs import index_digest, target_digest
    c, members = cohorts()
    if change == "reorder": members[c[0].name]["source_indices"].reverse()
    elif change == "target": members[c[0].name]["targets"][0] += 1
    elif change == "missing": members[c[0].name]["targets"].pop()
    elif change == "official_boundary": c[2]=replace(c[2],official_train_rows=512)
    elif change == "not_nested":
        v = members[c[0].name]; v["source_indices"] = list(range(1000,1256))
        c[0] = replace(c[0], source_index_sha256=index_digest(v["source_indices"]))
    else:
        v = members[c[2].name]; v["source_indices"] = list(range(256,260)); v["targets"] = [i/17 for i in v["source_indices"]]
        c[2] = replace(c[2], source_index_sha256=index_digest(v["source_indices"]), target_sha256=target_digest(v["targets"]))
    with pytest.raises(ValueError):
        validate_nested_cohorts(*c, memberships=members)


def sidecar(path):
    from molgap.cross_scale_inputs import index_digest
    from molgap.research_memory.trace import file_digest
    return {"path": path.name, "sha256": file_digest(path), "format": "edge-neighbor-mask-v1",
            "approval_sha256": sha("CPU-only"), "source_sha256": index_digest([1,2]),
            "topology_sha256": sha("topology"), "fields": ["edge_neighbor_mask"], "pure_2d": True,
            "edge_convention": "bidirected", "source_indices": [1,2]}


def test_sidecar_remount_identity_not_absolute_path(tmp_path):
    a, b = tmp_path/"a", tmp_path/"b"
    a.mkdir(); b.mkdir()
    for root in (a,b): (root/"mask.bin").write_bytes(b"synthetic bytes")
    binding = sidecar(a/"mask.bin")
    reports = [verify_sidecar(root, binding, expected_sources=[1,2], expected_topology_sha256=sha("topology"), max_bytes=100) for root in (a,b)]
    assert reports[0] == reports[1]
    (b/"mask.bin").write_bytes(b"tamper")
    with pytest.raises(ValueError, match="bytes"):
        verify_sidecar(b, binding, expected_sources=[1,2], expected_topology_sha256=sha("topology"), max_bytes=100)


@pytest.mark.parametrize("change", ["format", "3d", "order", "topology", "size", "traversal", "missing"])
def test_sidecar_identity_and_geometry_boundary(tmp_path, change):
    (tmp_path/"mask.bin").write_bytes(b"synthetic")
    binding = sidecar(tmp_path/"mask.bin")
    if change == "format": binding["format"] = "rrwp-v999"
    elif change == "3d": binding["fields"].append("pos")
    elif change == "order": binding["source_indices"].reverse()
    elif change == "topology": binding["topology_sha256"] = sha("wrong")
    elif change == "traversal": binding["path"] = "../mask.bin"
    elif change == "missing": del binding["approval_sha256"]
    with pytest.raises(ValueError):
        verify_sidecar(tmp_path, binding, expected_sources=[1,2], expected_topology_sha256=sha("topology"), max_bytes=1 if change == "size" else 100)


def test_graph_masks_batch_isolation_padding_and_permutation():
    edge = torch.tensor([[0,1,3,4],[1,0,4,3]])
    batch = torch.tensor([0,0,0,1,1,1]); valid = torch.tensor([1,1,0,1,1,0],dtype=torch.bool)
    mask = valid[:,None] & valid[None,:] & (batch[:,None] == batch[None,:])
    report = validate_graph_fixture(edge,batch,valid,attention_mask=mask)
    perm = torch.tensor([4,0,5,1,3,2]); inv = torch.argsort(perm)
    assert validate_graph_fixture(inv[edge], batch[perm],valid[perm],attention_mask=mask[perm][:,perm]) == report
    for i,j in ((0,2),(0,3)):
        bad=mask.clone(); bad[i,j]=True
        with pytest.raises(ValueError,match="padding or different"):
            validate_graph_fixture(edge,batch,valid,attention_mask=bad)
    with pytest.raises(ValueError,match="reverse"):
        validate_graph_fixture(edge[:,:1],batch,valid)


def test_wrong_reference_or_prediction_role_blocks_draft_review(tmp_path):
    import json
    from molgap.cross_scale_plan import review_draft_inputs
    from molgap.research_memory.trace import file_digest
    from test_comparison_readiness import _reference_bundle
    c,members=cohorts(); ref=_reference_bundle(); p=plan_dict()
    ref["prediction_manifest"].update(source_idx_sha256=c[2].source_index_sha256,
        target_sha256=c[2].target_sha256,row_count=c[2].rows,unique_source_idx=c[2].rows,
        evaluation_role_identity=c[2].role_identity)
    ref["comparison_identity"]["evaluation_role_identity"]=c[2].role_identity
    path=tmp_path/"reference.json"; path.write_text(json.dumps(ref))
    p["reference_id"]=ref["reference_id"]; p["reference_bundle_sha256"]=file_digest(path)
    binding={"path":path.name,"sha256":file_digest(path)}
    result=review_draft_inputs(tmp_path,DraftComparison.from_dict(p),memberships=members,reference_binding=binding)
    assert result["status"]=="DRAFT_REVIEWED_ONLY" and not result["training_authorized"]
    p["reference_id"]="wrong-reference"
    with pytest.raises(ValueError,match="Wrong accepted"):
        review_draft_inputs(tmp_path,DraftComparison.from_dict(p),memberships=members,reference_binding=binding)
    p["reference_id"]=ref["reference_id"]
    with pytest.raises(ValueError,match="SHA"):
        review_draft_inputs(tmp_path,DraftComparison.from_dict(p),memberships=members,
                            reference_binding={"path":path.name,"sha256":sha("wrong")})
