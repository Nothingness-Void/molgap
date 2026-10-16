"""Small real RDKit/SQLite/PyG fixtures, never actual PCQM labels or models."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest
import torch
from rdkit import rdBase
from torch_geometric.data import Data, InMemoryDataset

from molgap.pcqm_unified_roles import (
    ALGORITHM, FORMAT, MOLECULE_RULE, canonical_molecule, inventory_rows,
    choose_roles, validate_rule, validate_role_members, prepare_unified_roles,
    accept_unified_roles, authorize_role, verify_mirror_identity,
)
from molgap.pcqm_fixed_datasets import sha256_file
from molgap.research_memory.trace import json_bytes
from cross_scale_fixtures import sha


def rule():
    return {"format": FORMAT, "algorithm": ALGORITHM, "molecule_rule": MOLECULE_RULE,
        "rdkit_version": rdBase.rdkitVersion, "seed": 42, "official_train_rows": 30,
        "train_sizes": [2,4,6], "common_dev_rows": 2, "independent_audit_rows": 2,
        "excluded_ranges": [[6,8]], "source_manifest_sha256": sha("source"),
        "source_acceptance_sha256": sha("acceptance"), "role_inventory_sha256": sha("history"),
        "generator_sha256": sha("generator"), "source_commit": "f"*40,
        "scope": "CPU_SYNTHETIC_ONLY"}


def database(r=None, rows=None):
    r = r or rule(); db = sqlite3.connect(":memory:")
    inventory_rows(db, rows or [(i,"C"*(i+1),float(i)/10) for i in range(30)],r)
    return db


def test_nested_disjoint_roles_duplicate_rule_and_old_full_applicability():
    db = database(); roles=choose_roles(db,rule())
    checks=validate_role_members(db,roles,rule())
    assert checks["cross_role_exact_molecule_intersection"] == 0
    assert checks["old_full_independent_audit_applicability"].startswith("NOT_APPLICABLE")
    assert set(roles["train_1m"]) <= set(roles["full_minus_holdout"])
    assert not set(roles["common_dev"]) & set(roles["independent_audit"])
    assert len(roles["full_minus_holdout"]) == 26


def test_exact_molecule_groups_excluded_and_canonical_equivalence():
    assert canonical_molecule("OCC") == canonical_molecule("CCO")
    assert canonical_molecule("[CH3:2][CH3:1]") == canonical_molecule("CC")
    rows=[(i,"C"*(i+1),float(i)/10) for i in range(30)]
    rows[29]=(29,"CC",2.9)  # duplicate of preserved training member1
    db=database(rows=rows); roles=choose_roles(db,rule())
    assert 29 not in roles["common_dev"]+roles["independent_audit"]
    # If a duplicate group is held out, every duplicate must leave Full.
    rows[28]=(28,rows[27][1],2.8)
    db=database(rows=rows); roles=choose_roles(db,rule())
    checks=validate_role_members(db,roles,rule())
    assert checks["train_audit_intersection"] == 0


def test_sampling_determinism_seed_and_source_identity_change():
    a=choose_roles(database(),rule()); b=choose_roles(database(),rule())
    assert a == b
    other=rule(); other["seed"]=43
    assert choose_roles(database(other),other) != a
    assert validate_rule(other) != validate_rule(rule())
    other=rule(); other["source_manifest_sha256"]=sha("changed source")
    assert validate_rule(other) != validate_rule(rule())


@pytest.mark.parametrize("tamper", ["order", "train", "dev_overlap", "audit_overlap", "full_contains_audit"])
def test_role_membership_tamper(tamper):
    db=database(); roles=choose_roles(db,rule())
    if tamper=="order": roles["common_dev"].reverse()
    elif tamper=="train": roles["train_500k"][-1]=10
    elif tamper=="dev_overlap": roles["common_dev"][0]=0;roles["common_dev"].sort()
    elif tamper=="audit_overlap": roles["independent_audit"]=list(roles["common_dev"])
    else: roles["full_minus_holdout"].extend(roles["independent_audit"]);roles["full_minus_holdout"].sort()
    with pytest.raises(ValueError): validate_role_members(db,roles,rule())


def source_fixture(root):
    fixed=root/"fixed";fixed.mkdir()
    graphs=[]
    for i in range(30):
        graphs.append(Data(x=torch.ones((i+1,9),dtype=torch.long),edge_index=torch.empty((2,0),dtype=torch.long),
            edge_attr=torch.empty((0,3),dtype=torch.long),random_walk_pe=torch.zeros((i+1,16)),
            y=torch.tensor([float(i)/10]),source_idx=torch.tensor([i])))
    torch.save(InMemoryDataset.collate(graphs),fixed/"train.pt")
    with gzip.open(fixed/"rows.csv.gz","wt") as f:
        f.write("idx,smiles,homolumogap\n")
        for i in range(30): f.write(f"{i},{'C'*(i+1)},{float(i)/10}\n")
        f.write("30,DO_NOT_PARSE_OFFICIAL_VALIDATION,not-a-label\n")
    manifest={"status":"complete","store_assets":{"source":[{"file":"rows.csv.gz","sha256":sha256_file(fixed/"rows.csv.gz")}],
        "topology":[{"file":"train.pt","sha256":sha256_file(fixed/"train.pt"),"role":"train","rows":30,
                     "source_idx_min":0,"source_idx_max":29}]}}
    (fixed/"manifest.json").write_bytes(json_bytes(manifest))
    (fixed/"acceptance.json").write_bytes(json_bytes({"status":"accepted","verify_content":True}))
    r=rule();r.update(source_manifest_sha256=sha256_file(fixed/"manifest.json"),source_acceptance_sha256=sha256_file(fixed/"acceptance.json"))
    return fixed,r


def test_full_cpu_preparation_resume_content_and_no_validation_read(tmp_path):
    fixed,r=source_fixture(tmp_path)
    original={p.name:sha256_file(p) for p in fixed.iterdir()}
    out=tmp_path/"unified"
    report=prepare_unified_roles(boundary=tmp_path,fixed_root=fixed,output_root=out,rule=r)
    assert report["status"]=="SYNTHETIC_DATA_ROLES_VERIFIED"
    assert report["roles"]["common_dev"]["rows"]==2
    assert not report["official_validation_read"] and not report["graphs_rebuilt"]
    assert accept_unified_roles(out,rule=r)["nested"]
    assert {p.name:sha256_file(p) for p in fixed.iterdir()}==original
    # Resume after the final report is already committed must return the same
    # acceptance rather than republish new timing bytes over an immutable file.
    again=prepare_unified_roles(boundary=tmp_path,fixed_root=fixed,output_root=out,rule=r)
    assert again==report


@pytest.mark.parametrize("field", ["members", "labels", "topology"])
def test_derived_content_tampering_is_rejected(tmp_path,field):
    fixed,r=source_fixture(tmp_path);out=tmp_path/"unified"
    prepare_unified_roles(boundary=tmp_path,fixed_root=fixed,output_root=out,rule=r)
    if field=="members":
        p=out/"members/common_dev.i64le";p.write_bytes(p.read_bytes()[::-1])
    else:
        m=json.loads((out/"roles/common_dev.json").read_text());p=out/m["graph_chunks"][0]["file"]
        data,slices=torch.load(p,weights_only=False)
        if field=="labels": data.y[0]+=1
        else: data.x[0,0]+=1
        torch.save((data,slices),p)
    with pytest.raises(ValueError,match="content|identity|SHA"):
        accept_unified_roles(out,rule=r)


def test_audit_never_available_to_training_or_selection_and_mirrors_are_private_identity_only():
    audit={"role":"independent_audit","status":"complete"}
    for op in ("training_membership","prediction_input","labels_read","metric_computed","selection_used"):
        with pytest.raises(ValueError,match="ROLE_BLOCKED"):
            authorize_role(audit,operation=op,candidate_frozen=True,selection_frozen=True)
    with pytest.raises(TypeError): authorize_role(audit,operation="labels_read",candidate_frozen="False")
    master={k:"same" for k in ("format","version_identity","rows","source_index_sha256","ordering","logical_content_sha256","feature_schema")}
    master.update(role="common_dev",pure_2d=True,protected=False)
    assert verify_mirror_identity(master,deepcopy(master))["status"]=="LOGICAL_IDENTITY_MATCH_ONLY"
    mirror=deepcopy(master);mirror["source_index_sha256"]=sha("changed seed")
    with pytest.raises(ValueError):verify_mirror_identity(master,mirror)


def test_interrupted_chunk_preparation_resumes_without_recounting_rows(tmp_path,monkeypatch):
    import molgap.pcqm_unified_roles as owner
    fixed,r=source_fixture(tmp_path);out=tmp_path/"unified"
    original=owner._derive_graph_chunks
    def crash_after_chunk(*args,**kwargs):
        original(*args,**kwargs)
        raise OSError("simulated interruption after durable graph chunk")
    with monkeypatch.context() as patch:
        patch.setattr(owner,"_derive_graph_chunks",crash_after_chunk)
        with pytest.raises(OSError,match="interruption"):
            prepare_unified_roles(boundary=tmp_path,fixed_root=fixed,output_root=out,rule=r)
    assert not (out/"acceptance.json").exists()
    result=prepare_unified_roles(boundary=tmp_path,fixed_root=fixed,output_root=out,rule=r)
    assert result["roles"]["full_minus_holdout"]["rows"]==26
    assert accept_unified_roles(out,rule=r)["nested"]


def test_unknown_geometry_field_is_rejected_before_selected_payload_exists(tmp_path):
    fixed,r=source_fixture(tmp_path)
    data,slices=torch.load(fixed/"train.pt",weights_only=False)
    data.pos=torch.zeros((data.num_nodes,3));torch.save((data,slices),fixed/"train.pt")
    manifest=json.loads((fixed/"manifest.json").read_text())
    manifest["store_assets"]["topology"][0]["sha256"]=sha256_file(fixed/"train.pt")
    (fixed/"manifest.json").write_bytes(json_bytes(manifest));r["source_manifest_sha256"]=sha256_file(fixed/"manifest.json")
    with pytest.raises(ValueError,match="Non-2D"):
        prepare_unified_roles(boundary=tmp_path,fixed_root=fixed,output_root=tmp_path/"unified",rule=r)
    assert not (tmp_path/"unified/acceptance.json").exists()


def test_incomplete_candidate_pool_never_relaxes_rule():
    r=rule();r["common_dev_rows"]=20;r["independent_audit_rows"]=20
    with pytest.raises(ValueError,match="insufficient"):
        choose_roles(database(r),r)
