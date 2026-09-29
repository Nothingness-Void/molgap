"""Synthetic JSON-only acceptance tests; no weights/cache/model execution."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from molgap.k1_representation_records import accept, analyze, write_reports
from molgap.representation_diagnostics import exchange_summary, spectrum_summary
from molgap.research_memory.trace import file_digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def case(tmp_path):
    plan = tmp_path / "experiments" / "audit" / "representation"
    raw = tmp_path / "raw"
    receipt = {"job_id":"1", "source_commit":"a"*40, "source_archive_sha256":"b"*64}
    write(plan/"submission.json", receipt)
    write(tmp_path/"reference.json", {"checkpoint_identity":"c"*64})
    panel = {"source_idx":list(range(500000,501024)), "selection":"synthetic"}
    write(plan/"row_manifest.json", panel)
    write(plan/"input_binding.json", {"model100_reference":"reference.json", "model500_sha256":"d"*64,
        "row_manifest_sha256":file_digest(plan/"row_manifest.json")})
    write(raw/"row_manifest.json", panel)
    flags = {"official_validation_role_read":False,"test_dev_role_read":False,"test_challenge_role_read":False}
    write(raw/"started.json", {**receipt, **flags,"training_executed":False,"model_inference_executed":True})
    write(raw/"reproduction.json", {str(s):{"max_abs_eV":0,"mae_difference_eV":0} for s in (100,500)})
    events = []
    for s in (100,500):
        events.extend([{"scale":s,"role":"original_100k" if s==100 else "unseen_500k","purpose":"prediction-reproduction","rows":50000},
                       {"scale":s,"role":"unseen_500k","purpose":"representation-panel","rows":1024}])
    write(raw/"role_events.json", events)
    h = np.eye(3,192)
    layer = exchange_summary(h, np.ones_like(h)*.01, np.ones_like(h))
    layer["bond_spectrum"] = spectrum_summary(np.eye(4,64))
    for s in (100,500):
        for c in range(8):
            rows = [{"source_idx":500000+c*128+i,"target_eV":1.,"prediction_eV":1.1 if s==100 else 1.05,
                "descriptors":{"atom_count":3.,"bond_count":2.,"conjugated_bond_fraction":0.,"ring_atom_fraction":0.,"rwse_mean":.1},
                "layers":{str(l):layer for l in (3,6,9)}} for i in range(128)]
            write(raw/f"scale{s}_chunk{c:02d}.json", rows)
    manifest = {**receipt, **flags, "complete":True,"training_executed":False,"model_inference_executed":True,
        "experiment_purpose":"NO_TRAIN","comparison_class":"CONTEXT_ONLY","physical_batch":128,"precision":"fp32",
        "panel_rows_per_scale":1024,"checkpoint_sha256":{"100":"c"*64,"500":"d"*64},
        "wall_seconds":60.,"device_hours":1/60,"peak_allocated_bytes":1000,
        "artifact_sha256":{p.name:file_digest(p) for p in raw.glob("*.json")}}
    write(raw/"completion_manifest.json", manifest)
    scheduler = {"job_id":"1","state":"COMPLETED","exit_code":"0:0","elapsed_seconds":65}
    return raw, plan, scheduler


def mutate(case, name, callback, rehash=True):
    raw, _, _ = case
    path = raw/name
    data = json.loads(path.read_text())
    callback(data)
    write(path, data)
    if rehash and name != "completion_manifest.json":
        m = json.loads((raw/"completion_manifest.json").read_text())
        m["artifact_sha256"][name] = file_digest(path)
        write(raw/"completion_manifest.json", m)


def test_complete_json_and_analysis(case):
    acceptance, rows = accept(*case)
    assert acceptance["accepted"] and not acceptance["training_replay_ready"]
    result = analyze(rows)
    assert result["panel_mae_eV"]["500"] == pytest.approx(.05)
    assert result["groups"]["all"]["error_delta_500_minus_100"]["mean"] == pytest.approx(-.05)
    assert result["layers"]["3"]["normalized_rank_after"]["atom_count_adjusted_correlation_with_error_delta"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field,value", [("complete",False),("training_executed",True),("test_dev_role_read",True),
    ("source_commit","x"*40),("physical_batch",64),("comparison_class","STRICT_CAUSAL")])
def test_manifest_attacks(case, field, value):
    mutate(case,"completion_manifest.json",lambda m:m.update({field:value}))
    with pytest.raises(ValueError):
        accept(*case)


@pytest.mark.parametrize("attack", ["rows","targets","nan","layer","path","hash","reproduction","roles"])
def test_artifact_attacks(case, attack):
    if attack == "rows":
        mutate(case,"scale100_chunk00.json",lambda r:r[0].update(source_idx=500001))
    elif attack == "targets":
        mutate(case,"scale500_chunk00.json",lambda r:r[0].update(target_eV=2.))
    elif attack == "nan":
        mutate(case,"scale100_chunk00.json",lambda r:r[0].update(prediction_eV=float("nan")))
    elif attack == "layer":
        mutate(case,"scale100_chunk00.json",lambda r:r[0]["layers"].pop("6"))
    elif attack == "path":
        mutate(case,"completion_manifest.json",lambda m:m["artifact_sha256"].update({"../secret":"a"*64}))
    elif attack == "hash":
        mutate(case,"scale100_chunk00.json",lambda r:r[0].update(target_eV=2.),False)
    elif attack == "reproduction":
        mutate(case,"reproduction.json",lambda r:r["500"].update(max_abs_eV=.1))
    else:
        mutate(case,"role_events.json",lambda r:r.pop())
    with pytest.raises(ValueError):
        accept(*case)


def test_wrong_scheduler_and_no_overwrite(case, tmp_path):
    raw, plan, scheduler = case
    with pytest.raises(ValueError):
        accept(raw,plan,{**scheduler,"state":"RUNNING"})
    path = tmp_path/"scheduler.json"
    write(path,scheduler)
    output = tmp_path/"accepted"
    write_reports(raw,plan,path,output)
    with pytest.raises(ValueError):
        write_reports(raw,plan,path,output)
