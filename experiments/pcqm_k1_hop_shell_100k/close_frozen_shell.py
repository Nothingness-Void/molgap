"""Translate observed shell diagnostics using the existing RML finalizer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import struct
import zipfile

from molgap.research_memory.finalize import finalize
from molgap.research_memory.paths import artifact_locator
from molgap.research_memory.schemas import validate_cost_event, validate_role_event
from molgap.training_reproducibility import atomic_json, sha256_file


def close(experiment, output, source_bundle):
    root = Path(__file__).resolve().parents[2]
    here, output, bundle = (Path(p).resolve() for p in (experiment, output, source_bundle))
    rel = here.relative_to(root).as_posix()
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    inputs, prospective, result = read(here / "inputs.json"), read(here / "rml/trajectory.json"), read(output / "result.json")
    if (result["inputs_sha256"] != sha256_file(here / "inputs.json") or
            result["prospective_sha256"] != sha256_file(here / "rml/trajectory.json") or
            any(result[k] is not False for k in ("training_executed", "optimizer_created", "gradients_computed"))):
        raise ValueError("Result/prospective identity or execution scope differs")
    complete = result["status"] == "complete"
    if not complete and not (result["status"] == "failed" and result.get("error_type") == "AttributeError"
            and result.get("error") == "'NoneType' object has no attribute 'evaluation_memory_fill'"):
        raise ValueError("Unreviewed diagnostic disposition")
    if complete and (not result.get("checks") or not all(result["checks"].values())):
        raise ValueError("Diagnostic checks failed")
    retained = [output / "result.json"]
    if complete:
        completion = read(output / "completion.json")
        if not completion["complete"] or completion["inputs_sha256"] != sha256_file(here / "inputs.json"):
            raise ValueError("Completion binding differs")
        for name, digest in completion["files"].items():
            path = output / name
            if path.parent != output or sha256_file(path) != digest:
                raise ValueError("Completed output differs")
        retained = [output / name for name in completion["files"]] + [output / "completion.json"]
    with zipfile.ZipFile(bundle) as archive:
        if set(archive.namelist()) != set(inputs["source_sha256"]):
            raise ValueError("Source custody inventory differs")
        for name, digest in inputs["source_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Executed source custody differs")
    for name in ("protocol.md", "decision.md", "terminal_decision.md", "attribution.md"):
        if not (here / name).is_file():
            raise ValueError("Interpretation missing: " + name)
    results = here / "results"
    results.mkdir(exist_ok=True)
    for source in retained:
        if source.suffix == ".json":
            destination = results / source.name
            if destination.exists() and sha256_file(destination) != sha256_file(source):
                raise ValueError("Canonical result already differs")
            if not destination.exists():
                shutil.copyfile(source, destination)
    action = prospective["actions"][0]
    tid, run, attempt = prospective["trajectory_id"], action["run_ids"][0], action["attempt_ids"][0]
    short = tid.removeprefix("TB-")
    eid = "pcqm-" + short
    decision = dict(outcome="NO_TRAIN", decision_ref=rel + "/terminal_decision.md",
                    next_allowed_actions=[], reopen_conditions=["A separately prospective authorized discriminator"])
    outcome = dict(execution_status="complete_no_training" if complete else "failed_no_training",
        artifact_status="local_hash_verified", comparison_status="consumed_role_frozen_diagnostic" if complete else "not_evaluated",
        scientific_status="NO_TRAIN", transfer_status="not_evaluated", budget_decision="within_original_cumulative_ceiling",
        full_handoff_status="not_applicable")
    for name in ("wall_seconds", "process_cpu_seconds", "device_seconds"):
        if not 0 <= result[name] <= (inputs["ceiling_seconds"] if name != "process_cpu_seconds" else float("inf")):
            raise ValueError("Invalid diagnostic cost")
    cost = validate_cost_event(dict(schema="molgap-cost-event-v1", cost_event_id="cost-" + short + "-observed",
        trajectory_id=tid, action_id=action["action_id"], run_id=run, attempt_id=attempt, platform="local-windows",
        hardware=result["runtime"]["hardware"], category="inference" if complete else "other", evidence_ref=rel + "/acceptance.json",
        measurement={k:{"value":result[v]/3600,"status":"measured"} for k,v in
                     (("wall_hours","wall_seconds"),("cpu_hours","process_cpu_seconds"),("device_hours","device_seconds"))}
                    | {"queue_hours":{"value":None,"status":"not_applicable"}}))
    recipe = read(Path(inputs["recipe"]["path"]))
    # The reviewed first attempt completed data loading, then failed while
    # entering the first forward context. No model prediction was computed.
    rows = {"train_decoded":(0,100000,("labels_read",)),
            "development_decoded":(100000,150000,("labels_read",))}
    if complete:
        rows["development"] = (100000,150000,("prediction_input","metric_computed"))
    else:
        rows["development_batch_prepared"] = (100000,100128,("prediction_input",))
    events=[]
    for role,(start,stop,accesses) in rows.items():
        digest=hashlib.sha256(b"".join(struct.pack("<q",i) for i in range(start,stop))).hexdigest()
        for access in accesses:
            events.append(validate_role_event(dict(schema="molgap-role-event-v1",role_event_id=f"role-{short}-{role}-{access}",
                trajectory_id=tid,action_id=action["action_id"],run_id=run,dataset_identity=recipe["dataset"]["manifest_sha256"],
                row_manifest_hash=digest,role_name=role,access_kind=access,selection_used=False,evidence_ref=rel+"/acceptance.json")))
    role_use = {role:"consumed" for role in rows} | {role:"untouched" for role in
        ("official_validation","test_dev","test_challenge","common","ood","p8_hard")}
    acceptance=dict(format="molgap-local-frozen-diagnostic-acceptance-v1",evidence_id=eid,run_id=run,
        outcome=outcome,trajectory_decision=decision,role_use=role_use,costs=[cost],roles=events,
        checks=result.get("checks",{}),comparison_class="CONTEXT_ONLY",analysis_ref=rel+"/results/result.json",
        execution_scope="Frozen selected checkpoint; no optimizer, gradients, model selection or training replay qualification.",
        limitation="Previously selection-used development; row uncertainty is not training stochasticity.")
    atomic_json(here/"acceptance.json",acceptance)
    paths=[here/name for name in ("protocol.md","decision.md","terminal_decision.md","attribution.md","inputs.json","acceptance.json")]
    paths += list(results.iterdir()) + retained + [bundle,Path(__file__).resolve()]
    hashes={artifact_locator(root,p):sha256_file(p) for p in paths if p.is_file()}
    evidence=dict(format="molgap-v5-evidence-envelope-v1",contract="MOLGAP-COMMON-V5-FINAL",evidence_id=eid,
        track="B",scope="pcqm-k1-hop-shell-frozen",legacy_contract="pcqm-k1-hop-shell-frozen-v1",outcome=outcome,
        authority=dict(pointers=[rel+"/"+name for name in ("protocol.md","terminal_decision.md","attribution.md","acceptance.json")]),
        role_use=role_use,migration=dict(migrated_at=datetime.now(timezone.utc).date().isoformat(),training_executed=False,
            inference_executed=False,scientific_reinterpretation=False,verification_scope="Metadata ingestion of separately observed local diagnostic."),
        observed_execution=dict(training_executed=False,inference_executed=complete,training_replay_ready=False),
        artifacts=[dict(name=str(i),locator=n,sha256=h,availability="locally_retained_hash_verified") for i,(n,h) in enumerate(hashes.items())])
    terminal=dict(format="molgap-rml-terminal-package-v1",trajectory_id=tid,run_id=run,action_id=action["action_id"],
        finalized_at=datetime.now(timezone.utc).isoformat(),acceptance_ref=rel+"/acceptance.json",artifact_hashes=hashes,
        evidence=evidence,decision=decision,costs=[cost],roles=events)
    atomic_json(here/"terminal.json",terminal)
    return finalize(root,rel+"/rml",rel+"/terminal.json")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--source-bundle",required=True)
    args=parser.parse_args()
    print(json.dumps(close(args.experiment,args.output,args.source_bundle)))
