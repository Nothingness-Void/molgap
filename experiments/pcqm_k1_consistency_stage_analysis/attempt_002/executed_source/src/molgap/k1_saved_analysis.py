"""Bounded arithmetic on retained K1 traces/predictions, never model execution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

KINDS = {"trace_pair": "pcqm_k1_consistency_stage_analysis/attempt_002",
         "bn_rows": "pcqm_k1_bn_row_attribution"}
ARMS = ("k1_pretrained_mean2", "k1_pretrained_consistency")


def trace_pair(traces, manifests, inspection):
    """Require equal observed exposure; never interpolate or project an endpoint."""
    if inspection.get("mechanical_stage_pass") is not True or inspection.get("disposition") != "ACTIVE_PARTIAL_STAGE":
        raise ValueError("Accepted partial stage required")
    allowed_differences = {"arm", "loss_fingerprint", "binding_identity"}
    contracts = [manifests[a]["contract"] for a in ARMS]
    keys = set(contracts[0]) | set(contracts[1])
    if any(contracts[0].get(k) != contracts[1].get(k) for k in keys - allowed_differences):
        raise ValueError("Paired contracts differ beyond declared intervention")
    for key in ("spec_identity",):
        if contracts[0]["binding_identity"][key] != contracts[1]["binding_identity"][key]:
            raise ValueError("Paired binding identity differs")
    if contracts[0]["binding_identity"]["spec_identity"] != inspection.get("spec_identity"):
        raise ValueError("Spec identity differs from accepted inspection")
    for a in ARMS:
        if manifests[a]["source_sha256"] != inspection["source_sha256"] or not inspection["arms"][a]["stage_mechanical_pass"]:
            raise ValueError("Accepted source/arm differs")
    rows = [traces[a]["epochs"] for a in ARMS]
    if not rows[0] or len(rows[0]) != len(rows[1]):
        raise ValueError("Paired trace lengths differ")
    output, cumulative = [], [0., 0.]
    best = [math.inf, math.inf]
    for index, pair in enumerate(zip(*rows)):
        for side, row in enumerate(pair):
            expected_step = (index + 1) * contracts[side]["steps_per_epoch"]
            expected_samples = expected_step * contracts[side]["physical_batch_per_device"]
            if row["epoch"] != index or row["global_step"] != expected_step or row["sample_presentations"] != expected_samples:
                raise ValueError("Noncontiguous or mismatched observed exposure")
            fields = [row[k] for k in ("train_mae_eV", "development_mae_eV", "lr", "seconds")]
            fields += list(row["objective_components"].values())
            if not all(math.isfinite(float(x)) and x >= 0 for x in fields) or row["seconds"] <= 0:
                raise ValueError("Invalid trace metric")
            objective = row["objective_components"]
            coefficient = (0., .1)[side]
            if not math.isclose(objective["combined_loss"], objective["supervised_l1"] + coefficient * objective["disagreement"], abs_tol=1e-6):
                raise ValueError("Objective arithmetic differs")
            cumulative[side] += row["seconds"]
            best[side] = min(best[side], row["development_mae_eV"])
        if pair[0]["lr"] != pair[1]["lr"]:
            raise ValueError("Observed learning rates differ")
        output.append(dict(epoch_one_based=index + 1, optimizer_steps=pair[0]["global_step"],
            sample_presentations=pair[0]["sample_presentations"], learning_rate=pair[0]["lr"],
            control_dev_eV=pair[0]["development_mae_eV"], consistency_dev_eV=pair[1]["development_mae_eV"],
            consistency_minus_control_meV=1000 * (pair[1]["development_mae_eV"] - pair[0]["development_mae_eV"]),
            best_so_far_consistency_minus_control_meV=1000 * (best[1] - best[0]),
            control_train_stochastic_eV=pair[0]["train_mae_eV"], consistency_train_stochastic_eV=pair[1]["train_mae_eV"],
            control_supervised_normalized=pair[0]["objective_components"]["supervised_l1"],
            consistency_supervised_normalized=pair[1]["objective_components"]["supervised_l1"],
            control_disagreement_normalized_squared=pair[0]["objective_components"]["disagreement"],
            consistency_disagreement_normalized_squared=pair[1]["objective_components"]["disagreement"],
            consistency_penalty_to_supervised_ratio=.1 * pair[1]["objective_components"]["disagreement"] / max(pair[1]["objective_components"]["supervised_l1"], 1e-30),
            control_epoch_seconds=pair[0]["seconds"], consistency_epoch_seconds=pair[1]["seconds"],
            control_cumulative_epoch_window_seconds=cumulative[0], consistency_cumulative_epoch_window_seconds=cumulative[1]))
    for side, a in enumerate(ARMS):
        manifest = manifests[a]
        if manifest["next_epoch"] != len(output) or not math.isclose(best[side], manifest["best_development_mae_eV"], abs_tol=1e-9):
            raise ValueError("Manifest endpoint differs from trace")
        accepted = inspection["arms"][a]
        if accepted["optimizer_steps"] != output[-1]["optimizer_steps"] or accepted["sample_presentations"] != output[-1]["sample_presentations"]:
            raise ValueError("Accepted counters differ")
    gaps = [r["consistency_minus_control_meV"] for r in output]
    blocks = []
    for start in range(0, len(output), 5):
        part = output[start:start + 5]
        blocks.append(dict(epochs=[part[0]["epoch_one_based"], part[-1]["epoch_one_based"]],
            mean_consistency_minus_control_meV=sum(r["consistency_minus_control_meV"] for r in part) / len(part),
            consistency_better_epochs=sum(r["consistency_minus_control_meV"] < 0 for r in part)))
    return dict(curve=output, blocks=blocks, completed_epochs=len(output), contract_epochs=contracts[0]["epochs"],
        consistency_better_epochs=sum(x < 0 for x in gaps), same_step_last_difference_meV=gaps[-1],
        selected_stage_difference_meV=1000 * (best[1] - best[0]),
        epoch_window_seconds=dict(zip(ARMS, cumulative)), native_stage_cost=inspection["budget"],
        uncertainty="No seed uncertainty or epoch bootstrap: serially correlated aggregate observations",
        limitations=["Partial23/60 is not terminal truth or an early-stop calibration",
            "Both arms execute two forwards; coefficient comparison does not measure single-pass speed",
            "Train metrics are stochastic; dev is live clean inference with original BN",
            "Epoch window is not a phase/operator profile or total allocated/billed hours"])


def bn_rows(predictions, bounds=(500000, 550000)):
    import numpy as np
    import torch
    from .router import paired_bootstrap_mean
    expected = torch.arange(*bounds)
    arrays, truth = {}, None
    for name, value in predictions.items():
        ids = value["source_idx"].reshape(-1)
        target = value.get("target_eV", value.get("target")).reshape(-1)
        pred = value.get("prediction_eV", value.get("prediction")).reshape(-1)
        if ids.dtype not in (torch.int32, torch.int64) or not torch.equal(ids, expected):
            raise ValueError("Exact source membership/order differs")
        if len(target) != len(expected) or len(pred) != len(expected) or not torch.isfinite(target).all() or not torch.isfinite(pred).all():
            raise ValueError("Prediction length/finiteness differs")
        if truth is not None and not torch.equal(target, truth):
            raise ValueError("Paired targets differ")
        truth = target
        arrays[name] = pred.double().numpy()
    y = truth.double().numpy()
    errors = {n: np.abs(p - y) for n, p in arrays.items()}
    gain49 = errors["selected_raw"] - errors["selected_clean"]
    gain60 = errors["final_raw"] - errors["final_clean"]
    late = errors["final_clean"] - errors["selected_clean"]
    raw_late = errors["final_raw"] - errors["selected_raw"]
    if not np.allclose(raw_late, late + gain60 - gain49, rtol=0, atol=1e-14):
        raise ValueError("Per-row late/BN decomposition failed")
    def summary(delta):
        boot = paired_bootstrap_mean(delta, n_bootstrap=1000, seed=20261008)
        boot.pop("probability_better")
        return dict(mean_meV=1000 * boot["delta"], row_ci95_meV=[1000*x for x in boot["ci95"]],
            positive_rows_fraction=float((delta > 0).mean()), negative_rows_fraction=float((delta < 0).mean()),
            equal_rows_fraction=float((delta == 0).mean()), median_meV=float(np.median(delta) * 1000),
            positive_gross_eV=float(delta.clip(min=0).sum()), negative_gross_eV=float((-delta).clip(min=0).sum()))
    def group(indices):
        if not len(indices):
            return dict(rows=0, raw49_mae_eV=None, clean49_mae_eV=None, bn49_gain_meV=None,
                bn60_gain_meV=None, clean_late_deterioration_meV=None,
                raw_late_deterioration_meV=None, bn49_net_contribution_meV=0.)
        return dict(rows=len(indices), raw49_mae_eV=float(errors["selected_raw"][indices].mean()),
            clean49_mae_eV=float(errors["selected_clean"][indices].mean()),
            bn49_gain_meV=float(gain49[indices].mean()*1000), bn60_gain_meV=float(gain60[indices].mean()*1000),
            clean_late_deterioration_meV=float(late[indices].mean()*1000),
            raw_late_deterioration_meV=float(raw_late[indices].mean()*1000),
            bn49_net_contribution_meV=float(gain49[indices].sum()/len(y)*1000))
    order = np.lexsort((expected.numpy(), errors["selected_raw"]))
    quantiles = [0, .5, .9, .95, .99, 1.]
    strata = []
    for lo, hi in zip(quantiles[:-1], quantiles[1:]):
        indices = order[int(lo*len(y)):int(hi*len(y))]
        strata.append(dict(baseline_raw49_error_percentile=[lo, hi], **group(indices)))
    targets = []
    for chunk in np.array_split(np.lexsort((expected.numpy(), y)), 4):
        targets.append(dict(target_range_eV=[float(y[chunk].min()), float(y[chunk].max())], **group(chunk)))
    bins = []
    for chunk in np.array_split(np.lexsort((expected.numpy(), gain49)), 4):
        bins.append(dict(bn49_gain_range_meV=[float(gain49[chunk].min()*1000), float(gain49[chunk].max()*1000)], **group(chunk)))
    tails = []
    for fraction in (.01, .05, .1):
        indices = order[-max(1, math.ceil(len(y)*fraction)):]
        remaining = order[:-len(indices)]
        tails.append(dict(baseline_raw_error_top_fraction=fraction, **group(indices),
            remaining_bn49_gain_meV=float(gain49[remaining].mean()*1000) if len(remaining) else None,
            net_gain_share=float(gain49[indices].sum()/gain49.sum()) if gain49.sum() != 0 else None,
            positive_gross_gain_share=float(gain49[indices].clip(min=0).sum()/gain49.clip(min=0).sum()) if gain49.clip(min=0).sum() else None))
    def corr(a, b):
        return float(np.corrcoef(a,b)[0,1]) if a.std() > 0 and b.std() > 0 else None
    def cov(a, b):
        return float(((a-a.mean())*(b-b.mean())).mean())
    a, b, c = (errors[n] for n in ("selected_raw", "selected_clean", "final_clean"))
    coupling = dict(raw49_with_clean60=cov(a,c), minus_raw49_with_clean49=-cov(a,b),
        minus_clean49_with_clean60=-cov(b,c), shared_clean49_variance=cov(b,b))
    covariance = cov(gain49,late)
    if not math.isclose(covariance,sum(coupling.values()),rel_tol=1e-9,abs_tol=1e-14):
        raise ValueError("Shared-term covariance decomposition failed")
    report = dict(rows=len(y), mae_eV={n:float(e.mean()) for n,e in errors.items()},
        bn49_gain=summary(gain49), bn60_gain=summary(gain60), clean_late_deterioration=summary(late),
        raw_late_deterioration=summary(raw_late), baseline_error_strata=strata, baseline_error_tails=tails,
        target_quartiles=targets, bn49_gain_quartiles=bins,
        bn49_gain_vs_clean_late_pearson=corr(gain49, late),
        bn49_gain_vs_bn60_gain_pearson=corr(gain49, gain60),
        shared_term_covariance_eV_squared=dict(observed=covariance,terms=coupling,
            interpretation="Algebraic coupling decomposition, not a causal effect or calibrated null test"),
        bn49_improved_and_clean_late_worsened_fraction=float(((gain49>0)&(late>0)).mean()),
        clean_late_worsened_given_bn49_improved_fraction=float((late[gain49>0]>0).mean()) if (gain49>0).any() else None,
        clean_late_worsened_given_bn49_not_improved_fraction=float((late[gain49<=0]>0).mean()) if (gain49<=0).any() else None,
        signed_residual_mean_eV={n:float((p-y).mean()) for n,p in arrays.items()},
        prediction_shift_summary={n:dict(mean_eV=float((arrays[n+"_clean"]-arrays[n+"_raw"]).mean()),
            std_eV=float((arrays[n+"_clean"]-arrays[n+"_raw"]).std())) for n in ("selected","final")},
        limitations=["Consumed selection-used50K; posthoc descriptive strata, no routing or model choice",
            "Shared selected-clean error mechanically couples BN-gain and late-deterioration correlations",
            "Row bootstrap is not seed uncertainty; no multiplicity-adjusted slice claim",
            "Frozen output comparisons do not establish training-time BN/dropout or optimization causality"])
    rows = dict(source_idx=expected, bn49_gain_eV=torch.from_numpy(gain49.copy()),
        bn60_gain_eV=torch.from_numpy(gain60.copy()), clean_late_deterioration_eV=torch.from_numpy(late.copy()),
        raw_late_deterioration_eV=torch.from_numpy(raw_late.copy()))
    return report, rows


def prepare(repo, kind):
    from .research_memory.plan import plan
    from .training_reproducibility import atomic_json, sha256_file
    here = repo / "experiments" / KINDS[kind]
    rel = here.relative_to(repo).as_posix()
    if (here / "rml").exists():
        raise FileExistsError("Prospective already published")
    records = {}
    if kind == "trace_pair":
        owner = Path("D:/w/k1-consistency-500k")
        stage = owner / "platforms/_records/kaggle/training/pcqm_k1_consistency_ablation_500k_s42_v1/stages"
        for arm in ARMS:
            for name in ("trace.json", "stage_manifest.json"):
                records[f"{arm}/{name}"] = stage / arm / name
        accepted = owner / "experiments/pcqm_k1_consistency_ablation_500k/submission_v1/terminal_inspection"
        for name in ("inspection_report.json", "stage_acceptance.md"):
            records[name] = accepted / name
        records["owning_protocol.md"] = owner / "experiments/pcqm_k1_consistency_ablation_500k/protocol.md"
        artifacts = {}
    else:
        desktop = Path("D:/文档/molgap")
        prior = json.loads((desktop / "experiments/pcqm_k1_late_weight_average/inputs.json").read_text(encoding="utf-8"))
        artifacts = {"selected_raw":prior["artifacts"]["selected_original"],
                     "selected_clean":prior["artifacts"]["selected_clean_bn"],
                     "gptrans":prior["artifacts"]["gptrans_predictions"]}
        completed = desktop / "experiments/pcqm_k1_late_weight_average/results"
        completion = json.loads((completed/"completion.json").read_text(encoding="utf-8"))
        for name in ("final_raw", "final_clean"):
            file = {"final_raw":"final_original.pt", "final_clean":"final_clean_bn.pt"}[name]
            artifacts[name] = dict(path=str(completed/file), sha256=completion["files"][file])
        records = {"accepted_average_result.json":completed/"result.json",
                   "accepted_average_decision.md":desktop/"experiments/pcqm_k1_late_weight_average/terminal_decision.md"}
    snapshots = here / "source_snapshot"
    for name, path in records.items():
        target = snapshots / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for item in artifacts.values():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("Accepted prediction SHA differs")
    executed = ["src/molgap/k1_saved_analysis.py", f"{rel}/run.py"]
    inputs = dict(kind=kind, cpu_threads=4, ceiling_seconds=600, development_bounds=[500000,550000],
        artifacts=artifacts, executed_source_files={n:sha256_file(repo/n) for n in executed},
        snapshot_files={n:sha256_file(snapshots/n) for n in records},
        snapshot_origins={n:str(p) for n,p in records.items()})
    atomic_json(here / "inputs.json", inputs)
    roles = dict(internal_development="Aggregate existing trace only; no row labels decoded" if kind == "trace_pair" else
        "Already selection-used50000 retained prediction/target rows; new posthoc metric arithmetic only, no model selection",
        official_validation="untouched", test_dev="untouched", test_challenge="untouched", common="untouched", ood="untouched")
    atomic_json(here/"roles.json",roles)
    policy_id = "pcqm-k1-saved-" + kind.replace("_", "-")
    policy = json.loads((repo/"research_memory/policies/pcqm-k1-late-weight-average.1.json").read_text(encoding="utf-8"))
    policy.update(policy_id=policy_id, comparability_selector={"scientific_contract":policy_id+"-v1"}, created_from_source_digest=sha256_file(here/"protocol.md"))
    atomic_json(repo/f"research_memory/policies/{policy_id}.1.json",policy)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,text=True).strip()
    suffix = "-attempt002" if kind == "trace_pair" else ""
    tid, run, cost_id = "TB-k1-saved-"+kind.replace("_","-")+"-20261008"+suffix, "local-k1-saved-"+kind+"-20261008"+suffix, "cost-k1-saved-"+kind+"-expected"+suffix
    refs = ["pcqm-k1-complete-module-audit-20261008", "pcqm-k1-late-weight-average-20261008"]
    hypothesis = (dict(observed_deficiency="500K consistency stage is incomplete; selected endpoints obscure same-step evolution.",
        alternative_explanations=["Benefit is delayed", "Stage trajectory differs without identifying final outcome", "BN-state sensitivity confounds raw dev ordering"],
        changed_mechanism="No intervention: align already accepted aggregate traces by actual steps/exposure",
        cheapest_falsifier="Read two retained23-epoch traces and accepted metadata; no model/prediction tensors") if kind == "trace_pair" else
        dict(observed_deficiency="Aggregate BN gains and late losses do not describe breadth/concentration or molecular overlap.",
        alternative_explanations=["Gain is concentrated in baseline-error tails", "Broad small output-state shifts", "Correlation is mechanical due to shared errors"],
        changed_mechanism="No intervention: per-row decomposition of fixed raw/clean endpoint predictions",
        cheapest_falsifier="Hash-verify and align retained50K predictions; fixed strata and error-delta arithmetic only"))
    hypothesis.update(hypothesis_id="H-"+tid, supporting_evidence_ids=refs, expected_native_cost_ref=cost_id,
        related_closed_family_ids=["k1-complete-module-audit", "k1-late-weight-average"],
        decision_changed_if_positive="Record a descriptive discriminator for the existing route; no promotion/early-stop/training release",
        decision_changed_if_negative="Record insufficient or contrary descriptive evidence; no retry/new module")
    trajectory = dict(schema="molgap-trajectory-v1", trajectory_id=tid, record_mode="prospective", owner="desktop", track="B",
        family_id="k1-saved-"+kind, question=("When does the accepted partial consistency trajectory differ at matched exposure?" if kind == "trace_pair" else
        "Is retained BN gain broad or tail-concentrated, and how does it overlap clean late deterioration?"), hypothesis=hypothesis,
        state_at_start=dict(source_commit=commit, source_config_identity=sha256_file(here/"inputs.json"),
            contract_refs=[f"{rel}/protocol.md",f"{rel}/inputs.json"], reference_ids=[], parent_trajectory_ids=[], prior_evidence_ids=refs,
            role_snapshot_refs=[f"{rel}/roles.json"], budget_snapshot_ref=f"{rel}/protocol.md"),
        actions=[dict(action_id="A001", type="local_saved_artifact_analysis", run_ids=[run], attempt_ids=["attempt-001"], source_commit=commit,
            evidence_refs=[f"{rel}/inputs.json"], cost_event_ids=[cost_id])], result=dict(evidence_ids=[], evidence_refs=[]),
        decision=dict(outcome="ACTIVE", decision_ref=f"{rel}/decision.md", next_allowed_actions=["One bounded saved-artifact CPU analysis"], reopen_conditions=[]),
        comparison_class="CONTEXT_ONLY", comparison_readiness_ref=f"{rel}/protocol.md",
        comparison_blockers=["Partial trace is not terminal truth" if kind == "trace_pair" else "Consumed development and descriptive posthoc strata"], reference_bundle_id=None)
    cost = dict(schema="molgap-cost-event-v1",cost_event_id=cost_id,trajectory_id=tid,action_id="A001",run_id=run,attempt_id="attempt-001",
        platform="local-windows",hardware="CPU saved-artifact arithmetic; no accelerator",category="other",evidence_ref=f"{rel}/protocol.md",
        measurement=dict(wall_hours=dict(value=600/3600,status="estimated"),cpu_hours=dict(value=None,status="measurement_missing"),
            device_hours=dict(value=None,status="not_applicable"),queue_hours=dict(value=None,status="not_applicable")))
    spec = dict(trajectory=trajectory,costs=[cost],decision_state=dict(available_actions=["RUN_DIAGNOSTIC","NO_TRAIN"],chosen_action="RUN_DIAGNOSTIC",
        policy_id=policy_id,policy_version="1",state_timestamp=datetime.now(timezone.utc).isoformat(),source_commit=commit,
        budget_snapshot_ref=f"{rel}/protocol.md",role_snapshot_refs=[f"{rel}/roles.json"]))
    atomic_json(here/"plan_input.json",spec)
    atomic_json(here/"plan_receipt.json",plan(repo,spec,f"{rel}/rml"))
    print("PROSPECTIVE_PUBLISHED_NO_MODEL_EXECUTED",kind)


def run(repo, kind):
    start, cpu = time.perf_counter(), time.process_time()
    os.environ.update({n:"4" for n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")})
    from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file
    here = repo/"experiments"/KINDS[kind]
    inputs = json.loads((here/"inputs.json").read_text(encoding="utf-8"))
    trajectory = json.loads((here/"rml/trajectory.json").read_text(encoding="utf-8"))
    if inputs["kind"] != kind or trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Active matching prospective required")
    for name, digest in inputs["executed_source_files"].items():
        if sha256_file(repo/name) != digest:
            raise ValueError("Executed source differs")
    for name, digest in inputs["snapshot_files"].items():
        if sha256_file(here/"source_snapshot"/name) != digest:
            raise ValueError("Frozen metadata differs")
    output = here/"results"
    if output.exists():
        raise FileExistsError("Never overwrite an attempt")
    import torch
    from threadpoolctl import threadpool_info
    torch.set_num_threads(4)
    output.mkdir()
    read = lambda name: json.loads((here/"source_snapshot"/name).read_text(encoding="utf-8"))
    if kind == "trace_pair":
        traces, manifests = ({a:read(f"{a}/{n}") for a in ARMS} for n in ("trace.json","stage_manifest.json"))
        for a in ARMS:
            if inputs["snapshot_files"][f"{a}/trace.json"] != manifests[a]["artifacts"]["trace.json"]:
                raise ValueError("Trace not bound to immutable stage manifest")
        analysis = trace_pair(traces,manifests,read("inspection_report.json"))
        observed = []
    else:
        values = {}
        for name,item in inputs["artifacts"].items():
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise ValueError("Retained prediction bytes differ")
            values[name] = torch.load(item["path"],map_location="cpu",weights_only=True)
        analysis, rows = bn_rows(values,tuple(inputs["development_bounds"]))
        accepted = read("accepted_average_result.json")
        metric_names = {"selected_raw":"selected_original", "selected_clean":"selected_clean_bn",
            "final_raw":"final_original", "final_clean":"final_clean_bn", "gptrans":"gptrans_ema"}
        if any(not math.isclose(analysis["mae_eV"][n], accepted["metrics"][old]["mae_eV"], abs_tol=1e-10)
               for n,old in metric_names.items()):
            raise ValueError("Accepted whole-cohort endpoint does not reconstruct")
        atomic_torch_save(output/"row_deltas.pt",rows)
        observed = rows["source_idx"].tolist()
    result = dict(format="molgap-k1-saved-analysis-v1",kind=kind,status="complete",analysis=analysis,
        observed_source_idx=observed,source_hashes={n:item["sha256"] for n,item in inputs["artifacts"].items()},
        inputs_sha256=sha256_file(here/"inputs.json"),prospective_sha256=sha256_file(here/"rml/trajectory.json"),
        runtime=dict(device="cpu",cpu_threads=torch.get_num_threads(),torch=torch.__version__,
            native_threadpools=threadpool_info(),
            configured_native_thread_env={n:os.environ[n] for n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}),
        training_executed=False,inference_executed=False,optimizer_created=False,gradients_computed=False,
        checks=dict(input_hashes_verified=True,prospective_before_execution=True,no_model_execution=True,
            alignment_verified=True,partial_or_consumed_role_limits_preserved=True),
        wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu)
    if result["wall_seconds"] > inputs["ceiling_seconds"]:
        raise TimeoutError("Saved analysis ceiling exceeded")
    atomic_json(output/"result.json",result)
    atomic_json(output/"completion.json",dict(complete=True,inputs_sha256=result["inputs_sha256"],
        files={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(kind=kind,status="complete",wall_seconds=result["wall_seconds"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation",choices=("prepare","run","close"))
    parser.add_argument("kind",choices=KINDS)
    args = parser.parse_args()
    from .constants import REPO_ROOT
    if args.operation == "prepare":
        prepare(REPO_ROOT,args.kind)
    elif args.operation == "run":
        run(REPO_ROOT,args.kind)
    else:
        from .saved_diagnostic_closure import close_saved_diagnostic
        inputs = json.loads((REPO_ROOT/"experiments"/KINDS[args.kind]/"inputs.json").read_text(encoding="utf-8"))
        dev = list(range(*inputs["development_bounds"]))
        roles = {} if args.kind == "trace_pair" else {"internal_development":{a:dev for a in ("labels_read","metric_computed")}}
        print(json.dumps(close_saved_diagnostic(REPO_ROOT,REPO_ROOT/"experiments"/KINDS[args.kind])))


if __name__ == "__main__":
    main()
