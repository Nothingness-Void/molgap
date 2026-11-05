"""Draft the two authorized scientific inputs; no model or platform execution."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from molgap.experiment_execution import build_addon_declaration

root = Path(__file__).resolve().parents[2]
recipe_path = root / "experiments/pcqm_gptrans_norm_ema_100k_20261104/generated/92cb4884c511adb30ed6480456082af2054ec917a8f7d09f0c877c83b8c17bc2/uniform_wd_multi_ema/recipe.json"
source = json.loads((root / "experiments/pcqm_gptrans_triplet_norm_100k_20261104/data_declaration.json").read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


for slot, reference, candidate, reference_mode, candidate_addon in (
    ("ad", "full12", "full8_query4", "degree_bond_local_pair_update_norm_ema999", "gptrans_bond_pairnorm_query8"),
    ("bc", "full6", "full6_query6", "bond_pairnorm_full6", "gptrans_bond_pairnorm_query6"),
):
    experiment_id = f"pcqm_gptrans_query_readout_{slot}_100k_20261105"
    directory = root / "experiments" / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    relative = directory.relative_to(root).as_posix()
    recipe = json.loads(recipe_path.read_text())
    recipe["mode"] = reference_mode
    recipe["auxiliary_ema_decays"] = []
    recipe["execution_policy"]["auxiliary_ema"] = []
    recipe["profiling"] = {"warmup_steps": 2, "active_steps": 4,
                           "record_shapes": True, "profile_memory": True,
                           "module_paths": [], "capture_off_training_spans": False}
    write(directory / "reference_recipe.json", recipe)
    baseline_addon = ("gptrans_degree_bond_local_pair_update_norm" if slot == "ad"
                      else "gptrans_bond_pairnorm_full6")
    declaration = copy.deepcopy(source)
    arm = copy.deepcopy(source["arms"][0])
    arm["arm_id"] = reference
    arm["addons"] = [build_addon_declaration(("gptrans_t", "2"), baseline_addon, repo_root=root)]
    arm["training"]["recipe"]["sha256"] = hashlib.sha256((directory / "reference_recipe.json").read_bytes()).hexdigest()
    declaration["arms"] = [arm]
    write(directory / "data_declaration.json", declaration)
    value = {
        "format": "molgap-scientific-experiment-v1", "experiment_id": experiment_id,
        "logical_run_id": f"pcqm-gptrans-query-readout-{slot}-100k-20261105-s42",
        "attempt_id": "v1", "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_profile": "D:/molgap-machine-profile.json",
        "question": {
            "policy_id": experiment_id.replace("_", "-"), "action": "RUN_PAIRED_100K",
            "question": ("Can eight propagation layers and four dynamic relation-query layers retain full12 accuracy at lower cost?" if slot == "ad" else
                         "Do six dynamic relation-query layers recover accuracy beyond six propagation layers and the original head?"),
            "deficiency": "The final GPTrans head reads only the molecular node and self pair; the value of continuing all-atom propagation versus differentiable query readout is unmeasured.",
            "changed_mechanism": "propagation-depth-and-dynamic-molecular-query-readout",
            "supporting_evidence": ["ev-terminal-T-pcqm-gptrans-norm-ema-100k-20261104-s42-uniform_wd_multi_ema"],
            "alternatives": ["Later all-atom propagation is necessary", "Simple truncation is sufficient", "Dynamic relation queries recover useful computation"],
            "closed_families": ["gptrans-pair-memory-replay", "gptrans-frozen-readout"],
            "positive_decision": ("Nominate D only when upper paired95%CI for D-A <=1meV, point delta<=1meV and mature-pass training time saves >=15%; no automatic scale-up." if slot == "ad" else
                                  "Establish C-B readout value only for >=3meV improvement with upper paired95%CI<0; no cross-slot strict causal or promotion claim."),
            "negative_decision": "Retain truthful quality/cost evidence; no automatic successor, baseline substitution or scale-up."
        },
        "profile": {"owner": "desktop", "track": "B", "platform": "kaggle",
                    "hardware": "Tesla T4x2 FP32, one independent arm per device",
                    "estimated_device_hours": 6, "estimated_wall_hours": 3,
                    "cheapest_falsifier": "Same-job paired100K complete60 epochs; independent truncation pair separates removed propagation from readout value."},
        "authority": {"format": "molgap-execution-authority-v1", "user_authorized": True,
                      "account": "nvoid912", "arms": [reference, candidate], "seeds": [42],
                      "same_run_reference_arm_id": reference,
                      "allocated_t4_hours_ceiling": 24, "allocation_wall_limit_seconds": 43200,
                      "cleanup_reserve_seconds": 180,
                      "authority": "User explicitly authorized four100K arms60 epochs in two independent T4x2 notebooks; account selected from available API quota. No additional experiment total time cap. Numeric ceilings represent the12h platform session boundary only, not a user scientific budget.",
                      "scope": "A/full12 vs D/full8-query4 and B/full6 vs C/full6-query6; frozen seed42. Continuation authorized if necessary from reconciled state; no monitors or queue triggers.",
                      "development_role_label_use": True, "protected_roles_authorized": []},
        "seeds": [42],
        "shared": {"model": {"family": "gptrans_t", "version": "2", "addons": [baseline_addon]},
                   "recipe_ref": relative + "/reference_recipe.json",
                   "data_ref": {"spec_ref": relative + "/data_declaration.json", "arm_id": reference},
                   "initial_state_ref": "D:/molgap-artifacts/initializations/structural-bias-norm-100k/gptrans.pt",
                   "training": {}, "purpose": "architecture_comparison"},
        "arms": [{"arm_id": reference}, {"arm_id": candidate,
                  "model": {"family": "gptrans_t", "version": "2", "addons": [candidate_addon]}}],
        "same_run": {"reference_arm_id": reference,
                     "target_manifest_ref": "experiments/pcqm_k1_slot_width96/reference_binding/target_manifest.json"},
    }
    write(directory / "experiment.json", value)
    (directory / "decision.md").write_text("# Decision\n\nPending execution and strict acceptance. No model benefit or replay readiness is claimed.\n", encoding="utf-8")
    (directory / "STATUS.md").write_text("# Status\n\nProspective input drafted; not submitted. See experiment.json for exact identity and authority.\n", encoding="utf-8")
    (directory / "protocol.md").write_text(f"""# Protocol — dynamic molecular relation readout / {slot.upper()}

## Question and authority

User authorized two T4x2 notebooks, four seed42 arms, fixed100K complete60 epochs.
This notebook compares {reference} (GPU0) with {candidate} (GPU1). The companion
notebook answers a different matched contrast. Cross-notebook scores are descriptive.
Both reference arms are explicitly authorized controls, not historical-reference repairs.
No extra experiment total time cap was requested. The declared43200s allocation
and24 T4-device-hour fields encode the12h platform session boundary. Pause/resume
uses original source, exact cursor and reconciliation; never duplicate unknown work.
No local queue trigger, persistent monitor, official validation or test access.

## Scientific delta

A=12 full BondLocal+fresh-pair-update-norm blocks. B=first6 full blocks and original head.
C=first6 full blocks then original blocks7–12 as query-only readout.
D=first8 full blocks then original blocks9–12 as query-only readout.
All start with the same frozen12-layer initialization, including identical head,
overlapping layer weights and original12-layer DropPath probabilities. No depth
rescaling, new random initialization, changed cache or graph schema.
Query memory M contains split-point graph token and real atoms, stays unchanged
within the forward, and is never detached. Each readout layer applies its own
original LayerNorm and K/V projection to M; Q and the query-to-memory pair row evolve.
Column0 remains the split-point graph-token key, not the evolving query. Preserve
the two node messages, node-to-pair update, pair-to-node readback and key padding.
PairNorm means pair += normalize(new_update) over channels at each pair, never
normalize(pair+new_update). Readout layers omit BondLocal; that omission is part
of the complete architecture intervention, not a single-attention causal claim.
The unchanged head receives concat(query[:,0], pair_row[:,:,0,0]). No TopK or K/V cache.

## Recipe, data and evidence

reference_recipe.json explicitly pins BS128 drop_last, FP32/noTF32, LR0.001,
WD0.05,4-pass warmup/cosine1e-6, gradient clip1, EMA0.999, seed42 pass-randperm,
46860 optimizer updates /5998080 presentations /60 complete epochs. No auxiliary
EMA; same choice for all four arms. Train source rows[0,100000), development
[100000,150000), selected best-development EMA; official validation/test sealed.
Epoch cadence uses existing shared logging. Existing family sessions own native
repeat/resume qualification, checkpoints, curves, cost ledgers and terminal outputs.
Existing bounded profiler captures4 updates after2 warmup updates, including shapes
and memory. Instrumented windows are excluded from any clean throughput claim.
Report actual mature full-pass wall time (median passes10–60), throughput and native
cost separately. Peak device memory comes from measured profiler/runtime records;
missing whole-run peak stays unknown. No inference latency claim without a separately
identified measurement on authorized inputs.

## Gates

Primary AD: D-A selected EMA paired-row95%CI upper<=+1meV and point<=+1meV,
plus >=15% mature full-pass training-time saving. This is a prospective engineering
non-inferiority margin, not measured seed variance. BC: C improves B >=3meV with
paired95%CI upper<0. Use2000 paired bootstrap resamples on aligned finite50K rows.
BC alone cannot establish non-inferiority to A across notebooks. One training seed
does not measure training stochasticity. Full60-epoch curves, strict per-arm native
qualification, completion/exposure, selected artifacts, resume state, actual native
T4 cost and observed role use are required before terminal RML. No automatic500K.

## Prior evidence and non-duplication

RML exact-term lookup found no completed dynamic relation readout /6+6 /8+4 trial.
The historical architecture audit closes multi-depth/query-head interventions;
it does not test removing late propagation in this GPTrans BondLocal recipe.
experiments/pcqm_gptrans_500k_frozen_readout/decision.md tests static heads on
frozen encoders and explicitly leaves end-to-end attentive readout unresolved.
Pair-memory/conditional-flow negatives do not implement a single evolving query
with differentiable split-point memory. Latest triplet-ordering closure remains
negative and unchanged; this is not its successor gain/WD/EMA grid.

Literature: GPTrans https://www.ijcai.org/proceedings/2023/0396.pdf ;
CaiT https://arxiv.org/abs/2103.17239 (representation then class-query attention);
HySparse2 https://arxiv.org/html/2609.26368v1 is motivation only: no borrowed LLM
cache speedup numbers are treated as molecular measurements. An unprovided
synthetic equivalence claim is not qualification; local checks must execute.
""", encoding="utf-8")
