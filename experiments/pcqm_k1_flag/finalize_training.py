"""Retained prediction analysis and metadata-only export through RML finalize."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json
import shutil
import torch
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.research_memory.finalize import finalize

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'platforms/_records/kaggle/training/pcqm_k1_flag_kaggle3_v1'
REF = ROOT / 'platforms/_records/kaggle/training/pcqm_k1_width256_reference_kaggle3_v1/k1_pair/reference'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rel(p): return p.relative_to(ROOT).as_posix()
def write(p, obj): atomic_json(p, obj)

def main():
    mechanical = read(HERE / 'mechanical_acceptance.json')
    if mechanical['status'] != 'MECHANICALLY_VERIFIED': raise ValueError('Mechanical gate blocked')
    plan = read(HERE / 'family_acceptance_plan.json')['arms'][0]
    pred = OUT / 'development_predictions.pt'
    refpred = REF / 'development_predictions.pt'
    if sha256_file(pred) != mechanical['observed']['artifacts']['predictions']['sha256']: raise ValueError('Candidate hash changed')
    if sha256_file(refpred) != plan['reference_artifacts']['predictions']['sha256']: raise ValueError('Reference hash changed')
    module_spec = importlib.util.spec_from_file_location('retained_paired_metrics', ROOT / 'experiments/pcqm_gptrans_100k_transfer_control/analyze_pair.py')
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    metrics = module.paired_metrics(torch.load(refpred, map_location='cpu', weights_only=False), torch.load(pred, map_location='cpu', weights_only=False), 100000, 150000)
    metrics.update(bootstrap_replicates=1000, bootstrap_seed=42, reference_prediction_sha256=sha256_file(refpred), candidate_prediction_sha256=sha256_file(pred), training_stochasticity='unknown_single_seed', development_role='selection_used', endpoint_gate='gain>=0.003eV and positive lower95pct bound', scientific_outcome='NEGATIVE_UNDER_CONTRACT')
    if metrics['material_3mev_point_gain']: raise ValueError('Unexpected positive point gate; review decision')
    cr, rr = read(OUT / 'runtime_certificate.json'), read(REF / 'runtime_certificate.json')
    runtime = {key: {'reference': rr.get(key), 'candidate': cr.get(key), 'equal': rr.get(key)==cr.get(key)} for key in ('platform_id','runtime_fingerprint','software_fingerprint','determinism_fingerprint','precision','tf32_enabled','physical_batch_per_device','tail_batch_policy','calibration_fixture_sha256')}
    metrics['runtime_reference_comparison'] = runtime
    metrics['strict_causal_comparison'] = 'not_qualified_runtime_and_software_fingerprints_differ'
    metrics['replay_readiness'] = 'not_evaluated'
    write(HERE / 'scientific_metrics.json', metrics)
    arch=read(OUT/'architecture_preflight.json')
    if not (arch['accepted'] and cr['status']=='accepted' and arch['synchronized_step_overhead_fraction']<=3): raise ValueError('Runtime qualification failed')
    gain=metrics['paired_gain_eV']*1000; lo,hi=[v*1000 for v in metrics['row_bootstrap_95pct_eV']]
    (HERE/'terminal_decision.md').write_text(f'''# FLAG terminal decision — 2026-10-04

NEGATIVE_UNDER_CONTRACT. Exact Kaggle3 kernel nvoid912/molgap-k1-flag-100k-s42-v1,
ID136744623/version1 is COMPLETE. Manifest-bound retained artifacts pass mechanical
inspection. Candidate completed40epochs,31240optimizer updates and3998720
optimizer-batch presentations; selected live epoch40. FLAG uses11996160 gradient
loss-row evaluations, not equal computation with reference.

Retained K1 MAE{metrics['reference_mae_eV']:.12f}eV; FLAG{metrics['candidate_mae_eV']:.12f}eV.
Reference-minus-candidate gain{gain:.6f}meV; paired-row95% interval
[{lo:.6f},{hi:.6f}]meV (1000 draws,seed42). Point gain is below frozen3meV gate.
Row bootstrap does not measure training stochasticity; development was selection-used.

Candidate runtime/architecture repeatability,resume,selected-state and gradient
qualification pass. Synchronized step ratio{1+arch['synchronized_step_overhead_fraction']:.6f}
passes4x resource veto. Reference/candidate runtime and software fingerprints differ;
strict causal equivalence and RML replay readiness remain unqualified/unevaluated.
This does not change the observed frozen endpoint gate failure or permit retraining reference.

Assigned T4 training invocation9785.289311699s is a measured lower bound including
development/checkpoint work; diagnostic10.698678728s separately retained.
Full2T4 allocation,CPU,queue,bootstrap and idle-device costs are unknown.
No500K/full,model promotion,protected-role access or automatic successor follows.
New user-authorized research requires its own distinct question and prospective contract.
Complete history routes to archive; accepted canonical discovery routes to desktop.

Authority: protocol.md,scientific_metrics.json,mechanical_acceptance.json,
terminal_remote_status_20261004.json,selective_retrieval_receipt.json and training_acceptance.json.
''',encoding='utf-8')
    (HERE/'attribution.md').write_text('''# FLAG failure-mode attribution — 2026-10-04

Supervised atom-embedding robustness with M3/alpha0.001 did not buy the frozen
3meV clean-development improvement at retained100K/40epoch exposure. Finite,
aligned50K predictions,complete exposure,manifest/source and candidate runtime
qualification are retained; the small positive endpoint is not a material gain.
Clean inference architecture and parameter count remain original K1.

The extra computation is measured by a2.920258x synchronized step ratio. Epoch40
selection does not prove a plateau or exposure shortage. Online train MAE averages
three adversarial/dropout passes; it is not comparable to clean reference train fit.
Underfitting,overfitting,representation harm and insufficient exposure remain
insufficient_evidence. Runtime/software differences limit causal attribution.
Adversarial direction versus repeated dropout/loss computation is unidentified.

Cheapest missing discriminator is retained fixed-cohort clean fit/transfer evidence
if available under an authorized prospective analysis; mechanism-specific new
training would need distinct justification. Do not retry the closed objective by
nearby step size,seed or schedule. No inference or protected evaluation was executed
in this reconciliation; only retained consumed-development predictions were analyzed.
''',encoding='utf-8')
    prospective=HERE/'kaggle3_v1/flag/trajectory.json'; frozen=read(prospective)
    tid=frozen['trajectory_id']; action=frozen['actions'][0]; run=action['run_ids'][0]
    acceptance_path=HERE/'training_acceptance.json'; eid='pcqm-k1-flag-kaggle3-100k-s42-v1-terminal'
    outcome=dict(execution_status='complete',artifact_status='hash_verified_retained',comparison_status='PAIRED_ENDPOINT_single_seed_runtime_mismatch',scientific_status='NEGATIVE_UNDER_CONTRACT',transfer_status='not_evaluated',budget_decision='no_scale_up',full_handoff_status='not_applicable')
    decision=dict(outcome='NEGATIVE_UNDER_CONTRACT',decision_ref=rel(HERE/'terminal_decision.md'),next_allowed_actions=[],reopen_conditions=['Distinct new question with explicit user authority and prospective contract; no seed/schedule retry.'])
    role_use=dict(official_train_prefix_0_100000='used',fixed50k_development='selection_used',official_validation='untouched',test_dev='untouched',test_challenge='untouched')
    costs=[]
    for category,filename in [('training','allocation_cost.json'),('preflight','diagnostic_cost.json')]:
        raw=read(OUT/filename); measured={x['metric']:x['value'] for x in raw['costs']}
        costs.append(dict(schema='molgap-cost-event-v1',cost_event_id=f'cost-{tid}-observed-{category}',trajectory_id=tid,action_id='A001',run_id=run,attempt_id=action['attempt_ids'][0],category=category,platform='kaggle3',hardware='Tesla T4; exclusive assigned invocation lower bound',evidence_ref=rel(acceptance_path),measurement={'wall_hours':dict(status='measured',value=measured['wall_seconds']/3600),'device_hours':dict(status='measured',value=measured['device_seconds']/3600),'cpu_hours':dict(status='measurement_missing',value=None),'queue_hours':dict(status='measurement_missing',value=None)}))
    roles=[]
    for role,kinds,rowsha in [('official_train_prefix_0_100000',['training_membership','labels_read'],'1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d'),('fixed50k_development',['prediction_input','labels_read','metric_computed','selection_used'],plan['expected']['source_idx_sha256'])]:
        for kind in kinds:
            roles.append(dict(schema='molgap-role-event-v1',role_event_id=f'role-{tid}-{role}-{kind}',trajectory_id=tid,action_id='A001',run_id=run,dataset_identity='pcqm4mv2-ogb-fixed-100k-v1',row_manifest_hash=rowsha,role_name=role,access_kind=kind,selection_used=role=='fixed50k_development',evidence_ref=rel(acceptance_path)))
    acceptance=dict(format='molgap-k1-flag-training-terminal-acceptance-v1',evidence_id=eid,run_id=run,trajectory_id=tid,outcome=outcome,trajectory_decision=decision,role_use=role_use,costs=costs,roles=roles,mechanical_acceptance_ref=rel(HERE/'mechanical_acceptance.json'),scientific_metrics_ref=rel(HERE/'scientific_metrics.json'),runtime_qualification='candidate_qualified_reference_equivalence_not_established',replay_eligible=False,replay_exclusions=['strict_comparison_readiness_not_evaluated','runtime_software_reference_fingerprints_differ'],role_observation_basis='Frozen trainer/prospective,hash-bound predictions,selected/resume states and observed complete trace; no new protected access')
    write(acceptance_path,acceptance)
    retained=HERE/'retained_training'; retained.mkdir(exist_ok=True)
    for p in OUT.glob('*.json'):
        shutil.copyfile(p,retained/p.name)
    paths=[HERE/n for n in ('terminal_decision.md','attribution.md','scientific_metrics.json','training_acceptance.json','mechanical_acceptance.json','protocol.md','training_recipe.json','terminal_remote_status_20261004.json','selective_retrieval_receipt.json','terminal_source_verification.json','finalize_training.py')]+list(retained.glob('*'))+[OUT/n for n in ('development_predictions.pt','selected_model.pt','last_checkpoint.pt')]
    hashes={rel(p):sha256_file(p) for p in paths}
    authority=[rel(HERE/n) for n in ('terminal_decision.md','training_acceptance.json','protocol.md','scientific_metrics.json')]
    evidence=dict(format='molgap-v5-evidence-envelope-v1',contract='MOLGAP-COMMON-V5-FINAL',evidence_id=eid,track='B',scope='desktop_k1_flag_100k_saved_artifacts',legacy_contract='k1-flag-kaggle3-v1',outcome=outcome,authority=dict(pointers=authority),role_use=role_use,migration=dict(migrated_at='2026-10-04',training_executed=False,inference_executed=False,scientific_reinterpretation=False,verification_scope='Metadata export and retained-prediction comparison of independently executed prospective training'),observed_execution=dict(training_executed=True,inference_executed=False,execution_ref=rel(HERE/'terminal_remote_status_20261004.json')),artifacts=[dict(name=p.name,locator=rel(p),sha256=hashes[rel(p)],availability='locally_retained_hash_verified') for p in paths])
    context=mechanical['context']
    observation=dict(schema='molgap-same-run-observation-v1',spec_identity=context['spec_identity'],logical_run_id=context['logical_run_id'],platform_name='kaggle',platform_run_reference=context['run_reference'],attempt_id=action['attempt_ids'][0],source_commit=context['source_commit'],source_package_sha256=context['source_archive_sha256'])
    terminal=dict(format='molgap-rml-terminal-package-v1',trajectory_id=tid,run_id=run,action_id='A001',finalized_at=datetime.now(timezone.utc).isoformat(),acceptance_ref=rel(acceptance_path),artifact_hashes=hashes,evidence=evidence,decision=decision,costs=costs,roles=roles,same_run_observation=observation)
    write(HERE/'training_terminal.json',terminal)
    result=finalize(ROOT,rel(prospective),rel(HERE/'training_terminal.json'))
    print(json.dumps(dict(metrics=metrics,finalization=result),ensure_ascii=False))
if __name__=='__main__': main()
