"""Freeze this question's bounded CPU initialization and utility discriminator."""
from pathlib import Path
import json
import subprocess
from datetime import datetime,timezone
from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json,sha256_file

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
REL=HERE.relative_to(ROOT).as_posix()
TID='TB-k1-slot96-cpu-qualification-20261002'
RUN='local-k1-slot96-qualification-20261002'

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    if (HERE/'qualification').exists():raise FileExistsError('Prospective already exists')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    prior='pcqm-k1-slot-readout-diagnostic-20261002'
    prior_ref='pcqm-k1-v4-192-kaggle3-reference-custody-20261002'
    refs=[prior,prior_ref]
    cost={'schema':'molgap-cost-event-v1','cost_event_id':'cost-k1-slot96-cpu-expected',
        'trajectory_id':TID,'action_id':'A001','run_id':RUN,'attempt_id':'cpu-001',
        'platform':'local-windows','hardware':'CPU4threads; no accelerator','category':'preflight',
        'evidence_ref':f'{REL}/protocol.md','measurement':{
            'wall_hours':{'value':300/3600,'status':'estimated'},
            'cpu_hours':{'value':1200/3600,'status':'estimated'},
            'device_hours':{'value':None,'status':'not_applicable'},
            'queue_hours':{'value':None,'status':'not_applicable'}}}
    state={'source_commit':commit,'contract_refs':[f'{REL}/protocol.md',f'{REL}/evidence_review.md'],
        'reference_ids':[prior_ref],'parent_trajectory_ids':['TB-k1-slot-readout-diagnostic-20261002'],
        'prior_evidence_ids':refs,'role_snapshot_refs':[f'{REL}/role_plan.json'],
        'budget_snapshot_ref':f'{REL}/protocol.md',
        'source_config_identity':'K1 atom192 edge64 slot96 layers9; frozen model/worker hashes in cpu_frozen.json'}
    hypothesis={'hypothesis_id':'H-k1-slot96-cpu-20261002',
        'observed_deficiency':'Node widening regressed; observed final global return is substantial but causal utility and isolated slot96 capacity remain unresolved.',
        'supporting_evidence_ids':refs,'alternative_explanations':['Final slot can be harmful despite large norm.',
            'Slot64 may already suffice; extra capacity can worsen generalization.',
            'Single-seed optimization and runtime distributions may explain endpoint changes.'],
        'changed_mechanism':'Frozen selected192 last-mixer output is bypassed only for CPU causal utility measurement; new random slot96 state changes only latent dimension.',
        'cheapest_falsifier':'2048 original/zero-final-slot paired forwards plus finite candidate CPU construction; stop if original prediction reconstruction or 300-second ceiling fails.',
        'related_closed_family_ids':['k1-slot-readout-diagnostic','k1-node-width','k1-full-convergence'],
        'expected_native_cost_ref':cost['cost_event_id'],
        'decision_changed_if_positive':'If zero-slot regression >=1meV, review single candidate slot96 100K training already authorized by user; no automatic scale-up.',
        'decision_changed_if_negative':'No slot-capacity training if final-slot utility gate fails; retain diagnostic and unresolved information-quality attribution.',
        'historical_unknowns':['Single-seed variance.','Large-molecule effect absent from retained prefix.']}
    trajectory={'schema':'molgap-trajectory-v1','trajectory_id':TID,'record_mode':'prospective',
        'track':'B','owner':'desktop','family_id':'k1-slot-width96','question':
        'Does frozen final-slot utility justify testing isolated latent64-to96 capacity, and does candidate construction preserve family semantics?',
        'hypothesis':hypothesis,'state_at_start':state,'actions':[{'action_id':'A001','type':'cpu_frozen_ablation_and_initialization',
            'source_commit':commit,'run_ids':[RUN],'attempt_ids':['cpu-001'],'evidence_refs':[f'{REL}/protocol.md'],
            'cost_event_ids':[cost['cost_event_id']]}],'result':{'evidence_ids':[],'evidence_refs':[]},
        'decision':{'outcome':'ACTIVE','decision_ref':f'{REL}/protocol.md','next_allowed_actions':['A001'],'reopen_conditions':[]},
        'comparison_class':'CONTEXT_ONLY','comparison_readiness_ref':f'{REL}/protocol.md',
        'comparison_blockers':['Consumed development role and single seed; zero-slot interference does not prove capacity bottleneck.'],
        'reference_bundle_id':'k1-v4-192-kaggle3-reference-custody-20261002'}
    policy={'schema':'molgap-policy-v1','policy_id':'k1-slot96-cpu-qualification','version':'1',
        'policy_type':'research_action','status':'candidate','comparability_selector':{'scientific_contract':'k1-slot96-cpu-qualification-v1'},
        'required_observable_fields':['user_requested_scope_reviewed'],
        'action_rule':{'field':'user_requested_scope_reviewed','operator':'eq','threshold':1,'action':'QUALIFY_CPU'},
        'borderline_action':'NO_TRAIN','observation_point':None,'promotion_rule':None,'early_stop_rule':None,
        'cost_model':{'kind':'measured_only','assumptions':[]},
        'approval':{'approved_by':None,'approved_at':None,'authority_ref':None},'created_from_source_digest':sha256_file(HERE/'protocol.md')}
    atomic_json(ROOT/'research_memory/policies/k1-slot96-cpu-qualification.1.json',policy)
    action={'trajectory_id':TID,'state_timestamp':'2026-10-02','evidence_ids':refs,
        'user_requested_scope_reviewed':1,'authority':'User 2026-10-02 authorized selected new Kaggle3 experiment and shutdown after submission'}
    atomic_json(HERE/'cpu_action_inputs.json',action)
    spec={'trajectory':trajectory,'costs':[cost],'action_inputs_ref':f'{REL}/cpu_action_inputs.json',
        'decision_state':{'known_trajectory_ids':[],'known_evidence_ids':[],'active_reference_ids':[],
            'available_actions':['QUALIFY_CPU','NO_TRAIN'],'chosen_action':'QUALIFY_CPU','policy_id':policy['policy_id'],
            'policy_version':'1','role_snapshot_refs':state['role_snapshot_refs'],'budget_snapshot_ref':state['budget_snapshot_ref'],
            'state_timestamp':'2026-10-02','source_commit':commit}}
    atomic_json(HERE/'qualification_plan.json',spec)
    result=plan(ROOT,spec,HERE/'qualification')
    names=['experiments/pcqm_k1_slot_width96/qualify.py','src/molgap/k1_slot_width96.py',
        'src/molgap/qm9_neural_atom.py','src/molgap/k1_screen_training.py','src/molgap/training_reproducibility.py',
        'src/molgap/v4_runtime.py','experiments/pcqm_k1_slot_width96/protocol.md','experiments/pcqm_k1_slot_width96/role_plan.json']
    raw=ROOT/'platforms/_records/kaggle/training/pcqm_k1_width256_reference_kaggle3_v1/k1_pair/reference'
    checkpoint=raw/'selected_model.pt'; predictions=raw/'development_predictions.pt'
    target=HERE/'reference_binding/target_transform.json'
    from molgap.k1_slot_width96 import CONFIG,EXPECTED_PARAMETER_COUNT,REFERENCE_PARAMETER_COUNT
    freeze={'source_sha256':{name:sha256_file(ROOT/name) for name in names},
        'model_declarations':{'config':CONFIG,'expected_parameter_count':EXPECTED_PARAMETER_COUNT,'reference_parameter_count':REFERENCE_PARAMETER_COUNT},
        'development_shard':{'path':'D:/文档/molgap/data/pcqm_fixed_100k_v1/train/train_shard_0002.pt',
            'sha256':'f8c0d054d4794ce9a8887e6c1799806d89533f6ed3aeb93f3e838b7319ac842a'},
        'reference_checkpoint':{'path':checkpoint.as_posix(),'sha256':sha256_file(checkpoint)},
        'reference_predictions':{'path':predictions.as_posix(),'sha256':sha256_file(predictions)},
        'target_transform':{'path':target.as_posix(),'sha256':sha256_file(target)},
        'trajectory_sha256':sha256_file(HERE/'qualification/trajectory.json'),'max_wall_seconds':300,
        'source_commit':commit,'prepared_at_utc':datetime.now(timezone.utc).isoformat()}
    atomic_json(HERE/'cpu_frozen.json',freeze)
    print(json.dumps(result))

if __name__=='__main__':main()
