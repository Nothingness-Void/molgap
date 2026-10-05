"""Build registered two-arm plans; no model execution, planning publication or POST."""
from pathlib import Path
import argparse, copy, hashlib, json, re, shutil, subprocess
from molgap.experiment_execution import build_family_recipe
from molgap.experiment_spec import ExperimentSpec
from molgap.comparison_readiness import assess_comparison_prelaunch, validate_comparison_prelaunch
from molgap.k1_weight_ema import configuration
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import sha256_file
from molgap.v4_runtime import normalized_source_sha256, inspect_frozen_state_artifact

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
REL=HERE.relative_to(ROOT).as_posix(); REFERENCE=ROOT/'experiments/pcqm_k1_slot_width96/reference_binding'
POLICY='pcqm-k1-weight-ema-kaggle3-100k-launch'
TITLE='MolGap K1 Weight EMA 100K S42 V1'
RUN=re.sub(r'[^a-z0-9]+','-',TITLE.lower()).strip('-')
MODES=('reference','ema999')
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
def write(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(canonical(x))
def pin(p): return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha256_file(p)}
def generated(name,x): return {'path':f'{REL}/{name}','sha256':hashlib.sha256(canonical(x)).hexdigest()}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--state-timestamp',default='2026-10-06')
    parser.add_argument('--retained-owner',type=Path,default=Path('D:/w/k1-flag'))
    parser.add_argument('--refresh-source',action='store_true',help='Rebind unpublished plans to committed HEAD; preserve objective/recipe/initial bytes')
    args=parser.parse_args()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if args.source_commit!=commit: raise ValueError('Reviewed source commit must equal owner HEAD')
    if any((HERE/f'kaggle3_v1/{m}').exists() for m in MODES): raise FileExistsError('Prospective exists; reconcile instead of refresh')
    if (HERE/'experiment_spec_kaggle3_v1.json').exists() and not args.refresh_source: raise FileExistsError('Prepared spec exists; explicit --refresh-source required')
    expected=read(REFERENCE/'expected.json'); custody=read(REFERENCE/'custody_manifest.json')
    bundle=read(REFERENCE/'reference_bundle.json'); bindings=read(REFERENCE/'reference_artifact_bindings.json')
    retained=HERE/'reference_retained'; retained.mkdir(exist_ok=True)
    # Existing accepted metadata remain authoritative; copy missing ignored binaries
    # and runtime metadata into this question without replacing their original bytes.
    for key,entry in bindings.items():
        original=ROOT/entry['path']
        if not original.exists():
            source=args.retained_owner/entry['path']; destination=retained/source.name
            if sha256_file(source)!=entry['sha256']: raise ValueError('Retained reference hash differs: '+key)
            shutil.copyfile(source,destination); bindings[key]=pin(destination)
    bundle['reference_bundle_id']='k1-weight-ema-historical-clean-k1-custody-s42-v1'
    owners={'runtime_certificate':'runtime_certificate_ref','row_manifest':'row_manifest_ref','target_manifest':'target_manifest_ref','trace_manifest':'trace_manifest_ref','role_history':'role_history_ref','target_transform_asset':'target_transform_asset_ref','cost_records':'cost_records_ref','acceptance':'acceptance_ref','decision':'decision_ref'}
    for key,field in owners.items(): bundle[field]=bindings[key]['path']
    write(HERE/'historical_reference_bundle.json',bundle)
    write(HERE/'historical_reference_artifacts.json',bindings)
    initial_source=args.retained_owner/'experiments/pcqm_k1_flag/qualification/initial_state.pt'
    initial=HERE/'initial_state.pt'
    if initial.exists() and sha256_file(initial)!=sha256_file(initial_source): raise ValueError('Initial bytes changed')
    shutil.copyfile(initial_source,initial)
    initial_hash='8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd'
    inspect_frozen_state_artifact(initial,expected_state_sha256=initial_hash)
    refs=[custody['reference_evidence_id'],'pcqm-k1-flag-kaggle3-100k-s42-v1-terminal','pcqm-k1-slot-width96-kaggle3-100k-s42-v1-terminal','pcqm-k1-dropout-consistency2-kaggle3-100k-s42-v1-terminal','pcqm-k1-fusion-distill-strong-kaggle3-100k-s42-v1-terminal']
    policy={'schema':'molgap-policy-v1','policy_id':POLICY,'version':'1','policy_type':'research_action','status':'candidate','comparability_selector':{'scientific_contract':'k1-weight-ema-100k-v1'},'required_observable_fields':['evidence_review_complete'],'action_rule':{'field':'evidence_review_complete','operator':'eq','threshold':1,'action':'QUALIFY_THEN_TRAIN_PAIR_100K'},'borderline_action':'NO_TRAIN','observation_point':None,'promotion_rule':None,'early_stop_rule':None,'cost_model':{'kind':'measured_only','assumptions':[]},'approval':{'approved_by':None,'approved_at':None,'authority_ref':None},'created_from_source_digest':sha256_file(HERE/'protocol.md')}
    write(HERE/'launch_policy.json',policy)
    write(ROOT/f'research_memory/policies/{POLICY}.1.json',policy)
    arms=[]; prospects=[]; acceptance=[]; workflow_arms=[]
    for device,mode in enumerate(MODES):
        addon=None if mode=='reference' else 'k1_weight_ema'; tid='TB-k1-weight-ema-'+mode.replace('_','-')+'-kaggle3-100k-s42-v1'
        recipe=build_family_recipe(('neural_atom_k1','2'),addon=addon,source_idx_sha256=expected['source_idx_sha256'],target_sha256=expected['target_sha256'])
        if recipe['acceptance_requirements']!=expected: raise ValueError('Frozen exposure/roles differ')
        arm=copy.deepcopy(read(REFERENCE/'original_reference_arm.json'))
        arm.update(arm_id=mode,scientific_role='reference' if device==0 else 'candidate',addon_semantics='baseline' if device==0 else 'ordered')
        arm['initialization']={'kind':'frozen_state','seed':42,'state_sha256':initial_hash}
        arm['addons']=[] if device==0 else [{'name':addon,'version':'1','config':configuration(),'source_sha256':normalized_source_sha256(ROOT/'src/molgap/k1_weight_ema.py')}]
        arm['training']['recipe']['sha256']=hashlib.sha256(canonical(recipe)).hexdigest()
        if args.refresh_source:
            for name,value in ((f'training_recipe_{mode}.json',recipe),(f'arm_{mode}.json',arm)):
                prior=read(HERE/name)
                if name.startswith('arm_'):
                    # A reviewed source-only fix may rebind an unpublished addon.
                    for old,new in zip(prior['addons'],value['addons']):
                        old['source_sha256']=new['source_sha256']
                if canonical(prior)!=canonical(value): raise ValueError('Source refresh cannot change recipe/arm semantics: '+name)
        identity=dict(bundle['comparison_identity'])
        declared=[]
        if device==1:
            identity.update(ema_enabled=True,ema_decay=.999,ema_update_frequency='after-every-step',weight_semantics='ema',evaluation_weight_source='ema',checkpoint_selection_identity='best_internal_development_ema_model_over_40_epochs')
            declared=['ema_enabled','ema_decay','ema_update_frequency','weight_semantics','evaluation_weight_source','checkpoint_selection_identity']
        role_app={k:'applicable' for k in ('training_membership','prediction_input','labels_read','metric_computed','selection_used')}; role_app['external_submission']='not_applicable'
        trace={k:True for k in ('optimizer_step','sample_presentations','epoch_or_pass','learning_rate','live_train_metric','live_dev_metric','checkpoint_identity')}; trace['ema_dev_metric']=device==1
        comparison=assess_comparison_prelaunch(candidate_id=RUN+':'+mode,candidate_plan={'comparison_identity':identity,'source_config_status':'frozen','source_commit_or_archive':commit},reference_id=bundle['reference_id'],reference_bundle=bundle,experiment_purpose='contextual_experiment' if device==0 else 'ema_comparison',intervention_group_id='k1-weight-ema-selection',declared_intervention_fields=declared,role_applicability_plan=role_app,trace_plan=trace,runtime_qualification_plan={'status':'declared','runtime_certificate_required':True,'qualification_scope':'Assigned T4 train-only BS128; exact EMA recurrence, buffer copy, atomic EMA resume; same live optimizer trajectory; step ratio<=1.25'})
        validate_comparison_prelaunch(comparison)
        if not comparison['prelaunch_ready']: raise ValueError(comparison['blocker_codes'])
        protocol=f'{REL}/protocol.md'; recipe_name=f'training_recipe_{mode}.json'; comparison_name=f'comparison_prelaunch_{mode}.json'
        cost_id=f'cost-{tid}-expected-training'; run=RUN+':'+mode+':downstream'
        cost={'schema':'molgap-cost-event-v1','cost_event_id':cost_id,'trajectory_id':tid,'action_id':'A001','run_id':run,'attempt_id':'kaggle3-weight-ema-001','platform':'kaggle','hardware':f'Tesla T4; assigned GPU{device} in2T4 pair','category':'training','evidence_ref':protocol,'measurement':{'wall_hours':{'status':'estimated','value':4.0},'device_hours':{'status':'estimated','value':4.0},'cpu_hours':{'status':'measurement_missing','value':None},'queue_hours':{'status':'measurement_missing','value':None}}}
        hypothesis={'hypothesis_id':'H-'+tid,'observed_deficiency':'K1 weight averaging beyond raw selection remains untested. Cached server G1 slow-EMA lag is evidence for mechanism plausibility, not evidence K1 gains; K1 capacity/dropout/distillation changes remain subgate.','supporting_evidence_ids':refs,'alternative_explanations':['Raw weights already dominate EMA.','Live BN buffers mismatch averaged parameters.','Single-seed/development-selection variability explains small gains.'],'changed_mechanism':'Reference selects live weights; candidate keeps identical optimization and selects parameter EMA .999 only, updated after every optimizer step; buffers copy live; 40 checkpoint candidates each.','cheapest_falsifier':'Bounded train-only T4 recurrence/resume and overhead checks before all-arm training barrier; then matched complete100K40epoch endpoint.','related_closed_family_ids':['k1-flag','k1-slot-width96','k1-local-aggregation','k1-dropout-consistency','k1-fusion-distillation'],'expected_native_cost_ref':cost_id,'decision_changed_if_positive':'Review >=3meV gain and positive paired-row lower bound plus measured costs before considering transfer; no automatic scale-up.','decision_changed_if_negative':'Close fixed EMA .999 copy-live-buffer selection question; no automatic rate/window sweep.','historical_unknowns':['K1 EMA benefit.','Training stochasticity.','EMA BN compatibility.']}
        trajectory={'schema':'molgap-trajectory-v1','trajectory_id':tid,'record_mode':'prospective','track':'B','owner':'desktop','family_id':'k1-weight-ema','question':'Does parameter EMA .999 improve K1 beyond matched live-only selection at fixed100K?','hypothesis':hypothesis,'state_at_start':{'source_commit':commit,'source_config_identity':canonical_fingerprint(arm),'contract_refs':[protocol,f'{REL}/{recipe_name}',f'{REL}/evidence_review.md'],'reference_ids':[custody['reference_evidence_id']],'prior_evidence_ids':refs,'parent_trajectory_ids':[],'role_snapshot_refs':[f'{REL}/role_plan.json'],'budget_snapshot_ref':protocol},'actions':[{'action_id':'A001','type':'authorized_new_ema_selection_pair_100k','source_commit':commit,'run_ids':[run],'attempt_ids':['kaggle3-weight-ema-001'],'evidence_refs':[protocol],'cost_event_ids':[cost_id]}],'result':{'evidence_ids':[],'evidence_refs':[]},'decision':{'outcome':'ACTIVE','decision_ref':protocol,'next_allowed_actions':['A001'],'reopen_conditions':[]}}
        inputs={'trajectory_id':tid,'state_timestamp':args.state_timestamp,'evidence_ids':refs,'evidence_review_complete':1}
        input_name=f'action_inputs_{mode}.json'; plan_name=f'training_plan_{mode}.json'
        plan={'trajectory':trajectory,'costs':[cost],'action_inputs_ref':f'{REL}/{input_name}','decision_state':{'known_trajectory_ids':[],'known_evidence_ids':[],'active_reference_ids':[],'available_actions':['QUALIFY_THEN_TRAIN_PAIR_100K','NO_TRAIN'],'chosen_action':'QUALIFY_THEN_TRAIN_PAIR_100K','policy_id':POLICY,'policy_version':'1','role_snapshot_refs':[f'{REL}/role_plan.json'],'budget_snapshot_ref':protocol,'state_timestamp':args.state_timestamp,'source_commit':commit}}
        for name,value in ((recipe_name,recipe),(comparison_name,comparison),(input_name,inputs),(plan_name,plan),(f'arm_{mode}.json',arm)): write(HERE/name,value)
        arms.append(arm); prospects.append({'arm_id':mode,'trajectory_id':tid,'plan_spec_ref':f'{REL}/{plan_name}','plan_spec_sha256':hashlib.sha256(canonical(plan)).hexdigest(),'output':f'{REL}/kaggle3_v1/{mode}'})
        acceptance.append({'arm_id':mode,'adapter':'k1-screen-v1','expected':expected,'contract':generated(recipe_name,recipe),'comparison_prelaunch':generated(comparison_name,comparison),'reference_bundle':pin(HERE/'historical_reference_bundle.json'),'reference_artifacts':bindings})
        workflow_arms.append({'arm_id':mode,'device':device,'recipe':f'{REL}/{recipe_name}','initial_state':str(initial.resolve())})
    spec=ExperimentSpec({'schema_version':'molgap-experiment-spec-v2','experiment_id':'pcqm-k1-weight-ema-kaggle3-100k','logical_run_id':RUN,'arms':arms,'platform':{'name':'kaggle','accelerator':'Tesla T4','device_count':2,'cpu_cores':4,'memory_gib':29,'atomic_checkpoints':True,'retrievable_chunks':True},'prospective':{'arms':prospects,'same_run_replay':{'reference_arm_id':MODES[0],'candidate_arm_ids':[MODES[1]]}},'evidence':{'policy':{'name':'molgap-v5','version':'1','sha256':normalized_source_sha256(ROOT/'docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md')},'required_artifacts':['v5_evidence','costs','roles','trace_manifest','terminal_artifact']},'terminal_protocol':'molgap-experiment-terminal-descriptor-v1'})
    source='nvoid912/molgap-k1-weight-ema-source-s42-v1'
    write(HERE/'experiment_spec_kaggle3_v1.json',spec.to_dict())
    write(HERE/'family_acceptance_plan.json',{'format':'molgap-family-acceptance-plan-v1','spec_identity':spec.identity,'arms':acceptance})
    write(HERE/'workflow_plan_kaggle3_v1.json',{'format':'molgap-experiment-workflow-v1','spec_identity':spec.identity,'source_files':[],'arms':workflow_arms,'acceptance_plan':f'{REL}/family_acceptance_plan.json','kaggle':{'account':'nvoid912','kernel':'nvoid912/'+RUN,'title':TITLE,'datasets':[source,'nvoid912/pcqm4mv2-ogb-fixed-100k-v1'],'source_dataset':source,'accelerator':'NvidiaTeslaT4'}})
    print(json.dumps({'status':'CONFIGS_PREPARED_UNPUBLISHED','spec_identity':spec.identity,'source_commit':commit,'kernel':'nvoid912/'+RUN,'initial_file_sha256':sha256_file(initial),'models_executed':False,'submitted':False}))
if __name__=='__main__': main()
