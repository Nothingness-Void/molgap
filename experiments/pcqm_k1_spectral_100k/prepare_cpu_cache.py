"""Publish CPU-cache prospective before any topology processing; no GPU work."""
from pathlib import Path
import json, subprocess, hashlib
from molgap.research_memory.plan import plan
from molgap.training_reproducibility import sha256_file
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
REL=HERE.relative_to(ROOT).as_posix()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,separators=(',',':')),encoding='utf-8')
def main():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tid='TB-k1-spectral-cpu-cache-100k-20261006'; cid='cost-'+tid+'-expected-cpu'
    refs=['pcqm-k1-slot-readout-diagnostic-20261002','pcqm-k1-slot-width96-kaggle3-100k-s42-v1-terminal']
    protocol=REL+'/protocol.md'; policy_id='pcqm-k1-spectral-cpu-cache-20261006'
    policy={'schema':'molgap-policy-v1','policy_id':policy_id,'version':'1','policy_type':'research_action','status':'candidate','comparability_selector':{'scientific_contract':'k1-spectral-cpu-cache-100k-v1'},'required_observable_fields':['evidence_review_complete'],'action_rule':{'field':'evidence_review_complete','operator':'eq','threshold':1,'action':'PREPARE_CPU_CACHE'},'borderline_action':'NO_TRAIN','observation_point':None,'promotion_rule':None,'early_stop_rule':None,'cost_model':{'kind':'measured_only','assumptions':[]},'approval':{'approved_by':None,'approved_at':None,'authority_ref':None},'created_from_source_digest':sha256_file(HERE/'protocol.md')}
    write(ROOT/f'research_memory/policies/{policy_id}.1.json',policy)
    write(HERE/'cpu_action_inputs.json',{'trajectory_id':tid,'state_timestamp':'2026-10-06','evidence_ids':refs,'evidence_review_complete':1})
    hypothesis={'observed_deficiency':'No established causal slot saturation; existing width and readout changes lack material gains. A full-spectrum Gaussian communication intervention requires qualified topology-only immutable inputs.','supporting_evidence_ids':refs,'alternative_explanations':['CPU cache is too expensive or has row/basis identity errors.','Compact spectral residual provides no useful information beyond K1.'],'changed_mechanism':'CPU-only complete normalized-Laplacian eigensystems from accepted real-bond topology, no labels used for construction.','cheapest_falsifier':'Validate source hashes, finite full EVD, row ordering, cache size and measured CPU cost before any GPU release.','related_closed_family_ids':['k1-slot-readout','k1-slot-width','k1-random-walk'],'expected_native_cost_ref':cid,'decision_changed_if_positive':'Freeze cache pins and prepare separately authorized two-arm100K workflow.','decision_changed_if_negative':'Stop GPU release; preserve infrastructure reason.','historical_unknowns':['Cache CPU cost and compact spectral prediction utility.']}
    trajectory={'schema':'molgap-trajectory-v1','trajectory_id':tid,'record_mode':'prospective','track':'B','owner':'desktop','family_id':'k1-spectral','question':'Can CPU-only full-spectrum inputs be qualified for one compact Gaussian spectral K1 residual?','hypothesis':hypothesis,'state_at_start':{'source_commit':commit,'source_config_identity':hashlib.sha256(json.dumps({'cache':'full-EVD','manifest':'1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d'},sort_keys=True,separators=(',',':')).encode()).hexdigest(),'contract_refs':[protocol,REL+'/evidence_review.md'],'reference_ids':[],'prior_evidence_ids':refs,'parent_trajectory_ids':[],'role_snapshot_refs':[REL+'/role_plan.json'],'budget_snapshot_ref':protocol},'actions':[{'action_id':'A001','type':'authorized_cpu_topology_cache','source_commit':commit,'run_ids':[tid+':cpu'],'attempt_ids':['cpu-cache-001'],'evidence_refs':[protocol],'cost_event_ids':[cid]}],'result':{'evidence_ids':[],'evidence_refs':[]},'decision':{'outcome':'ACTIVE','decision_ref':protocol,'next_allowed_actions':['A001'],'reopen_conditions':[]}}
    cost={'schema':'molgap-cost-event-v1','cost_event_id':cid,'trajectory_id':tid,'action_id':'A001','run_id':tid+':cpu','attempt_id':'cpu-cache-001','platform':'local','hardware':'desktop CPU; one BLAS thread; no accelerator work','category':'cache_build','evidence_ref':protocol,'measurement':{'wall_hours':{'status':'estimated','value':0.25},'device_hours':{'status':'not_applicable','value':None},'cpu_hours':{'status':'estimated','value':0.25},'queue_hours':{'status':'not_applicable','value':None}}}
    spec={'trajectory':trajectory,'costs':[cost],'action_inputs_ref':REL+'/cpu_action_inputs.json','decision_state':{'known_trajectory_ids':[],'known_evidence_ids':[],'active_reference_ids':[],'available_actions':['PREPARE_CPU_CACHE','NO_TRAIN'],'chosen_action':'PREPARE_CPU_CACHE','policy_id':policy_id,'policy_version':'1','role_snapshot_refs':[REL+'/role_plan.json'],'budget_snapshot_ref':protocol,'state_timestamp':'2026-10-06','source_commit':commit}}
    write(HERE/'cpu_cache_plan.json',spec)
    result=plan(ROOT,spec,REL+'/cpu_preparation')
    print(json.dumps(result,default=str))
if __name__=='__main__':main()


