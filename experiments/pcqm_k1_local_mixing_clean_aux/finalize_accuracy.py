"""Thin per-arm evidence export through existing RML finalization."""
from reconcile_accuracy import ROOT,HERE,RECORD,OUT,read,write,sha
from molgap.research_memory.finalize import finalize,verified_receipt
import shutil

def rel(path): return path.relative_to(ROOT).as_posix()
def main():
    metrics=read(RECORD/'scientific_metrics.json');scheduler=read(RECORD/'remote_observation.json')
    snapshots=RECORD/'input_authority';snapshots.mkdir(exist_ok=True)
    for name in ('experiment_family_artifacts.py','experiment_family_workflow.py'):
        source=__import__('pathlib').Path('D:/w/k1-slot96/src/molgap')/name
        destination=snapshots/name
        if destination.exists() and sha(destination)!=sha(source):raise ValueError('Inspector snapshot changed')
        shutil.copyfile(source,destination)
    write(snapshots/'source_identity.json',{'owner':'D:/w/k1-slot96','owner_head':'cae2fd934169a5c1b052efc33da5c5ec2243a2ce','scope':'Reviewed helper snapshots for metadata-only K1 float32 output inspection','files':{p.name:sha(p) for p in snapshots.glob('*.py')}})
    decision_path=RECORD/'terminal_decision.md'
    decision_path.write_text('''# SSMA accuracy terminal decision — 2026-10-02

Candidate: NEGATIVE_UNDER_CONTRACT. The original K1 reference is a completed,
accepted control; no standalone model promotion is asserted. Both arms completed
the exact authorized Kaggle3 kernel nvoid912/molgap-k1-v4-ssma-accuracy-100k-s42-v1,
ID136643751, actual version1, frozen source f21920ba. Authenticated COMPLETE,
entry bytes and manifest-bound artifacts are verified. Both selected epoch40;
40epochs,31240steps and3998720 presentations per arm are retained.

Original K1 MAE0.1412944608205557eV; SSMA0.14336151182115078eV.
Reference-minus-candidate gain−2.067051meV; paired-row95% interval
[−3.005820,−1.105339]meV,10000 draws,seed42. The predeclared3meV gain/sign gates
fail. Single-seed training stochasticity is unknown; development was selection-used.

Both arms pass independent mechanical inspection and their frozen report_only
runtime qualification. The old owner inspector used float64 target hashing;
the reviewed k1-screen-v1 profile correctly preserves the recipe's rawfloat32
pin a1c0b711… in both arms. Raw contract, manifest and predictions are unchanged;
reviewed inspector/profile bytes are retained under input_authority.

SSMA synchronized optimizer-step calibration0.097606659→0.145411622s,
overhead48.977153%; peak memory549407232→865263616bytes. Exceeding25% is reported
and cannot veto this accuracy-authorized attempt. Assigned-device training windows
5553.8131s(reference),6831.8262s(SSMA) are invocation lower bounds; CPU,queue,
bootstrap and full allocation are unknown. No causal whole-job speed claim.

No retry,500K/full scale-up,adoption or successor follows from this result.
Strict RML replay/comparison-readiness admission remains unevaluated; retained
trace evidence does not manufacture a passing readiness object. Protected roles
remain sealed. Prior cost-gated NO_TRAIN records retain their original decisions.

Authority: scientific_metrics.json, per-arm mechanical and acceptance records,
remote_observation.json, frozen training_protocol_kaggle3_accuracy_v1.md and
the original per-arm prospective trajectories. Attribution is in attribution.md.
''',encoding='utf-8')
    (RECORD/'attribution.md').write_text('''# SSMA failure-mode attribution — 2026-10-02

Observed deficiency: bounded layer6 SSMA on actual post-gate incoming messages
regresses2.067051meV at matched100K/40epoch exposure. Manifest/source/row/target,
finite states, complete exposure, selected-state and resume checks pass; no
execution failure explains the endpoint. The reused finite-width joint-neighbor
aggregation did not buy material accuracy at its measured cost.

Both arms select epoch40. This is not a plateau or exposure-shortage proof.
Online pre-update dropout training metrics differ from fixed-cohort evaluation;
single-seed endpoints and row bootstrap do not identify underfitting,overfitting,
representation harm or lost bond information. These causes remain insufficient_evidence.
The original incoming sum remains intact; the addon sees already gated messages.
An SSMA loss therefore does not prove no useful pre-gate neighbor signal exists.

Cheapest missing discriminator is a separately justified matched fixed-cohort
fit/transfer analysis or equation-level distinction for a new mechanism. Do not
repeat the closed FFT residual by seed,schedule or width. No diagnostic/training
was executed in this reconciliation and no protected role was accessed.
''',encoding='utf-8')
    for name in ('reference','ssma'):
        prospective=HERE/'kaggle3_accuracy_v1'/name/'trajectory.json';frozen=read(prospective)
        dest=prospective.parent/'rml_finalized'
        if dest.exists():print(verified_receipt(dest));continue
        tid=frozen['trajectory_id'];run=frozen['actions'][0]['run_ids'][0];eid='pcqm-k1-'+name+'-accuracy-kaggle3-100k-s42-v1-terminal'
        mechanical=read(RECORD/(name+'_mechanical.json'))
        if mechanical['status']!='MECHANICALLY_VERIFIED':raise ValueError('Mechanical gate blocked')
        outcome={'execution_status':'complete','artifact_status':'hash_verified_retained','comparison_status':'PAIRED_ENDPOINT_single_seed','scientific_status':'NEGATIVE_UNDER_CONTRACT' if name=='ssma' else 'context_only','transfer_status':'not_evaluated','budget_decision':'no_successor_or_scale_up','full_handoff_status':'not_applicable'}
        decision={'outcome':'NEGATIVE_UNDER_CONTRACT' if name=='ssma' else 'CLOSED','decision_ref':rel(decision_path),'next_allowed_actions':[],'reopen_conditions':['new_decision_relevant_explicitly_authorized_contract']}
        acceptance_path=RECORD/(name+'_acceptance.json');base=OUT/name/'family_outputs';contract=read(base/'training_contract.json');expect=contract['acceptance_requirements']
        role_use={'official_train_prefix_0_100000':'used','fixed50k_development':'selection_used','official_validation':'untouched','test_dev':'untouched','test_challenge':'untouched'}
        costs=[]
        for category,filename in [('training','allocation_cost.json'),('preflight','diagnostic_cost.json')]:
            raw=read(RECORD/'qualification'/name/filename);measure={i['metric']:i['value'] for i in raw['costs']}
            costs.append({'schema':'molgap-cost-event-v1','cost_event_id':f'cost-{tid}-observed-{category}','trajectory_id':tid,'action_id':'A001','run_id':run,'attempt_id':frozen['actions'][0]['attempt_ids'][0],'category':category,'platform':'kaggle3','hardware':'Tesla T4; exclusive assigned invocation lower bound','evidence_ref':rel(acceptance_path),'measurement':{'device_hours':{'status':'measured','value':measure['device_seconds']/3600},'wall_hours':{'status':'measured','value':measure['wall_seconds']/3600},'cpu_hours':{'status':'measurement_missing','value':None},'queue_hours':{'status':'measurement_missing','value':None}}})
        roles=[]
        for role,kinds in [('official_train_prefix_0_100000',['training_membership','labels_read']),('fixed50k_development',['prediction_input','labels_read','metric_computed','selection_used'])]:
            for kind in kinds:
                roles.append({'schema':'molgap-role-event-v1','role_event_id':f'role-{tid}-{role}-{kind}','trajectory_id':tid,'action_id':'A001','run_id':run,'dataset_identity':'pcqm4mv2-ogb-fixed-100k-v1','row_manifest_hash':expect['source_idx_sha256'] if role=='fixed50k_development' else '1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d','role_name':role,'access_kind':kind,'selection_used':role=='fixed50k_development','evidence_ref':rel(acceptance_path)})
        acceptance={'format':'molgap-k1-accuracy-terminal-acceptance-v1','evidence_id':eid,'run_id':run,'trajectory_id':tid,'outcome':outcome,'trajectory_decision':decision,'role_use':role_use,'costs':costs,'roles':roles,'mechanical_acceptance_ref':rel(RECORD/(name+'_mechanical.json')),'scientific_metrics_ref':rel(RECORD/'scientific_metrics.json'),'runtime_qualification':'qualified_report_only','replay_eligible':False,'replay_exclusions':['strict_comparison_readiness_not_evaluated'],'role_observation_basis':'Frozen trainer/prospective, hash-bound predictions,selected/resume states and 40epoch observed trace; no new protected access'}
        write(acceptance_path,acceptance)
        paths=[decision_path,acceptance_path,RECORD/'attribution.md',RECORD/'scientific_metrics.json',RECORD/'remote_observation.json',RECORD/(name+'_mechanical.json'),RECORD/(name+'_retrieval.json'),HERE/'training_protocol_kaggle3_accuracy_v1.md',HERE/('training_recipe_kaggle3_accuracy_'+name+'_v1.json'),HERE/'kaggle3_accuracy_submission_v1/submission-response.json',HERE/'reconcile_accuracy.py',HERE/'analyze_accuracy.py',__import__('pathlib').Path(__file__)]
        paths+=list(snapshots.iterdir())+list((RECORD/'qualification'/name).glob('*.json'))+list(base.glob('*'))
        hashes={rel(p):sha(p) for p in paths if p.is_file()};retained=[acceptance_path,decision_path,RECORD/'attribution.md',RECORD/'scientific_metrics.json']+list(base.glob('*'))
        evidence={'format':'molgap-v5-evidence-envelope-v1','contract':'MOLGAP-COMMON-V5-FINAL','evidence_id':eid,'track':'B','scope':'desktop_same_job_k1_ssma_accuracy_100k_saved_artifacts','legacy_contract':'k1-ssma-kaggle3-accuracy-v1','outcome':outcome,'authority':{'pointers':[rel(decision_path),rel(acceptance_path),rel(HERE/'training_protocol_kaggle3_accuracy_v1.md'),rel(RECORD/'scientific_metrics.json')]},'role_use':role_use,'migration':{'migrated_at':'2026-10-02','training_executed':False,'inference_executed':False,'scientific_reinterpretation':False,'verification_scope':'No-inference metadata export of independently executed prospective training'},'observed_execution':{'training_executed':True,'inference_executed':False,'execution_ref':rel(RECORD/'remote_observation.json')},'artifacts':[{'name':p.name,'locator':rel(p),'sha256':hashes[rel(p)],'availability':'locally_retained_hash_verified'} for p in retained]}
        observation={'schema':'molgap-same-run-observation-v1','spec_identity':mechanical['context']['spec_identity'],'logical_run_id':mechanical['context']['logical_run_id'],'platform_name':'kaggle','platform_run_reference':mechanical['context']['run_reference'],'attempt_id':frozen['actions'][0]['attempt_ids'][0],'source_commit':mechanical['context']['source_commit'],'source_package_sha256':mechanical['context']['source_archive_sha256']}
        terminal={'format':'molgap-rml-terminal-package-v1','trajectory_id':tid,'run_id':run,'action_id':'A001','finalized_at':scheduler['observed_at_utc'],'acceptance_ref':rel(acceptance_path),'artifact_hashes':hashes,'evidence':evidence,'decision':decision,'costs':costs,'roles':roles,'same_run_observation':observation}
        terminal_path=RECORD/(name+'_terminal.json');write(terminal_path,terminal)
        print({'arm':name,'finalization':finalize(ROOT,rel(prospective),rel(terminal_path))['status']})
if __name__=='__main__':main()
