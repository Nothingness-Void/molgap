"""Retain platform observations using the existing immutable launch receipt owner."""
from pathlib import Path
import json
import shutil
from molgap.experiment_launch import build_launch_receipt,canonical_json,write_launch_receipt
from molgap.experiment_spec import ExperimentSpec
from molgap.training_reproducibility import atomic_json,sha256_file

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
REL=HERE.relative_to(ROOT).as_posix()
PREPARED=ROOT/'platforms/_records/kaggle/staging/slot96_workflow_v1b'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def fact(value):return {'value':value,'missing_reason':None}

def main():
    spec=ExperimentSpec(read(HERE/'experiment_spec_kaggle3_v1.json'))
    submission=read(HERE/'submission_response.json')
    observation_path=sorted(HERE.glob('remote_observation_*.json'))[-1]
    observation=read(observation_path)
    manifest=read(PREPARED/'package/package_manifest.json')
    if (submission['status']!='submitted' or submission['reconciliation_required'] or
        observation['remote_metadata']['id']!=submission['kernel_id'] or
        observation['remote_metadata']['currentVersionNumber']!=submission['version_number'] or
        observation['remote_metadata']['machineShape']!='NvidiaTeslaT4' or
        observation['remote_metadata']['datasetDataSources']!=read(PREPARED/'kernel/kernel-metadata.json')['dataset_sources'] or
        observation['remote_entry_sha256']!=observation['submitted_entry_sha256'] or
        observation['status']['status']!='RUNNING'):
        raise RuntimeError('Actual source/version/allocation/startup reconciliation incomplete')
    prepared_receipt=build_launch_receipt(spec,PREPARED/'package',expected_package_identity=manifest['package_identity'])
    version=str(submission['version_number'])
    response={'format':'molgap-platform-response','version':1,'mode':'observed','outcome':'accepted',
      'conflict_kind':None,'binding':prepared_receipt['binding'],
      'canonical_platform_reference':fact(submission['kernel']),'platform_version':fact(version),
      'physical_runs':fact([{'run_identity':str(submission['kernel_id'])+':v'+version,
        'canonical_reference':submission['kernel'],'platform_version':fact(version),'arm_ids':['slot96']}]),
      'timestamp':fact(observation['observed_at_utc']),
      'monitor_paths':fact(['experiment/pair_state.json','experiment/slot96/preflight.log',
                           'experiment/slot96/output_manifest.json'])}
    receipt=build_launch_receipt(spec,PREPARED/'package',expected_package_identity=manifest['package_identity'],
                                response_json=canonical_json(response))
    receipts=HERE/'submission_v1/receipts';receipts.mkdir(exist_ok=True)
    receipt_path=write_launch_receipt(canonical_json(receipt),receipts,spec,PREPARED/'package',
                    expected_package_identity=manifest['package_identity'])
    atomic_json(HERE/'offline_handoff.json',{
      'scope':'Submission reconciliation only; runtime and scientific acceptance pending',
      'owner_branch':'codex/exp/k1-slot-width96','owner_checkout':'D:/w/k1-slot96',
      'kernel':submission['kernel'],'kernel_id':submission['kernel_id'],'version_number':submission['version_number'],
      'script_version_id':submission['script_version_id'],'last_observed_remote_state':observation['status'],
      'observation_ref':observation_path.relative_to(ROOT).as_posix(),
      'source_commit':manifest['source_commit'],'source_archive_sha256':manifest['archive_sha256'],
      'spec_identity':spec.identity,'package_identity':manifest['package_identity'],
      'initial_state_file_sha256':read(HERE/'qualification_result.json')['initial_state_file_sha256'],
      'source_dataset':'nvoid912/molgap-k1-slot96-source-s42-v1',
      'source_dataset_version_number':None,'source_version_reason':'Create response omitted version; all nine downloaded bytes verified',
      'dataset':'nvoid912/pcqm4mv2-ogb-fixed-100k-v1',
      'fixed_manifest_sha256':'1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d',
      'receipt_ref':receipt_path.relative_to(ROOT).as_posix(),
      'expected_outputs':'/kaggle/working/experiment',
      'actual_startup_log_available':observation['log_available'],
      'runtime_qualification':'pending; RUNNING is not proof that T4 barrier passed',
      'next_allowed_action':'Reconcile this exact run/version, retrieve small state/manifest, then manifest-bound artifacts and acceptance',
      'automatic_retry_or_successor':False,'desktop_server_takeover':False,
      'retained_records':{p.relative_to(ROOT).as_posix():sha256_file(p) for p in
        [HERE/'submission_response.json',HERE/'source_verification.json',observation_path,
         HERE/'archive_release_qualification.json',receipt_path]},
      'local_shutdown_authority':'User explicitly instructed shutdown after submission completed'})
    qualification=read(HERE/'archive_release_qualification.json')
    trajectory=read(HERE/'kaggle3_v1/slot96/trajectory.json')
    action=trajectory['actions'][0];run=action['run_ids'][0]
    role={'schema':'molgap-role-event-v1','role_event_id':'role-k1-slot96-archive-release-input-20261002',
      'trajectory_id':trajectory['trajectory_id'],'action_id':action['action_id'],'run_id':run,
      'dataset_identity':'pcqm4mv2-ogb-fixed-100k-v1',
      'row_manifest_hash':'3270b30071692d712c468fa52fa247d37f0a8e4b5444dc7c913aadb8ca0832c4',
      'role_name':'internal_development','access_kind':'prediction_input','selection_used':False,
      'evidence_ref':f'{REL}/archive_release_qualification.json'}
    atomic_json(HERE/'kaggle3_v1/slot96/roles/archive_release_input.json',role)
    cost={'schema':'molgap-cost-event-v1','cost_event_id':'cost-k1-slot96-archive-release-20261002',
      'trajectory_id':trajectory['trajectory_id'],'action_id':action['action_id'],'run_id':run,
      'attempt_id':action['attempt_ids'][0],'platform':'local-windows','hardware':'CPU4threads; no accelerator',
      'category':'preflight','evidence_ref':f'{REL}/archive_release_qualification.json','measurement':{
        'wall_hours':{'value':qualification['wall_seconds']/3600,'status':'measured'},
        'cpu_hours':{'value':None,'status':'measurement_missing'},
        'device_hours':{'value':None,'status':'not_applicable'},
        'queue_hours':{'value':None,'status':'not_applicable'}}}
    atomic_json(HERE/'kaggle3_v1/slot96/costs/archive_release.json',cost)
    failed=read(ROOT/'platforms/_records/kaggle/staging/slot96_workflow_v1/workflow_report.json')
    atomic_json(HERE/'preparation_reconciliation.json',{
      'failed_local_preparation_status':failed['status'],'reason':failed['error'],
      'published_training_trajectory':False,'gpu_submission':False,
      'correction':'Use canonical accepted reference evidence ID for RML state.reference_ids; retain run reference ID in comparison bundle',
      'successful_preparation':'slot96_workflow_v1b','executable_source_or_recipe_changed':False,
      'source_commit':manifest['source_commit'],'remote_submission_attempts':1})
    print(json.dumps({'status':'SUBMISSION_HANDOFF_RETAINED','receipt':str(receipt_path)}))

if __name__=='__main__':main()
