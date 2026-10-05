"""Finish one partially published workflow using its unchanged source package."""
from pathlib import Path
import json
from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_package import verify_experiment_source_package
from molgap.experiment_execution import validate_execution_plan
from molgap.experiment_preflight import check_release_inputs, _atomic
from molgap.kaggle_workflow import validate_plan, stage_inputs, freeze_inputs, bind_release

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'platforms/_records/local/k1_weight_ema/workflow_v1'
OUTPUT = ROOT / 'platforms/_records/local/k1_weight_ema/workflow_reconciled_v1'


def main():
    previous = json.loads((PREVIOUS / 'workflow_report.json').read_text())
    if previous['status'] != 'RECONCILIATION_REQUIRED' or previous['stage'] != 'prospective':
        raise ValueError('Only the recorded partial prospective workflow is supported')
    publication = previous['stages'][-1]['result']
    if publication['status'] != 'PROSPECTIVE_PLANNED_RML_REBUILD_FAILED':
        raise ValueError('Expected both plans published with an isolated historical RML rebuild failure')
    spec = ExperimentSpec.from_json((HERE / 'experiment_spec_kaggle3_v1.json').read_text())
    manifest = json.loads((PREVIOUS / 'package/package_manifest.json').read_text())
    if spec.identity != previous['spec_identity'] or spec.identity != manifest['spec_identity']:
        raise ValueError('Frozen Spec changed')
    package = PREVIOUS / 'package'
    verified = verify_experiment_source_package(package, repo_root=ROOT)
    if verified['package_identity'] != manifest['package_identity']:
        raise ValueError('Package identity differs')
    plan = json.loads((HERE / 'workflow_plan_kaggle3_v1.json').read_text())
    bindings = {row['arm_id']: row for row in spec.to_dict()['prospective']['arms']}
    jobs = [{'arm_id': row['arm_id'], 'device': row['device'], 'recipe': row['recipe'],
             'initial_state': f"initial_states/{row['arm_id']}.pt",
             'trajectory_id': bindings[row['arm_id']]['trajectory_id']} for row in plan['arms']]
    qualified = validate_execution_plan(spec, jobs)
    for row in bindings.values():
        trajectory = json.loads((ROOT / row['output'] / 'trajectory.json').read_text())
        if trajectory['state_at_start']['source_commit'] != manifest['source_commit']:
            raise ValueError('Published source identity differs')
    if OUTPUT.exists():
        raise FileExistsError('Continuation output already exists; reconcile rather than overwrite')
    OUTPUT.mkdir(parents=True)
    stage = stage_inputs(repo_root=ROOT, output=OUTPUT, package=package, manifest=manifest,
        spec=spec, platform_plan=plan['kaggle'], metadata=validate_plan(spec, plan['kaggle']),
        initial_states={row['arm_id']: Path(row['initial_state']) for row in plan['arms']},
        jobs=jobs, acceptance_plan_path=plan['acceptance_plan'])
    release = check_release_inputs(spec, package, expected_package_identity=manifest['package_identity'],
        recipe_files={row['arm_id']: row['recipe'] for row in plan['arms']},
        initial_states={arm: stage['input_root'] / 'initial_states' / (arm + '.pt') for arm in bindings},
        required_modules=sorted({name for job in qualified for name in job['required_modules']} |
                                {'molgap.experiment_training_worker'}), input_root=stage['input_root'])
    if release['errors']:
        _atomic(OUTPUT / 'release_report.json', release)
        raise ValueError(release['errors'])
    freeze_inputs(stage, {arm: ROOT / row['output'] / 'trajectory.json' for arm, row in bindings.items()})
    bind_release(stage, spec, release)
    _atomic(OUTPUT / 'release_report.json', release)
    report = {'format': 'molgap-workflow-preparation-continuation-v1', 'status': 'PREPARED_FOR_PLATFORM',
        'previous_workflow_report': str(PREVIOUS / 'workflow_report.json'),
        'reconciliation': 'Original historical artifact bytes restored; RML rebuilt; unchanged published plans and source package reused.',
        'source_commit': manifest['source_commit'], 'package_identity': manifest['package_identity'],
        'spec_identity': spec.identity, 'source_dataset_dir': str(stage['input_root']),
        'kernel_dir': str(stage['kernel_dir']), 'release_report': str(OUTPUT / 'release_report.json'),
        'package_dir': str(package), 'prospective_published': True, 'submitted': False,
        'models_executed': False}
    _atomic(OUTPUT / 'workflow_report.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
