"""Frozen continuation declarations over the existing platform preparation owners."""
import hashlib
import json
from pathlib import Path
import shutil

from molgap.experiment_package import verify_experiment_source_package
from molgap.experiment_spec import ExperimentSpec
from molgap.kaggle_workflow import _metadata, stage_inputs, freeze_inputs
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent


def main():
    old = ROOT / 'platforms/_records/kaggle/source_packages/k1_fusion_distill_v1'
    package = old / 'package'
    manifest = verify_experiment_source_package(package)
    spec = ExperimentSpec.from_json((package / 'experiment_spec.json').read_text())
    launch = json.loads((old / 'source_dataset/experiment_launch.json').read_text())
    destination = ROOT / 'platforms/_records/kaggle/source_packages/k1_fusion_distill_resume_v2_release'
    destination.mkdir(exist_ok=False)
    plan = json.loads((EXP / 'workflow_plan.json').read_text())['kaggle']
    plan['source_dataset'] = 'nvoid912/molgap-k1-fusion-distill-resume-s42-v2'
    plan['datasets'] = [plan['source_dataset'], 'nvoid912/pcqm4mv2-ogb-fixed-100k-v1',
                        'nvoid912/molgap-k1-fusion-teacher-train100k-v1']
    stage = stage_inputs(repo_root=ROOT, output=destination, package=package,
        manifest=manifest, spec=spec, platform_plan=plan, metadata=_metadata(plan),
        initial_states={j['arm_id']: old / 'source_dataset' / j['initial_state'] for j in launch['jobs']},
        jobs=launch['jobs'], acceptance_plan_path='experiments/pcqm_k1_fusion_distillation/family_acceptance_plan.json')
    retained = ROOT / 'platforms/_records/kaggle/training/k1_fusion_distill_v1_failure/experiment/distill_strong'
    names = ('last_checkpoint.pt', 'selected_model.pt', 'development_predictions.pt',
             'canonical_trace.json', 'canonical_trace.json.context.json')
    recovery = stage['input_root'] / 'recovery'
    recovery.mkdir()
    pins = {}
    for name in names:
        shutil.copyfile(retained / name, recovery / name)
        pins[name] = sha256_file(recovery / name)
    binding = {'spec_identity': spec.identity, 'package_identity': manifest['package_identity'],
        'original_run': launch['run_reference'], 'original_version': 1,
        'arm_id': 'distill_strong', 'start_epoch': 39, 'end_epoch': 40,
        'runtime_fingerprint': json.loads((retained / 'runtime_certificate.json').read_text())['runtime_fingerprint'],
        'target_identity': stage['launch']['target_identity'], 'files': pins}
    atomic_json(EXP / 'resume_binding_v2.json', binding)
    shutil.copyfile(EXP / 'resume_binding_v2.json', recovery / 'resume_binding.json')
    stage['launch']['recovery_binding_sha256'] = sha256_file(recovery / 'resume_binding.json')
    adapter = ROOT / 'platforms/kaggle/resume_frozen_k1_arm.py'
    stage['launch']['recovery_adapter_sha256'] = sha256_file(adapter)
    text = stage['entry_text']
    # The reviewed platform adapter is embedded, so the release entry hash binds
    # its exact bytes independently of the unchanged scientific source archive.
    text = text.replace('def main():', adapter.read_text() + '\n\ndef main():', 1)
    text = text.replace('    run_two_phase_pair(source_root=', '    resume_one_arm(source_root=', 1)
    text = text.replace('    config = json.loads(launch.read_text(encoding="utf-8"))',
        '    config = json.loads(launch.read_text(encoding="utf-8"))\n'
        '    if hashlib.sha256((launch.parent / "recovery/resume_binding.json").read_bytes()).hexdigest() != config["recovery_binding_sha256"]:\n'
        '        raise RuntimeError("Recovery binding changed after release")', 1)
    # Set CUDA visibility before any torch import, including the parent checker.
    guard = ('import os\nif os.environ.get("MOLGAP_RECOVERY_CHILD") != "1":\n'
        '    child_env = dict(os.environ, MOLGAP_RECOVERY_CHILD="1", CUDA_VISIBLE_DEVICES="1", PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8")\n'
        '    subprocess.run([sys.executable, __file__], env=child_env, check=True, timeout=1900)\n'
        '    raise SystemExit(0)\n\n')
    text = text.replace('# Local preparation replaces', guard + '# Local preparation replaces', 1)
    stage['entry_text'] = text
    bindings = {b['arm_id']: ROOT / b['output'] / 'trajectory.json' for b in spec.to_dict()['prospective']['arms']}
    freeze_inputs(stage, bindings)
    atomic_json(EXP / 'recovery_preparation_v2.json', {
        'status': 'STAGED_FOR_RELEASE_CHECK', 'source_package': str(package.relative_to(ROOT)),
        'prepared_directory': str(destination.relative_to(ROOT)), 'spec_identity': spec.identity,
        'package_identity': manifest['package_identity'], 'source_commit': manifest['source_commit'],
        'entry_sha256': sha256_file(stage['entry_path']), 'adapter_sha256': sha256_file(adapter),
        'binding_sha256': sha256_file(recovery / 'resume_binding.json'),
        'source_dataset': plan['source_dataset'], 'datasets': plan['datasets'],
        'arm_id': 'distill_strong', 'additional_epochs': 1, 'weak_reused_from_version': 1})
    print(json.dumps({'status': 'STAGED_FOR_RELEASE_CHECK', 'directory': str(destination)}))


if __name__ == '__main__':
    main()
