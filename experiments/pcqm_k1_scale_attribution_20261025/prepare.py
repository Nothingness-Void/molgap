"""Bind retained exact inputs and publish through the existing RML planner."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import tarfile
import zipfile
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.experiment_package import verify_experiment_source_package
from molgap.research_memory.plan import plan

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = HERE.relative_to(ROOT).as_posix()
POLICY = 'pcqm-k1-scale-attribution-20261025'
TID = 'TB-k1-scale-attribution-20261025'


def main():
    if (HERE / 'rml').exists():
        raise FileExistsError('Prospective already published')
    artifacts = Path('D:/molgap-artifacts/diagnostics/k1-scale-attribution-20261025')
    artifacts.mkdir(parents=True, exist_ok=False)
    archive = Path('D:/下载/k1-bn-mechanism-a100-20261008.zip')
    original = json.loads((ROOT / 'experiments/pcqm_k1_complete_local_audit/inputs.json').read_text())
    bind = lambda p: {'path': str(p.resolve()), 'sha256': sha256_file(p)}
    probe = artifacts / 'probes'
    probe.mkdir()
    with zipfile.ZipFile(archive) as z:
        for role in ('train_probe', 'development_probe'):
            name = role + '.pt'
            p = probe / name
            with z.open(name) as src, p.open('wb') as out:
                import shutil
                shutil.copyfileobj(src, out)
            if sha256_file(p) != original['artifacts'][role]['sha256']:
                raise ValueError('Original probe hash differs: ' + name)
    package = Path('D:/molgap-artifacts/runs/pcqm-k1-tail-regularization-500k-s42/v1/prepared/package')
    verified = verify_experiment_source_package(package)
    atomic_json(HERE / 'package_verification.json', verified)
    frozen = artifacts / 'frozen_source'
    frozen.mkdir()
    with tarfile.open(package / 'source.tar.gz', 'r:gz') as tar:
        for m in tar.getmembers():
            if not m.name.startswith('src/'):
                continue
            p = (frozen / m.name).resolve()
            if not m.isfile() or not p.is_relative_to(frozen.resolve()) or m.linkname:
                raise ValueError('Unsafe source member')
            p.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src, p.open('wb') as out:
                shutil.copyfileobj(src, out)
    initial = Path('D:/molgap-artifacts/initializations/k1-tail-regularization-500k/selected_epoch49.pt')
    endpoints = {'initial': {'checkpoint': bind(initial), 'predictions': bind(initial), 'state_key': 'model_state',
        'state_sha256': 'dfe681a620ffd248b584fec44ccc29fdad38654407a44753b7830d4f33ef2e20', 'mae_eV': 0.10490408539772034}}
    source = Path('D:/molgap-artifacts/retrieved/pcqm-k1-tail-regularization-500k-s42/v1/experiment')
    for arm, selected, final in [('tail_wd1e3', .10582958906888962, .10644728690385818), ('tail_wd1e2', .10561738163232803, .10646752268075943)]:
        manifest = json.loads((source / arm / 'output_manifest.json').read_text())
        for kind, filename, mae in [('selected', 'selected_model.pt', selected), ('final', 'last_checkpoint.pt', final)]:
            endpoints[arm + '_' + kind] = {'checkpoint': bind(source / arm / filename), 'state_key': 'model', 'mae_eV': mae, 'context': manifest['context']}
            if kind == 'selected':
                endpoints[arm + '_' + kind]['predictions'] = bind(source / arm / 'development_predictions.pt')
    files = [REL + '/' + n for n in ['prepare.py', 'run.py', 'protocol.md']]
    files += ['src/molgap/k1_retained_tail_diagnostic.py', 'src/molgap/frozen_diagnostic_closure.py']
    inputs = {'artifacts': {role: bind(probe / (role + '.pt')) for role in ['train_probe', 'development_probe']},
        'probe_archive': bind(archive), 'source_archive': bind(package / 'source.tar.gz'), 'source_inventory': bind(package / 'SOURCE_FILES.json'),
        'frozen_source_root': str(frozen), 'endpoints': endpoints, 'train_source_idx': original['train_source_idx'],
        'development_source_idx': original['development_source_idx'], 'mean': 5.413434982299805, 'std': 1.17672860622406,
        'cpu_threads': 4, 'ceiling_seconds': 1200, 'absolute_deadline_unix': 1792914095,
        'executed_source_files': {n: sha256_file(ROOT / n) for n in files}}
    atomic_json(HERE / 'inputs.json', inputs)
    atomic_json(HERE / 'roles.json', {'train_rows': len(inputs['train_source_idx']), 'train_order_owner': REL + '/inputs.json',
        'train_bounds': [0, 500000], 'development_bounds': [500000, 550000], 'development_previously_selection_used': True, 'protected_roles': 'untouched'})
    # Reuse the established planning schema; replace every experiment identity.
    spec = json.loads((ROOT / 'experiments/pcqm_k1_clean_fit_generalization_500k/plan_input.json').read_text())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    run, cost_id = 'local-k1-scale-attribution-20261025', 'cost-k1-scale-attribution-expected'
    refs = ['pcqm-k1-clean-fit-generalization-500k-20261009', 'pcqm-k1-late-weight-average-20261008']
    t = spec['trajectory']
    t.update(trajectory_id=TID, family_id='k1-scale-attribution', question='Do matched clean TRAIN and consumed development separate tail weight effects from BN state effects, and which prior100K gains actually transferred?')
    t['hypothesis'].update(hypothesis_id='H-k1-scale-attribution-20261025', observed_deficiency='Both selected tail recipes regress; direct clean-fit and BN attribution remained unexecuted.',
        supporting_evidence_ids=refs, alternative_explanations=['Tail regression disappears after BN repair', 'Clean TRAIN improves while development worsens', 'Both clean TRAIN and development worsen'],
        changed_mechanism='Independent TRAIN-only BN recalibration of five frozen LIVE states; no learned weight change', cheapest_falsifier='Exact retained16384TRAIN and50000development CPU inference using frozen source',
        expected_native_cost_ref=cost_id, related_closed_family_ids=['k1-clean-fit-generalization', 'k1-tail-regularization'])
    t['state_at_start'].update(source_commit=commit, source_config_identity=sha256_file(HERE / 'inputs.json'),
        contract_refs=[REL + '/protocol.md', REL + '/inputs.json'], prior_evidence_ids=refs, role_snapshot_refs=[REL + '/roles.json'], budget_snapshot_ref=REL + '/protocol.md')
    t['actions']=[dict(action_id='A001', type='authorized_local_frozen_attribution', run_ids=[run], attempt_ids=['attempt-001'], source_commit=commit, evidence_refs=[REL + '/inputs.json'], cost_event_ids=[cost_id])]
    t['decision'].update(decision_ref=REL + '/decision.md', next_allowed_actions=['One bounded fixed CPU matrix thenNO_TRAIN closure'])
    t['comparison_readiness_ref']=REL + '/protocol.md'
    c=spec['costs'][0]; c.update(cost_event_id=cost_id, trajectory_id=TID, run_id=run, evidence_ref=REL + '/protocol.md')
    spec['decision_state'].update(policy_id=POLICY, state_timestamp=datetime.now(timezone.utc).isoformat(), budget_snapshot_ref=REL + '/protocol.md', role_snapshot_refs=[REL + '/roles.json'], source_commit=commit)
    policy=json.loads((ROOT / 'research_memory/policies/pcqm-k1-clean-fit-generalization-500k.1.json').read_text())
    policy.update(policy_id=POLICY, comparability_selector={'scientific_contract': POLICY + '-v1'}, created_from_source_digest=sha256_file(HERE / 'protocol.md'))
    atomic_json(ROOT / ('research_memory/policies/' + POLICY + '.1.json'), policy)
    atomic_json(HERE / 'plan_input.json', spec)
    atomic_json(HERE / 'plan_receipt.json', plan(ROOT, spec, REL + '/rml'))
    print('PROSPECTIVE_PUBLISHED_NO_INFERENCE')


if __name__ == '__main__':
    main()
