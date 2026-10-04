"""Question-specific declarations; shared workflow owns packaging and planning."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path

from molgap.comparison_readiness import assess_comparison_prelaunch
from molgap.experiment_execution import build_family_recipe
from molgap.experiment_spec import ExperimentSpec
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import sha256_file
from molgap.v4_runtime import normalized_source_sha256

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
REL = EXP.relative_to(ROOT).as_posix()
OLD = ROOT / 'experiments/pcqm_k1_dropout_consistency'
RUN = 'molgap-k1-fusion-distill-100k-s42-v1'
EVIDENCE = 'pcqm-k1-consistency-fusion-transfer-20261004'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True), encoding='utf-8', newline='')


def main():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    template = read(OLD / 'experiment_spec_kaggle3_v1.json')
    old_plan = read(OLD / 'training_plan_dropout_mean2.json')
    old_accept = read(OLD / 'family_acceptance_plan.json')['arms'][0]
    old_prelaunch = read(OLD / 'comparison_prelaunch_dropout_mean2.json')
    bundle = read(ROOT / old_accept['reference_bundle']['path'])
    teacher = read(EXP / 'teacher_cache/manifest.json')
    measurement = read(EXP / 'teacher_generation.json')
    assert sha256_file(EXP / 'teacher_cache/manifest.json') == measurement['manifest_sha256']
    policy = read(ROOT / 'research_memory/policies/pcqm-k1-dropout-consistency-kaggle3-100k-launch.1.json')
    policy.update(policy_id='pcqm-k1-fusion-distillation-kaggle3-100k-launch',
                  created_from_source_digest=sha256_file(EXP / 'protocol.md'),
                  comparability_selector={'scientific_contract': 'k1-fusion-distillation-100k-v1'})
    write(ROOT / 'research_memory/policies/pcqm-k1-fusion-distillation-kaggle3-100k-launch.1.json', policy)
    write(EXP / 'role_plan.json', {
        'format': 'molgap-k1-fusion-distillation-role-plan-v1',
        'train': 'source_idx [0,100000): label supervision and frozen teacher targets; qualification uses train only',
        'development': 'source_idx [100000,150000): clean live selection and retained aligned comparisons; no teacher-cache rows',
        'official_validation': 'untouched', 'test_dev': 'untouched', 'test_challenge': 'untouched',
        'authority': 'User authorized Kaggle3 100K two new distillation arms; no protected roles or baseline retraining'})
    arms, bindings, acceptance, workflow_arms = [], [], [], []
    for device, (mode, weight) in enumerate((('distill_weak', 0.1), ('distill_strong', 1.0))):
        addon = 'k1_fusion_' + mode
        config = {'weight': weight, 'teacher_identity': teacher['teacher_identity'],
                  'cache_manifest_sha256': measurement['manifest_sha256']}
        recipe = build_family_recipe(('neural_atom_k1', '2'), addon=addon, addon_config=config,
            source_idx_sha256=old_accept['expected']['source_idx_sha256'],
            target_sha256=old_accept['expected']['target_sha256'])
        recipe_path = f'{REL}/training_recipe_{mode}.json'
        write(ROOT / recipe_path, recipe)
        arm = copy.deepcopy(template['arms'][0])
        arm.update(arm_id=mode, scientific_role='candidate', addons=[{
            'name': addon, 'version': '1', 'config': config,
            'source_sha256': normalized_source_sha256(ROOT / 'src/molgap/k1_teacher_cache.py')}])
        arm['training']['recipe']['sha256'] = sha256_file(ROOT / recipe_path)
        arm['training']['objective'] = {'name': 'normalized-gap-l1-plus-frozen-teacher-mse',
            'version': '1', 'sha256': canonical_fingerprint({'label': 'normalized-gap-l1', 'teacher': 'normalized-gap-mse', **config})}
        arms.append(arm)
        tid = f'TB-k1-fusion-{mode.replace("_", "-")}-kaggle3-100k-s42-v1'
        run = RUN + ':' + mode + ':downstream'
        plan = copy.deepcopy(old_plan)
        action_path = f'{REL}/action_inputs_{mode}.json'
        write(ROOT / action_path, {'evidence_ids': [EVIDENCE], 'evidence_review_complete': 1,
                                   'state_timestamp': '2026-10-04', 'trajectory_id': tid})
        plan['action_inputs_ref'] = action_path
        state = plan['decision_state']
        state.update(policy_id=policy['policy_id'], budget_snapshot_ref=f'{REL}/protocol.md',
                     role_snapshot_refs=[f'{REL}/role_plan.json'], source_commit=commit)
        cost = plan['costs'][0]
        cost_id = f'cost-{tid}-expected-training'
        cost.update(trajectory_id=tid, cost_event_id=cost_id, attempt_id='kaggle3-distill-001',
                    run_id=run, evidence_ref=f'{REL}/protocol.md', hardware=f'Tesla T4; assigned GPU{device} in 2T4 pair')
        trajectory = plan['trajectory']
        trajectory.update(trajectory_id=tid, family_id='k1-fusion-distillation',
            question=f'Does fixed fusion teacher distillation at lambda={weight} retain ensemble benefit in one K1 student?')
        trajectory['actions'] = [{'action_id': 'A001', 'attempt_ids': ['kaggle3-distill-001'],
            'cost_event_ids': [cost_id], 'evidence_refs': [f'{REL}/protocol.md'], 'run_ids': [run],
            'source_commit': commit, 'type': 'authorized_frozen_fusion_teacher_distillation_100k'}]
        trajectory['decision'].update(decision_ref=f'{REL}/protocol.md')
        trajectory['hypothesis'] = {
            'hypothesis_id': 'H-' + tid,
            'observed_deficiency': 'Fixed fusion improves accepted separate 50K cohort by 2.812 meV versus stronger constituent, but requires two forward models.',
            'changed_mechanism': f'One unchanged clean K1 student adds lambda={weight} normalized detached teacher MSE to normalized label L1; same fixed train-only 50:50 teacher cache.',
            'alternative_explanations': ['Teacher fitting of training labels does not transfer to development.', 'Strong imitation suppresses useful student variation.', 'One-seed stochasticity explains apparent gains.'],
            'cheapest_falsifier': 'One concentrated CPU/synthetic gate and shared all-arm train-only T4 runtime qualification, then existing fixed 100K 40-epoch contract.',
            'decision_changed_if_positive': 'Review single-model compression nomination and missing strict V5 requirements; no automatic scale-up.',
            'decision_changed_if_negative': 'Close tested weights with retained curves/exposure and imitation attribution; no unplanned tuning or baseline retry.',
            'supporting_evidence_ids': [EVIDENCE], 'related_closed_family_ids': ['k1-dropout-consistency'],
            'historical_unknowns': ['Training stochasticity', 'Single student transfer of fixed fusion gains', 'Strict historical runtime equivalence'],
            'expected_native_cost_ref': cost_id}
        trajectory['state_at_start'].update(budget_snapshot_ref=f'{REL}/protocol.md',
            contract_refs=[f'{REL}/protocol.md', recipe_path, f'{REL}/evidence_review.md',
                           f'{REL}/teacher_cache/manifest.json', f'{REL}/teacher_generation.json'],
            parent_trajectory_ids=['TB-k1-consistency-fusion-transfer-20261004'], prior_evidence_ids=[EVIDENCE],
            reference_ids=[bundle['reference_id']], role_snapshot_refs=[f'{REL}/role_plan.json'],
            source_commit=commit, source_config_identity=canonical_fingerprint(arm))
        plan_path = f'{REL}/training_plan_{mode}.json'
        write(ROOT / plan_path, plan)
        bindings.append({'arm_id': mode, 'trajectory_id': tid, 'plan_spec_ref': plan_path,
                         'plan_spec_sha256': sha256_file(ROOT / plan_path), 'output': f'{REL}/kaggle3_v1/{mode}'})
        identity = copy.deepcopy(bundle['comparison_identity'])
        identity['loss_identity'] = arm['training']['objective']['sha256']
        prelaunch = assess_comparison_prelaunch(candidate_id=RUN + ':' + mode,
            candidate_plan={'comparison_identity': identity, 'source_config_status': 'frozen', 'source_commit_or_archive': commit},
            reference_id=bundle['reference_id'], reference_bundle=bundle,
            experiment_purpose='training_objective_comparison', intervention_group_id='k1-fusion-distillation-objective',
            declared_intervention_fields=['loss_identity'], role_applicability_plan=old_prelaunch['role_applicability_plan'],
            trace_plan=old_prelaunch['trace_plan'], runtime_qualification_plan={
                'status': 'declared', 'runtime_certificate_required': True,
                'qualification_scope': 'Assigned T4 train-only BS128; pinned teacher cache; shared repeatability, resume and overhead qualification'})
        prelaunch_path = f'{REL}/comparison_prelaunch_{mode}.json'
        write(ROOT / prelaunch_path, prelaunch)
        entry = copy.deepcopy(old_accept)
        entry.update(arm_id=mode, comparison_prelaunch={'path': prelaunch_path, 'sha256': sha256_file(ROOT / prelaunch_path)},
                     contract={'path': recipe_path, 'sha256': sha256_file(ROOT / recipe_path)})
        acceptance.append(entry)
        workflow_arms.append({'arm_id': mode, 'device': device, 'recipe': recipe_path,
                             'initial_state': 'D:/w/k1-dropout-consistency/experiments/pcqm_k1_dropout_consistency/initial_state.pt'})
    template.update(arms=arms, experiment_id='pcqm-k1-fusion-distillation-kaggle3-100k', logical_run_id=RUN,
                    prospective={'arms': bindings})
    spec = ExperimentSpec(template)
    write(EXP / 'experiment_spec.json', spec.to_dict())
    write(EXP / 'family_acceptance_plan.json', {'format': 'molgap-family-acceptance-plan-v1', 'spec_identity': spec.identity, 'arms': acceptance})
    write(EXP / 'workflow_plan.json', {'format': 'molgap-experiment-workflow-v1', 'spec_identity': spec.identity,
        'source_files': [], 'arms': workflow_arms, 'acceptance_plan': f'{REL}/family_acceptance_plan.json',
        'kaggle': {'account': 'nvoid912', 'kernel': 'nvoid912/' + RUN, 'title': 'MolGap K1 Fusion Distill 100K S42 V1',
                  'datasets': ['nvoid912/molgap-k1-fusion-distill-source-s42-v1', 'nvoid912/pcqm4mv2-ogb-fixed-100k-v1',
                               'nvoid912/molgap-k1-fusion-teacher-train100k-v1'],
                  'source_dataset': 'nvoid912/molgap-k1-fusion-distill-source-s42-v1', 'accelerator': 'NvidiaTeslaT4'}})
    print(json.dumps({'status': 'declarations_prepared', 'spec_identity': spec.identity, 'arms': [a['arm_id'] for a in arms]}))


if __name__ == '__main__':
    main()
