"""Translate retained family results into the existing RML terminal pipeline.

This is an evidence adapter, not a trainer or a replacement scientific gate.
Strict reference qualification remains pending; terminal attempts are inconclusive.
"""
from pathlib import Path
import json

from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.research_memory.recovery import recover_trace
from molgap.research_memory.terminal_wiring import close_terminal_multi_arm


def main():
    root = Path.cwd()
    exp = Path('experiments/pcqm_pretraining_family_pair')
    raw = Path('platforms/_records/kaggle/training/pcqm_pretraining_family_pair_release3')
    review = json.loads((exp / 'result_review.json').read_text())
    contract = json.loads((exp / 'training_contract.json').read_text())
    for locator, digest in review['artifact_sha256'].items():
        if sha256_file(raw / locator) != digest:
            raise ValueError(f'Retained reviewed artifact changed: {locator}')
    if review['independent_exit_codes'] != {'k1': 0, 'gptrans_joint': 0}:
        raise ValueError('Both frozen workers must complete before this export')
    closures = []
    for family in ('k1', 'gptrans_joint'):
        prospective = exp / 'prospective_release2' / family
        trajectory = json.loads((prospective / 'trajectory.json').read_text())
        tid = trajectory['trajectory_id']
        run = trajectory['actions'][0]['run_ids'][0]
        dest = exp / 'acceptance' / family
        dest.mkdir(parents=True, exist_ok=True)
        acceptance_path = dest / 'acceptance.json'
        trace_path = dest / 'canonical_downstream_trace.json'
        eid = f'pcqm-pretraining-{family}-100k-s42-release3-terminal'
        metric = {'metric': 'Gap MAE', 'unit': 'eV', 'target': 'gap',
                  'role_identity': 'pcqm4mv2-fixed-100k:development-100000-150000',
                  'weights': 'live' if family == 'k1' else 'ema', 'direction': 'minimize'}
        training_metric = {**metric, 'unit': 'normalized_gap', 'weights': 'live',
                           'role_identity': 'pcqm4mv2-fixed-100k:online-training-0-100000'}
        dev_field = 'live_dev_metric' if family == 'k1' else 'ema_dev_metric'
        mapping = {'epoch_or_pass': 'epoch', 'live_train_metric': 'train_normalized_mae' if family == 'k1' else 'train_normalized_gap_mae',
                   dev_field: 'development_gap_mae_eV', 'wall_time_seconds': 'seconds' if family == 'k1' else 'epoch_seconds'}
        if family == 'k1':
            mapping.update(optimizer_step='optimizer_steps', sample_presentations='sample_presentations', learning_rate='learning_rate')
        recovery = {'trajectory_id': tid, 'run_id': run, 'rows_key': 'epochs',
                    'field_mapping': mapping, 'metric_semantics': {'live_train_metric': training_metric,
                    'live_dev_metric': metric if family == 'k1' else None,
                    'ema_dev_metric': metric if family != 'k1' else None}}
        if not trace_path.exists():
            recover_trace([raw / family / 'trace.json'], recovery, trace_path, repo_root=root)
        decision = {'outcome': 'INCONCLUSIVE', 'decision_ref': (exp / 'terminal_decision.md').as_posix(),
                    'next_allowed_actions': ['reconcile_retained_reference_artifacts_and_readiness_without_training'],
                    'reopen_conditions': ['Verified historical reference artifacts and strict V5 comparison; no automatic training or scale-up.']}
        outcome = {'execution_status': 'complete', 'artifact_status': 'complete_local_verified',
                   'comparison_status': 'pending_strict_reference_qualification', 'scientific_status': 'INCONCLUSIVE',
                   'transfer_status': 'not_qualified', 'budget_decision': 'no_scaleup', 'full_handoff_status': 'not_released'}
        role_use = {'train': 'consumed', 'internal_development': 'consumed',
                    'official_validation': 'untouched', 'test_dev': 'untouched', 'test_challenge': 'untouched'}
        costs = []
        for attempt, source in [('attempt-001', exp / 'attempts/release2' / family / 'native_cost_observation.json'),
                                ('attempt-002', raw / family / 'native_cost_observation.json')]:
            obs = json.loads(source.read_text())
            hours = obs['t4_process_wall_seconds'] / 3600
            costs.append({'schema': 'molgap-cost-event-v1', 'cost_event_id': f'cost-{tid}-{attempt}-observed',
                          'trajectory_id': tid, 'action_id': 'A001', 'run_id': run, 'attempt_id': attempt,
                          'category': 'infrastructure_failure' if attempt == 'attempt-001' else 'training',
                          'platform': 'kaggle1', 'hardware': 'Nvidia Tesla T4', 'evidence_ref': acceptance_path.as_posix(),
                          'measurement': {'device_hours': {'value': None, 'status': 'measurement_missing'},
                                          'wall_hours': {'value': hours, 'status': 'measured'},
                                          'cpu_hours': {'value': None, 'status': 'measurement_missing'},
                                          'queue_hours': {'value': None, 'status': 'measurement_missing'}}})
        roles = []
        for name, access, selected, digest in [('train', 'training_membership', False, 'baa5f49fbad78af4964d9ec7eaf2d6327b2d2ca1f4dcf54e2394dfff2e36d58e'),
                                             ('internal_development', 'selection_used', True, '3270b30071692d712c468fa52fa247d37f0a8e4b5444dc7c913aadb8ca0832c4')]:
            roles.append({'schema': 'molgap-role-event-v1', 'role_event_id': f'role-{tid}-{name}-release3',
                          'trajectory_id': tid, 'action_id': 'A001', 'run_id': run,
                          'dataset_identity': contract['fixed_manifest_sha256'], 'row_manifest_hash': digest,
                          'role_name': name, 'access_kind': access, 'selection_used': selected,
                          'evidence_ref': acceptance_path.as_posix()})
        acceptance = {'evidence_id': eid, 'run_id': run, 'outcome': outcome, 'trajectory_decision': decision,
                      'role_use': role_use, 'costs': costs, 'roles': roles,
                      'verified_result': review['arms'][family], 'retention_manifest': review['artifact_sha256'],
                      'role_evidence': ['Frozen source trainer, validated row membership, aligned finite predictions and checkpoint selection'],
                      'cost_semantics': 'Measured assigned-arm process wall time; GPU busy time and scheduler allocation billing not measured.',
                      'strict_reference_accepted': False, 'model_execution_during_acceptance': False}
        atomic_json(acceptance_path, acceptance)
        retained = [trace_path, raw / family / 'best_model.pt', raw / family / 'last_checkpoint.pt',
                    raw / family / 'pretraining/last_checkpoint.pt', raw / family / 'pretraining/stage_complete.json',
                    raw / family / ('best_development_payload.pt' if family == 'k1' else 'development_predictions.pt')]
        artifacts = [{'name': 'downstream_training_trace' if path == trace_path else path.relative_to(raw / family).as_posix(),
                      'locator': path.as_posix(), 'sha256': sha256_file(path), 'availability': 'local_verified'} for path in retained]
        pointers = [acceptance_path, exp / 'terminal_decision.md', exp / 'result_review.json',
                    exp / 'protocol.md', exp / 'training_contract.json', exp / 'retry_release3_binding.json',
                    exp / 'submission_readiness.json', raw / family / 'trace.json', raw / family / 'pretraining/trace.json',
                    raw / family / 'native_cost_observation.json', raw / 'pulled_source/kernel-metadata.json',
                    raw / 'worker_exits.json', exp / 'attempts/release2' / family / 'native_cost_observation.json']
        evidence = {'format': 'molgap-v5-evidence-envelope-v1', 'contract': 'MOLGAP-COMMON-V5-FINAL',
                    'evidence_id': eid, 'track': 'B', 'scope': 'two_family_pretraining_100k_terminal_attempt',
                    'legacy_contract': contract['format'], 'outcome': outcome,
                    'authority': {'pointers': [path.as_posix() for path in pointers]}, 'artifacts': artifacts,
                    'role_use': role_use, 'metrics': review['arms'][family],
                    'migration': {'migrated_at': '2026-09-30', 'verification_scope': 'Retained remote source, checkpoints, predictions and metadata; strict reference pending',
                                  'training_executed': False, 'inference_executed': False, 'scientific_reinterpretation': False}}
        terminal = {'format': 'molgap-rml-terminal-package-v1', 'trajectory_id': tid, 'run_id': run,
                    'action_id': 'A001', 'finalized_at': review['observed_at_utc'], 'acceptance_ref': acceptance_path.as_posix(),
                    'artifact_hashes': {path.as_posix(): sha256_file(path) for path in [*pointers, *retained]},
                    'evidence': evidence, 'decision': decision, 'costs': costs, 'roles': roles, 'role_use': role_use}
        terminal_path = dest / 'terminal.json'
        atomic_json(terminal_path, terminal)
        closures.append({'trajectory': (prospective / 'trajectory.json').as_posix(),
                         'terminal': terminal_path.as_posix(), 'trace': trace_path.as_posix(), 'arm_identifier': family})
    result = close_terminal_multi_arm(root, closures)
    atomic_json(exp / 'rml_terminal_report.json', {'arms': result})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
