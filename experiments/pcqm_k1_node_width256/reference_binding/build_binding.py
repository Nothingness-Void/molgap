"""Bind the retained Kaggle3 K1 reference without retraining or inference."""
from pathlib import Path
import argparse
import hashlib
import json

from molgap.comparison_readiness import (
    assess_comparison_prelaunch, reference_bundle_digest,
    target_transform_asset_digest, validate_comparison_prelaunch,
    validate_reference_bundle, validate_target_transform_asset,
)
from molgap.experiment_family_workflow import RunContext, inspect_output
from molgap.experiment_spec import canonical_fingerprint
from molgap.k1_screen_training import validate_runtime_preflight

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REMOTE = ROOT / 'platforms/_records/kaggle/training/pcqm_k1_width256_reference_kaggle3_v1/k1_pair/reference'
OWNER = Path('D:/w/k1-local-aux/experiments/pcqm_k1_local_mixing_clean_aux')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ref(path):
    return Path(path).relative_to(ROOT).as_posix()


def write(name, value):
    path = OUT / name
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True), encoding='utf-8')
    return ref(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-source')
    args = parser.parse_args()
    manifest = read(REMOTE / 'output_manifest.json')
    contract = read(REMOTE / 'training_contract.json')
    spec = read(OWNER / 'experiment_spec_kaggle3_accuracy_v1.json')
    arm = next(a for a in spec['arms'] if a['arm_id'] == 'reference')
    context = RunContext(**manifest['context'])
    assert context.spec_identity == canonical_fingerprint(spec)
    assert context.arm_identity == canonical_fingerprint(arm)
    acceptance = inspect_output(REMOTE, context=context, expected=contract['acceptance_requirements'])
    acceptance_ref = write('mechanical_acceptance.json', acceptance)
    assert acceptance['status'] == 'MECHANICALLY_VERIFIED', acceptance
    trace = read(REMOTE / 'canonical_trace.json')
    rows = [r for r in trace['observations'] if r['event'] == 'observation']
    fields = ['optimizer_step', 'sample_presentations', 'epoch_or_pass', 'learning_rate', 'live_train_metric', 'live_dev_metric', 'checkpoint_identity']
    assert all(all(r.get(f) is not None for f in fields) for r in rows)
    assert rows[-1]['checkpoint_identity'] == 'sha256:' + manifest['artifacts']['resume']['sha256']
    trace_plan = {f: True for f in fields} | {'ema_dev_metric': False}
    trace_ref = write('trace_manifest.json', {'format': 'molgap-k1-retained-trace-binding-v1', 'trace_ref': ref(REMOTE / 'canonical_trace.json'), 'sha256': manifest['artifacts']['trace']['sha256'], 'field_availability': trace_plan, 'epochs': len(rows), 'final_checkpoint_identity': rows[-1]['checkpoint_identity']})
    import torch
    payload = torch.load(REMOTE / 'development_predictions.pt', map_location='cpu', weights_only=True)
    prediction_manifest = {'prediction_sha256': manifest['artifacts']['predictions']['sha256'], 'source_idx_sha256': contract['acceptance_requirements']['source_idx_sha256'], 'target_sha256': contract['acceptance_requirements']['target_sha256'], 'target_encoding': 'little-endian-float32-contiguous-raw-bytes', 'ordering_semantics': 'source_idx-ascending-100000-to-149999', 'evaluation_role_identity': contract['development_role_identity'], 'row_count': len(payload['source_idx']), 'unique_source_idx': len(torch.unique(payload['source_idx']))}
    assert torch.equal(payload['source_idx'], torch.arange(100000, 150000))
    write('prediction_manifest.json', prediction_manifest)
    row_ref = write('row_manifest.json', {'training_membership': 'source_idx-0-to-99999', 'development_membership': 'source_idx-100000-to-149999', 'training_row_order_sha256': contract['row_order_fingerprint'], 'development_source_idx_sha256': prediction_manifest['source_idx_sha256'], 'dataset_manifest_sha256': arm['data']['dataset']['sha256']})
    target_ref = write('target_manifest.json', {'development_target_sha256': prediction_manifest['target_sha256'], 'target_encoding': prediction_manifest['target_encoding'], 'training_float32_target_sha256': 'df5da6a53a51d25edaacfbd72462b53a554c1ac2a667f00718a4df6308d786ce', 'target': arm['data']['target'], 'training_target_source': context.source_commit + ':src/molgap/k1_screen_training.py:_target_stats'})
    transform = {'format': 'molgap-target-transform-asset-v1', 'asset_id': 'k1-v4-fixed-train-100k-fp32-gap-transform', 'target_identity': arm['data']['target'], 'variance_convention': 'sample-standard-deviation-correction-1; accepted-fp32-asset', 'ddof': 1, 'mean': 5.3383002281188965, 'std': 1.275090217590332, 'source_row_manifest_sha256': hashlib.sha256((OUT / 'row_manifest.json').read_bytes()).hexdigest(), 'target_sha256': 'df5da6a53a51d25edaacfbd72462b53a554c1ac2a667f00718a4df6308d786ce', 'source': context.source_commit + ':src/molgap/k1_screen_training.py:_target_stats'}
    transform['asset_sha256'] = target_transform_asset_digest(transform)
    validate_target_transform_asset(transform)
    transform_ref = write('target_transform.json', transform)
    role_plan = {f: 'applicable' for f in ['training_membership', 'prediction_input', 'labels_read', 'metric_computed', 'selection_used']} | {'external_submission': 'not_applicable'}
    role_ref = write('role_history.json', {'format': 'molgap-k1-retained-role-binding-v1', 'source_spec_identity': context.spec_identity, 'source_commit': context.source_commit, 'source_training_role': 'pcqm4mv2-ogb-fixed-100k-v1:training-0-100000', 'evaluation_role_identity': contract['development_role_identity'], 'role_applicability_plan': role_plan, 'observed_role_event_kinds': sorted(f for f, status in role_plan.items() if status == 'applicable'), 'official_validation_used': False, 'test_dev_used': False, 'test_challenge_used': False, 'evidence_refs': [ref(REMOTE / 'training_contract.json'), ref(REMOTE / 'canonical_trace.json'), ref(REMOTE / 'output_manifest.json')], 'interpretation': 'Recovered original trainer/trace usage; this binding consumes no new labels.'})
    cost_ref = write('cost_records.json', {'format': 'molgap-k1-retained-native-cost-binding-v1', 'records': manifest['costs'], 'cpu_hours': {'status': 'measurement_missing', 'value': None}, 'queue_hours': {'status': 'measurement_missing', 'value': None}, 'source_ref': ref(REMOTE / 'output_manifest.json')})
    cert = read(REMOTE / 'runtime_certificate.json')
    provenance = read(REMOTE / 'runtime_provenance.json')
    assert validate_runtime_preflight(REMOTE, provenance) == cert
    write('runtime_qualification.json', {'status': 'accepted', 'scope': 'retained reference original runtime fixture/repeatability/resume/source binding', 'runtime_certificate_ref': ref(REMOTE / 'runtime_certificate.json'), 'architecture_preflight_ref': ref(REMOTE / 'architecture_preflight.json'), 'runtime_manifest_ref': ref(REMOTE / 'runtime_manifest.json'), 'runtime_provenance_ref': ref(REMOTE / 'runtime_provenance.json'), 'owner_helper': 'molgap.k1_screen_training.validate_runtime_preflight'})
    assert cert['status'] == 'accepted' and cert['calibration_checks_passed'] is True
    assert cert['provenance_sha256'] == canonical_fingerprint(provenance)
    identity = {'benchmark_identity': contract['scientific_contract'], 'dataset_identity': arm['data']['dataset']['sha256'], 'data_role_identity': canonical_fingerprint({'roles': arm['data']['roles']}), 'row_membership_identity': arm['data']['split']['sha256'], 'row_order_identity': contract['row_order_fingerprint'], 'feature_identity': arm['data']['feature_sha256'], 'target_identity': arm['data']['target'], 'seed': 42, 'precision': cert['precision'], 'tf32_enabled': cert['tf32_enabled'], 'deterministic_algorithms': cert['deterministic_algorithms'], 'physical_batch_per_device': cert['physical_batch_per_device'], 'gradient_accumulation_steps': 1, 'tail_batch_policy': cert['tail_batch_policy'], 'optimizer_identity': canonical_fingerprint({'name': 'AdamW', 'lr': .0004, 'weight_decay': .00001, 'clip': 1.0}), 'optimizer_mode': 'AdamW-default', 'optimizer_fused': False, 'schedule_identity': canonical_fingerprint({'name': 'CosineAnnealingLR', 'epochs': 40, 'eta_min': .000001}), 'loss_identity': arm['training']['objective']['sha256'], 'target_transform_identity': transform['asset_id'], 'target_transform_asset_sha256': transform['asset_sha256'], 'sample_presentations': 3998720, 'optimizer_steps': 31240, 'checkpoint_selection_identity': contract['selection_semantics'], 'evaluation_role_identity': contract['development_role_identity'], 'selection_role_identity': contract['development_role_identity'], 'architecture_config_identity': cert['architecture_sha256'], 'runtime_certificate_scope': 'kaggle3-Tesla-T4-fp32-per-arm-calibration', 'weight_semantics': 'live', 'ema_enabled': False, 'ema_decay': None, 'ema_update_frequency': 'not_applicable', 'evaluation_weight_source': 'live'}
    decision_ref = write('reference_reuse_decision.json', {'status': 'RETAINED_REFERENCE_BOUND', 'reference_source': context.to_dict(), 'mechanical_acceptance_ref': acceptance_ref, 'meaning': 'Reference reuse for planned width-only comparison; no model promotion or original-pair scientific decision.', 'training_stochasticity': 'unavailable-single-seed', 'native_cost_scope': 'producer process wall and assigned device seconds; CPU/queue unknown'})
    bundle = {'format': 'molgap-reference-bundle-v1', 'reference_bundle_id': 'k1-v4-192-kaggle3-accuracy-reference-s42-v1', 'reference_id': context.logical_run_id + ':reference', 'contract_ref': ref(REMOTE / 'training_contract.json'), 'architecture_config_identity': identity['architecture_config_identity'], 'source_commit_or_archive': context.source_commit, 'checkpoint_identity': manifest['artifacts']['selected_model']['sha256'], 'runtime_certificate_ref': ref(REMOTE / 'runtime_certificate.json'), 'prediction_manifest': prediction_manifest, 'row_manifest_ref': row_ref, 'target_manifest_ref': target_ref, 'trace_manifest_ref': trace_ref, 'role_history_ref': role_ref, 'target_transform_asset_ref': transform_ref, 'cost_records_ref': cost_ref, 'acceptance_ref': acceptance_ref, 'decision_ref': decision_ref, 'comparison_identity': identity, 'stochasticity': {'row_bootstrap_uncertainty': {'status': 'not_requested'}, 'training_stochasticity': {'status': 'unavailable', 'estimation_method': 'none', 'source': 'single-seed', 'same_contract_repeat_ids': [], 'n_repeats': 0, 'stochasticity_floor_eV': None}}}
    validate_reference_bundle(bundle)
    write('reference_bundle.json', bundle)
    write('reference_artifacts.json', {key: 'complete' for key in ['checkpoint', 'runtime_certificate', 'prediction_manifest', 'row_manifest', 'target_manifest', 'trace_manifest', 'role_history', 'target_transform_asset', 'cost_records', 'acceptance', 'decision']})
    if args.candidate_source:
        candidate = read(ROOT / 'experiments/pcqm_k1_node_width256/candidate_arm.json')
        recipe = read(ROOT / 'experiments/pcqm_k1_node_width256/training_recipe.json')
        assert candidate['data'] == arm['data']
        assert recipe['training_recipe'] == contract['training_recipe']
        assert recipe['development_role_identity'] == contract['development_role_identity']
        candidate_identity = dict(identity)
        candidate_identity['architecture_config_identity'] = canonical_fingerprint({'base': candidate['base'], 'addons': candidate['addons']})
        planned = assess_comparison_prelaunch(candidate_id='pcqm-k1-node-width256-100k-s42-v1', candidate_plan={'comparison_identity': candidate_identity, 'source_config_status': 'frozen', 'source_commit_or_archive': args.candidate_source}, reference_id=bundle['reference_id'], reference_bundle=bundle, experiment_purpose='architecture_comparison', intervention_group_id='k1-node-width', declared_intervention_fields=['architecture_config_identity'], role_applicability_plan=role_plan, trace_plan=trace_plan, runtime_qualification_plan={'status': 'declared', 'runtime_certificate_required': True, 'qualification_scope': 'width256 candidate actual T4 tuple and repeatability/resume calibration'})
        validate_comparison_prelaunch(planned)
        write('comparison_prelaunch.json', planned)
    print(json.dumps({'status': 'REFERENCE_BOUND', 'bundle_digest': reference_bundle_digest(bundle), 'mechanical_acceptance': acceptance['status'], 'candidate_prelaunch': bool(args.candidate_source)}))


if __name__ == '__main__':
    main()

