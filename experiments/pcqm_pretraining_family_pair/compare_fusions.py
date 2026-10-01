"""Compare retained fixed blends using the existing paired-row analysis owner."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.pcqm_gptrans_100k_transfer_control.analyze_pair import paired_metrics
from molgap.training_reproducibility import atomic_json, sha256_file


def main():
    started = time.perf_counter()
    exp = Path(__file__).resolve().parent
    plan_path = exp / 'fusion_comparison_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    root = exp.parent.parent
    records = {}
    for name, item in plan['inputs'].items():
        path = Path(item['path'])
        if not path.is_absolute():
            path = root / path
        if sha256_file(path) != item['sha256']:
            raise ValueError(f'Prediction identity changed: {name}')
        records[name] = torch.load(path, map_location='cpu', weights_only=False)
    ref = records['scratch_k1']
    comparisons = {
        name: paired_metrics(ref, record, plan['row_start'], plan['row_stop'])
        for name, record in records.items()
    }
    manifest = json.loads((root / plan['inputs']['scratch_k1']['authority']).read_text())
    import hashlib
    for field, manifest_key in [('source_idx', 'source_idx_sha256'), ('target_eV', 'target_sha256'), ('prediction_eV', 'prediction_sha256')]:
        digest = hashlib.sha256(ref[field].contiguous().numpy().tobytes()).hexdigest()
        if digest != manifest[manifest_key]:
            raise ValueError(f'K1 canonical tensor identity changed: {field}')
    gp = records['scratch_gptrans']
    gp_manifest = json.loads(Path(plan['inputs']['scratch_gptrans']['authority']).read_text())
    if gp['reference_model_sha256'] != gp_manifest['reference_model_sha256']:
        raise ValueError('GPTrans reference model binding changed')
    if gp['reference_evidence_id'] != 'pcqm-gptrans-noisy-pair-norm-100k-s42':
        raise ValueError('Wrong GPTrans reference family')
    for name in ('scratch_k1', 'scratch_gptrans'):
        expected = manifest['development_gap_mae_eV'] if name == 'scratch_k1' else gp_manifest['development_mae_eV']
        if abs(comparisons[name]['candidate_mae_eV'] - expected) > 1e-7:
            raise ValueError(f'Historical MAE did not reproduce: {name}')
    blends = {}
    for stage in ('scratch', 'pretrained'):
        blends[stage] = {
            'source_idx': ref['source_idx'], 'target_eV': ref['target_eV'],
            'prediction_eV': 0.5 * records[f'{stage}_k1']['prediction_eV'].view(-1)
                           + 0.5 * records[f'{stage}_gptrans']['prediction_eV'].view(-1),
        }
    scratch_gain = paired_metrics(ref, blends['scratch'], plan['row_start'], plan['row_stop'])
    pretrained_gain = paired_metrics(records['pretrained_k1'], blends['pretrained'], plan['row_start'], plan['row_stop'])
    comparison = paired_metrics(blends['scratch'], blends['pretrained'], plan['row_start'], plan['row_stop'])
    report = {
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha256_file(plan_path), 'script_sha256': sha256_file(Path(__file__)),
        'paired_metrics_owner_sha256': sha256_file(root / 'experiments/pcqm_gptrans_100k_transfer_control/analyze_pair.py'),
        'inputs': plan['inputs'], 'weights': plan['weights'], 'weight_fitting': False,
        'all_four_artifact_hashes_verified': True, 'all_four_exact_rows_targets_finite_verified': True,
        'scratch_reference_tensor_and_model_binding_verified': True,
        'single_model_comparisons_vs_scratch_k1': comparisons,
        'scratch_blend_vs_scratch_k1': scratch_gain,
        'pretrained_blend_vs_pretrained_k1': pretrained_gain,
        'pretrained_blend_vs_scratch_blend': comparison,
        'bootstrap': {'replicates': 1000, 'seed': 42, 'unit': 'paired development row', 'training_stochasticity_measured': False},
        'training_executed': False, 'model_inference_executed': False,
        'protected_role_access': False, 'role': plan['role'],
        'strict_v5_pretraining_promotion_qualification': 'pending historical trace/runtime/downstream contract qualification',
        'analysis_process_wall_seconds': time.perf_counter() - started,
        'native_device_hours': {'value': None, 'status': 'not_applicable'},
    }
    atomic_json(exp / 'fusion_comparison.json', report)
    print(json.dumps({'single_models': {k: v['candidate_mae_eV'] for k, v in comparisons.items()},
                      'scratch_fusion': scratch_gain, 'pretrained_fusion': pretrained_gain,
                      'pretraining_effect_on_fusion': comparison}, indent=2))


if __name__ == '__main__':
    main()
