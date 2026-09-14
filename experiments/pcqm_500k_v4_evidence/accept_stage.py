"""No-inference, fail-closed acceptance of a bounded evidence stage."""
import argparse
import json
from pathlib import Path

from molgap.pcqm_500k_v4_evidence import PARAMETERS, scientific_contract
from molgap.screen_policy import canonical_fingerprint, validate_runtime_certificate
from molgap.training_reproducibility import sha256_file


def accept(root):
    manifest = json.loads((root / 'stage_manifest.json').read_text())
    if manifest['status'] not in ('STAGE_COMPLETE', 'COMPLETE'):
        raise ValueError('Stage was not published successfully')
    if manifest['parameters'] != PARAMETERS[manifest['arm']]:
        raise ValueError('Model identity changed')
    if not 1 <= manifest['next_epoch'] <= 60:
        raise ValueError('Invalid epoch cursor')
    expected = scientific_contract()
    actual = manifest['contract']
    for key, value in expected.items():
        if key != 'target_transform_fingerprint' and actual[key] != value:
            raise ValueError(f'Contract mismatch: {key}')
    for role in ('official_validation_role_read', 'test_dev_role_read', 'test_challenge_role_read'):
        if manifest[role] is not False:
            raise ValueError(f'Sealed role accessed: {role}')
    for name in ('last_checkpoint.pt', 'best_model.pt', 'best_predictions.pt', 'initial_state.pt',
                 'trace.json', 'runtime_certificate.json', 'calibration.json'):
        if name not in manifest['artifacts']:
            raise ValueError(f'Missing required artifact: {name}')
    for name, checksum in manifest['artifacts'].items():
        path = (root / name).resolve()
        path.relative_to(root.resolve())
        if sha256_file(path) != checksum:
            raise ValueError(f'Artifact hash mismatch: {name}')
    certificate = json.loads((root / 'runtime_certificate.json').read_text())
    validate_runtime_certificate(certificate, {'platform_id': certificate['platform_id'],
        'accelerator': certificate['accelerator'], 'runtime_certificate_id': manifest['runtime_certificate_id']})
    calibration = json.loads((root / 'calibration.json').read_text())
    if len(calibration['repeats']) != 2 or calibration['repeats'][0] != calibration['repeats'][1]:
        raise ValueError('Calibration replay failed')
    trace = json.loads((root / 'trace.json').read_text())['epochs']
    if [row['epoch'] for row in trace] != list(range(manifest['next_epoch'])):
        raise ValueError('Training epochs missing or repeated')
    for row in trace:
        if row['global_step'] != (row['epoch']+1)*3906 or row['sample_presentations'] != row['global_step']*128:
            raise ValueError('Sample exposure mismatch')
    return {'accepted_stage': True, 'arm': manifest['arm'], 'next_epoch': manifest['next_epoch'],
        'training_complete': manifest['status'] == 'COMPLETE',
        'manifest_sha256': sha256_file(root / 'stage_manifest.json'),
        'contract_fingerprint': canonical_fingerprint(actual)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(accept(args.root), indent=2))
