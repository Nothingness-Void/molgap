"""Translate shared-family retained endpoints into existing frozen K1 diagnostics."""
from pathlib import Path
import json
import os
import sys
import time


def run(experiment, output):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    started, cpu_started = time.perf_counter(), time.process_time()
    experiment, output = Path(experiment), Path(output)
    inputs = json.loads((experiment / 'inputs.json').read_text(encoding='utf-8'))
    trajectory = json.loads((experiment / 'rml/trajectory.json').read_text(encoding='utf-8'))
    if trajectory['record_mode'] != 'prospective' or trajectory['decision']['outcome'] != 'ACTIVE':
        raise ValueError('Active prospective required')
    import molgap
    frozen = Path(inputs['frozen_source_root'])
    if any('molgap.' + n in sys.modules for n in ('qm9_neural_atom', 'k1_screen_training')):
        raise ValueError('Model import preceded frozen binding')
    molgap.__path__.insert(0, str(frozen / 'src/molgap'))
    import torch
    from torch_geometric.loader import DataLoader
    from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file, configure_fp32_determinism, build_runtime_manifest
    from .v4_runtime import state_dict_sha256, normalized_source_sha256
    from .qm9_neural_atom import make_encoder
    from .k1_frozen_inference import predict_clean
    from .k1_bn_calibration import recalibrated_batch_norm
    from .k1_screen_training import FORBIDDEN_MODEL_FIELDS
    from .router import paired_bootstrap_mean

    if trajectory['state_at_start']['source_config_identity'] != sha256_file(experiment / 'inputs.json'):
        raise ValueError('Prospective input hash differs')
    for name, digest in inputs['executed_source_files'].items():
        if sha256_file(experiment.parents[1] / name) != digest:
            raise ValueError('Diagnostic source changed: ' + name)
    inventory = json.loads(Path(inputs['source_inventory']['path']).read_text())
    pins = {x['path']: x['sha256'] for x in inventory['files']}
    def verify_imports():
        for name, module in tuple(sys.modules.items()):
            if not name.startswith('molgap.') or name == __name__:
                continue
            path = Path(module.__file__).resolve()
            relative = path.relative_to(frozen.resolve()).as_posix()
            if pins.get(relative) != normalized_source_sha256(path):
                raise ValueError('Unbound imported model/helper: ' + name)
    def load(item, graphs=False):
        path = Path(item['path'])
        if sha256_file(path) != item['sha256']:
            raise ValueError('Artifact hash differs: ' + str(path))
        return torch.load(path, map_location='cpu', weights_only=not graphs)

    settings = configure_fp32_determinism(42)
    torch.set_num_threads(inputs['cpu_threads'])
    deadline = min(started + inputs['ceiling_seconds'],
                   started + max(0, inputs['absolute_deadline_unix'] - time.time()))
    train, dev = load(inputs['artifacts']['train_probe'], True), load(inputs['artifacts']['development_probe'], True)
    ids = {r: torch.tensor(inputs[r + '_source_idx'], dtype=torch.long) for r in ('train', 'development')}
    if len(ids['train']) != 16384 or len(ids['train'].unique()) != 16384 or not bool(((ids['train'] >= 0) & (ids['train'] < 500000)).all()) or not torch.equal(ids['development'], torch.arange(500000, 550000)):
        raise ValueError('Role membership differs')
    for role, graphs in [('train', train), ('development', dev)]:
        if [int(g.source_idx) for g in graphs] != ids[role].tolist() or any(any(f in g for f in FORBIDDEN_MODEL_FIELDS) for g in graphs):
            raise ValueError('Graph identity/geometry differs')
    verify_imports()
    output.mkdir(exist_ok=False)
    loaders = {r: DataLoader(g, batch_size=128, shuffle=False, num_workers=0) for r, g in [('train', train), ('development', dev)]}
    report = {'format': 'molgap-k1-retained-tail-diagnostic-v1', 'status': 'running',
              'runtime': {'device': 'cpu', 'cpu_threads': torch.get_num_threads(), 'software': build_runtime_manifest(settings)}, 'cases': {}}
    targets, predictions, errors = {}, {}, {}
    def progress():
        report.update(wall_seconds=time.perf_counter() - started, process_cpu_seconds=time.process_time() - cpu_started)
        atomic_json(output / 'progress.json', report)
    def predict(model, case, role):
        value, timing = predict_clean(model, loaders[role], mean=inputs['mean'], std=inputs['std'], device='cpu', deadline=deadline)
        if not torch.equal(value['source_idx'], ids[role]) or any(v.ndim != 1 or len(v) != len(ids[role]) or not torch.isfinite(v).all() for v in value.values()):
            raise ValueError('Invalid prediction identity or values')
        if role in targets and not torch.equal(value['target_eV'], targets[role]):
            raise ValueError('Target identity differs')
        targets[role] = value['target_eV']
        errors[case, role] = (value['prediction_eV'].double() - value['target_eV'].double()).abs().numpy()
        predictions[case, role] = value
        atomic_torch_save(output / (case + '_' + role + '.pt'), value)
        report['cases'].setdefault(case, {})[role] = dict(timing, mae_eV=float(errors[case, role].mean()))
        progress()
        return value

    for name, endpoint in inputs['endpoints'].items():
        payload = load(endpoint['checkpoint'])
        state = payload[endpoint['state_key']]
        if endpoint.get('context') is not None and payload.get('context') != endpoint['context']:
            raise ValueError('Endpoint RunContext differs')
        model = make_encoder('neural_atom_k1').eval().requires_grad_(False)
        model.load_state_dict(state, strict=True)
        if sum(p.numel() for p in model.parameters()) != 3658817:
            raise ValueError('Architecture changed')
        before = state_dict_sha256(model.state_dict())
        if endpoint.get('state_sha256') and before != endpoint['state_sha256']:
            raise ValueError('Initial tensor state differs')
        verify_imports()
        case = name + '_original'
        actual = predict(model, case, 'development')
        guard = {'mae_difference_eV': abs(report['cases'][case]['development']['mae_eV'] - endpoint['mae_eV'])}
        if endpoint.get('predictions'):
            expected = load(endpoint['predictions'])
            if 'reference_development' in expected:
                expected = expected['reference_development']
            if not torch.equal(actual['source_idx'], expected['source_idx']) or not torch.equal(actual['target_eV'], expected['target_eV']):
                raise ValueError('Retained prediction rows/targets differ')
            guard['max_abs_difference_eV'] = float((actual['prediction_eV'] - expected['prediction_eV']).abs().max())
            if guard['max_abs_difference_eV'] > 1e-4:
                raise ValueError('Local original predictions do not reproduce retained output')
        if guard['mae_difference_eV'] > 1e-6:
            raise ValueError('Local original MAE does not reproduce retained output')
        report['cases'][case]['reproduction_guard'] = guard
        predict(model, case, 'train')
        with recalibrated_batch_norm(model, loaders['train'], source_idx=ids['train'], source_bounds=(0, 500000), device='cpu', deadline=deadline) as calibration:
            clean = name + '_calibrated'
            for role in loaders:
                predict(model, clean, role)
            atomic_torch_save(output / (clean + '_buffers.pt'), {k: v.detach().clone() for k, v in model.named_buffers()})
        if state_dict_sha256(model.state_dict()) != before or not calibration['buffers_restored']:
            raise ValueError('Original state not restored')
        report['cases'][clean]['calibration'] = calibration
        del model, payload
    contrasts = {}
    pairs = [(name + '_original', name + '_calibrated') for name in inputs['endpoints']]
    for state in ('original', 'calibrated'):
        pairs += [('initial_' + state, name + '_' + state) for name in inputs['endpoints'] if name != 'initial']
        pairs += [(arm + '_selected_' + state, arm + '_final_' + state) for arm in ('tail_wd1e3', 'tail_wd1e2')]
        pairs += [('tail_wd1e3_' + kind + '_' + state, 'tail_wd1e2_' + kind + '_' + state) for kind in ('selected', 'final')]
    for role in loaders:
        for first, second in pairs:
            if time.perf_counter() >= deadline:
                raise TimeoutError('Analysis deadline exceeded')
            contrasts[role + ':' + first + '->' + second] = paired_bootstrap_mean(errors[second, role] - errors[first, role], n_bootstrap=1000, seed=42)
    report.update(status='complete', contrasts=contrasts, contrast_sign='second-minus-first error; positive is worse',
                  inputs_sha256=sha256_file(experiment / 'inputs.json'), prospective_sha256=sha256_file(experiment / 'rml/trajectory.json'),
                  training_executed=False, gradients_computed=False, optimizer_created=False,
                  checks=dict(frozen_source=True, artifact_hashes=True, rows_targets=True, finite_predictions=True,
                              original_reproduction=True, parameters_unchanged=True, full_state_restored=True, protected_roles_untouched=True))
    verify_imports()
    progress()
    atomic_json(output / 'result.json', report)
    atomic_json(output / 'completion.json', {'complete': True, 'inputs_sha256': report['inputs_sha256'],
                'files': {p.name: sha256_file(p) for p in output.iterdir() if p.suffix in ('.pt', '.json')}})
    return report
