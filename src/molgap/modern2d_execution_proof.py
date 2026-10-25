"""Execution-only shadow measurement; drift is evidence, never training release."""
from pathlib import Path
import time


def compare_tensors(candidate, reference, *, atol=3e-6, rtol=2e-5):
    import torch
    result = dict(schema_equal=candidate.keys() == reference.keys(), finite=True,
                  update_equivalent=True, max_abs=0., mismatches=[])
    for name in sorted(set(candidate) | set(reference)):
        a, b = candidate.get(name), reference.get(name)
        schema = (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                  and a.shape == b.shape and a.dtype == b.dtype)
        if not schema:
            result['schema_equal'] = result['update_equivalent'] = False
            result['mismatches'].append(name)
            continue
        if a.is_floating_point() or a.is_complex():
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            result['finite'] &= finite
            delta = float((a - b).abs().max()) if a.numel() else 0.
            result['max_abs'] = max(result['max_abs'], delta)
            equal = finite and torch.allclose(a, b, atol=atol, rtol=rtol)
        else:
            equal = torch.equal(a, b)
        if not equal:
            result['update_equivalent'] = False
            result['mismatches'].append(name)
    result.update(atol=atol, rtol=rtol)
    return result


def execution_repair_requested(spec, source_root):
    from .modern2d_training_release import plan_path, REPAIR_PURPOSE, _pinned
    from .experiment_family_workflow import _json, _artifact_path
    root = Path(source_root)
    plan = _json(_artifact_path(root, plan_path(spec)))
    return _json(_pinned(root, plan['authority'])).get('purpose') == REPAIR_PURPOSE


def bind_candidate(model, family):
    if family == 'ppgt_small':
        model.forward = model.forward_bucketed


def compare_states(candidate, reference):
    from .family_training_runner import _state_fingerprint
    import torch
    def tensors(value, prefix=''):
        if isinstance(value, torch.Tensor):
            return {prefix: value}
        result = {}
        children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (tuple, list)) else ()
        for key, child in children:
            result.update(tensors(child, prefix + '/' + str(key)))
        return result
    def metadata(value):
        if isinstance(value, torch.Tensor):
            return dict(shape=list(value.shape), dtype=str(value.dtype))
        if isinstance(value, dict):
            return {str(k): metadata(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [metadata(v) for v in value]
        return value
    model = compare_tensors(candidate['model'], reference['model'])
    optimizer = compare_tensors(tensors(candidate['optimizer']), tensors(reference['optimizer']))
    exact = {key: _state_fingerprint(candidate[key]) == _state_fingerprint(reference[key])
             for key in ('cursor', 'scheduler', 'rng_state')}
    exact['optimizer_metadata'] = metadata(candidate['optimizer']) == metadata(reference['optimizer'])
    return dict(model_update=model, optimizer_update=optimizer, exact_components=exact,
        update_equivalent=model['update_equivalent'] and optimizer['update_equivalent'] and all(exact.values()),
        exact_full_state=_state_fingerprint(candidate) == _state_fingerprint(reference))


def run_reference(*, probe, data, tuning, options, output, formal_admission,
                  candidate_proof, candidate_wall_seconds, reference_mode="graphwise"):
    import torch
    from .family_training_kernel import FamilyKernel
    from .family_training_session import FamilyTrainingSession
    from .family_training_loaders import FamilyLoaders
    from .training_reproducibility import configure_fp32_determinism, atomic_torch_save, atomic_json
    from .research_memory.trace import file_digest
    began = time.perf_counter()
    configure_fp32_determinism(probe.seed)
    kernel = FamilyKernel(probe, Path(options['initial_state_path']), device='cuda',
                          formal_training=formal_admission)
    if reference_mode == "graphwise":
        kernel.model.forward = kernel.model.forward_reference
    elif reference_mode == "accepted-native":
        # Compare only the new preparation path: retain the accepted PPGT
        # bucketed mathematics and ESA's historical dense attention backend.
        bind_candidate(kernel.model, probe.family)
    else:
        raise ValueError("Unknown execution reference backend")
    reference_path = Path(output) / 'execution-reference.pt'
    def save(progress, state):
        atomic_torch_save(reference_path, state)
    session = FamilyTrainingSession(kernel, FamilyLoaders(probe, data, tuning),
        checkpoint=save, selected=lambda *args: None, pass_finished=lambda *args: None)
    session.run()
    torch.cuda.synchronize(kernel.device)
    elapsed = time.perf_counter() - began
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    candidate = torch.load(Path(output) / candidate_proof['path'], map_location='cpu', weights_only=True)
    compared = compare_states(candidate, reference)
    model = compared['model_update']
    if not model['schema_equal'] or not model['finite']:
        raise ValueError('Execution reference/candidate model state schema or finite gate failed')
    report = dict(format='molgap-modern2d-execution-repair-proof-v1',
        family=probe.family, recipe_identity=probe.identity, reference_mode=reference_mode,
        reference=dict(path=reference_path.name, sha256=file_digest(reference_path)),
        candidate=dict(path=candidate_proof['path'], sha256=candidate_proof['sha256']),
        **compared, scientific_training_released=False,
        status='UPDATE_DRIFT_RECORDED' if not compared['update_equivalent'] else 'UPDATE_PROBE_CLOSE',
        candidate_process_wall_seconds=candidate_wall_seconds,
        reference_process_wall_seconds=elapsed,
        scope='Two cold scratch updates per path, including setup/checkpoint IO; not steady-state throughput or model quality',
        long_training_equivalence='NOT_ESTABLISHED_BY_TWO_STEP_PROBE')
    atomic_json(Path(output) / 'execution_repair.json', report)
    session.close()
    return dict(path='execution_repair.json', sha256=file_digest(Path(output) / 'execution_repair.json'))


def verify_repair_proof(root, proof, probe):
    """Recompute strict model-update drift from hash-bound saved CPU states."""
    from .experiment_family_workflow import _json, _artifact_path
    from .native_proof_storage import load_proof
    from .research_memory.trace import file_digest
    root = Path(root)
    path = _artifact_path(root, 'execution_repair.json')
    if proof != dict(path=path.name, sha256=file_digest(path)):
        raise ValueError('Execution repair report binding changed')
    report = _json(path)
    if (report.get('format') != 'molgap-modern2d-execution-repair-proof-v1'
            or report.get('recipe_identity') != probe.identity
            or report.get('scientific_training_released') is not False):
        raise ValueError('Execution diagnostic cannot grant scientific training release')
    states = {}
    for role, name in (('reference', 'execution-reference.pt'), ('candidate', 'repeat-b.pt')):
        binding = report[role]
        if (type(binding) is not dict or set(binding) != {'path', 'sha256'}
                or binding['path'] != name):
            raise ValueError('Execution diagnostic state binding changed')
        # Preflight may retire exact-byte proofs into native_proofs.zip before
        # the training process rechecks them. The storage owner verifies the
        # original SHA for either representation; numerical gates stay here.
        states[role] = load_proof(root, binding)
        from .family_training_session import validate_session_resume
        if validate_session_resume(probe, states[role]).status != 'complete':
            raise ValueError('Execution diagnostic scratch endpoint is incomplete')
    observed = compare_states(states['candidate'], states['reference'])
    if (any(report.get(k) != v for k, v in observed.items())
            or not observed['model_update']['schema_equal'] or not observed['model_update']['finite']):
        raise ValueError('Saved execution model-update observation changed')
    return report
