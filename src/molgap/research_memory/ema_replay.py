"""Qualified EMA/optimizer intervention worlds; ordinary match keys stay strict."""
from copy import deepcopy
import hashlib

from molgap.comparison_readiness import validate_comparison_prelaunch, validate_comparison_readiness, reference_bundle_digest
from molgap.evidence_pointers import load_json_object
from .backtest import BASE_COMPARABILITY_FIELDS
from .paths import resolve_repo_pointer, verify_bound_artifact
from .trace import json_bytes


def ema_intervention_worlds(root, manifests, records):
    """Add a scoped reference view only after actual strict terminal acceptance.

    This is not permission to reuse an EMA candidate in a matched-EMA world.
    Both observed identities survive beside an explicitly declared intervention.
    """
    additions, replacements = [], {}
    trajectories = {t['trajectory_id']: t for _, t in records['trajectories']}
    bundles = [b for _, b in records.get('reference_bundles', [])]
    for path, ready in records.get('comparison_readiness', []):
        purpose = ready.get('experiment_purpose')
        if purpose not in {'ema_comparison', 'optimizer_comparison'} or not ready.get('strict_ready'):
            continue
        validate_comparison_readiness(ready, evidence_verifier=lambda p, s: verify_bound_artifact(root, p, s))
        expected_fields = {'ema_decay', 'checkpoint_selection_identity'} if purpose == 'ema_comparison' else {'optimizer_identity', 'optimizer_mode'}
        world_field = 'ema_semantics' if purpose == 'ema_comparison' else 'optimizer_identity'
        if ready['comparison_class'] != 'STRICT_CAUSAL' or set(ready['mismatched_fields']) != expected_fields:
            continue
        candidates = [m for m in manifests if m['comparison_role'] == 'candidate'
            and m['reference_id'] == ready['reference_id']
            and load_json_object(resolve_repo_pointer(root, m['terminal_evidence_ref']))['evidence_id'] == ready['candidate_id']]
        for candidate in candidates:
            trajectory = trajectories[candidate['trajectory_id']]
            evidence = load_json_object(resolve_repo_pointer(root, candidate['terminal_evidence_ref']))
            pointer = path.resolve().relative_to(root.resolve()).as_posix()
            if not any(a['locator'] == pointer and a.get('sha256') for a in evidence['artifacts']):
                raise ValueError('EMA replay readiness is not bound by terminal evidence')
            for a in evidence['artifacts']:
                if a['locator'] == pointer:
                    verify_bound_artifact(root, pointer, a['sha256'])
            plans = [load_json_object(resolve_repo_pointer(root, p)) for p in trajectory['state_at_start']['contract_refs']
                if p.endswith('/comparison_readiness_prelaunch.json')]
            if len(plans) != 1:
                raise ValueError('EMA replay requires one frozen intervention plan')
            planned = validate_comparison_prelaunch(plans[0])
            for key in ('experiment_purpose', 'declared_intervention_fields', 'mismatched_fields', 'matched_fields',
                        'reference_id', 'reference_bundle_id', 'reference_bundle_sha256', 'mechanism_id', 'intervention_group_id'):
                if ready[key] != planned[key]:
                    raise ValueError('EMA terminal intervention differs from prospective plan')
            bound = [b for b in bundles if b['reference_bundle_id'] == ready['reference_bundle_id']
                and reference_bundle_digest(b) == ready['reference_bundle_sha256']]
            if len(bound) != 1 or 'candidate_reference_qualification_ref' not in bound[0]:
                raise ValueError('EMA replay requires its verified accepted comparator')
            from .candidate_reference import verify_candidate_reference
            reference = verify_candidate_reference(root, bound[0]['candidate_reference_qualification_ref'], expected_bundle=bound[0])
            if any(candidate['comparability_identity'][k] != reference['comparability_identity'][k]
                   for k in (*BASE_COMPARABILITY_FIELDS, 'architecture_identity') if k != world_field):
                raise ValueError('EMA intervention has an undeclared replay-world mismatch')
            accepted = load_json_object(resolve_repo_pointer(root, ready['candidate_artifact_bindings']['acceptance']['ref']))
            arms = [a for a in accepted['arms'].values() if a['trajectory_id'] == candidate['trajectory_id']]
            expected = dict(ready['matched_fields'])
            expected.update({k: v['candidate'] for k, v in ready['mismatched_fields'].items()})
            if len(arms) != 1 or not arms[0]['accepted'] or arms[0]['comparison_identity'] != expected:
                raise ValueError('EMA replay intervention is not independently observed')
            world = {'purpose': purpose, 'reference_id': ready['reference_id'],
                'intervention': ready['mismatched_fields'], 'prelaunch_bundle_sha256': ready['reference_bundle_sha256']}
            world_id = ('qualified-ema-intervention:' if purpose == 'ema_comparison' else 'qualified-optimizer-intervention:') + hashlib.sha256(json_bytes(world)).hexdigest()
            for manifest in (candidate, reference):
                view = deepcopy(manifest)
                view['_observed_comparability_identity'] = deepcopy(manifest['comparability_identity'])
                view['_intervention_world'] = {**world, 'readiness_ref': pointer}
                view['comparability_identity'][world_field] = world_id
                view['comparability_identity']['matched_architecture_required'] = True
                if manifest is candidate:
                    replacements[(manifest['trajectory_id'], manifest['run_id'])] = view
                else:
                    additions.append(view)
    return [replacements.get((m['trajectory_id'], m['run_id']), m) if m['comparison_role'] == 'candidate' else m
            for m in manifests] + additions
