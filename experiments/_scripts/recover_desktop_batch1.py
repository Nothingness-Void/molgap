"""Deterministic Batch 1 metadata recovery; no corpus validation or model execution.

This fixed historical adapter uses only retained metadata and local Git objects.
It never rebuilds RML. Source snapshots preserve bytes, including line endings.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.research_memory.trace import canonicalize_trace, json_bytes

ROOT = Path(REPO_ROOT)
REF = 'experiments/pcqm_gptrans_t_100k_v4'
CAND = 'experiments/pcqm_gptrans_flow_ablation_100k'
BRIDGE = 'experiments/pcqm_gptrans_t_500k'
GINE = 'experiments/pcqm_gine_expert'
REF_SOURCE = 'platforms/_records/kaggle/training/pcqm_gptrans_t_v4_reference_s42/pcqm_gptrans_t_100k_v4/training/trace.json'
CAND_SOURCE = 'platforms/_records/kaggle/training/pcqm_gptrans_flow_ablation_s42_v1/gptrans_flow_ablation/no_pair_to_node/training/trace.json'
BRIDGE_SOURCE = 'platforms/_records/scnet/pcqm_gptrans_t_core_500k_0c0108e/completion_manifest.json'
GINE_SOURCE = GINE + '/results/local_scaleup_1m_v7_frozen_bn/pcqm_gine_train_log.csv'
IMPORT_COMMIT = '76c8fd3ed713774cc155477e925fbce97b1418cb'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


def write(path, data):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f'refusing to replace different retained artifact: {path}')
    path.write_bytes(data)


def emit(path, value):
    write(path, json_bytes(value))


def snapshot(base, source, name):
    data = (ROOT / source).read_bytes()
    retained = base + '/historical_recovery/' + name
    write(retained, data)
    return {'source': source, 'sha256': digest(data), 'retained_ref': retained}


def git_semantics(base, commit, path, markers):
    raw = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)
    lines = raw.decode().splitlines()
    selected = set()
    for index, line in enumerate(lines):
        if any(marker in line for marker in markers):
            selected.update(range(max(0, index - 3), min(len(lines), index + 8)))
    binding = {'source_commit': commit, 'source_path': path, 'source_sha256': digest(raw),
               'excerpts': [{'line': i + 1, 'text': lines[i]} for i in sorted(selected)]}
    emit(base + '/historical_recovery/source_semantics.json', binding)
    return binding


def metric(role, weights):
    return {'metric': 'MAE', 'unit': 'eV', 'target': 'PCQM4Mv2 Gap',
            'role_identity': role, 'weights': weights, 'direction': 'minimize'}


def generate(base, source, rows, mapping, *, cumulative, ema, roles, semantics, bindings):
    trajectory = read(base + '/trajectory.json')
    source_binding = snapshot(base, source, 'source_trace' + Path(source).suffix)
    observations = []
    totals = {'optimizer_step': 0, 'sample_presentations': 0}
    for row in rows:
        observation = {target: row.get(column) for target, column in mapping.items()}
        for field, value in list(observation.items()):
            if value is not None and value != '':
                observation[field] = int(value) if field in {'epoch_or_pass', 'optimizer_step', 'sample_presentations'} else float(value)
            else:
                observation[field] = None
        if cumulative:
            for field in totals:
                # Only sum the recorded increments whose semantics were read
                # from the exact source commit. No epoch/size extrapolation.
                totals[field] += observation[field]
                observation[field] = totals[field]
        observations.append(observation)
    run_id = ('nothingnessvoid/molgap-gptrans-flow-ablation-s42:v1/no_pair_to_node'
              if base == CAND else trajectory['actions'][0]['run_ids'][0])
    canonical = canonicalize_trace({
        'trajectory_id': trajectory['trajectory_id'], 'run_id': run_id,
        'metric_semantics': {'live_train_metric': metric(roles[0], 'live'),
                             'live_dev_metric': None if ema else metric(roles[1], 'live'),
                             'ema_dev_metric': metric(roles[1], 'ema') if ema else None},
        'device_time_semantics': None, 'observations': observations,
        'provenance': [{**source_binding, 'field_mapping': mapping,
                        'cumulative_sum_fields': sorted(totals) if cumulative else [],
                        'source_semantics_ref': base + '/historical_recovery/source_semantics.json',
                        'source_semantics_sha256': digest(json_bytes(semantics)),
                        'record_mode': 'retrospective_partial', 'capability': 'historical_partial'}],
    })
    trace_ref = base + '/historical_recovery/canonical_trace.json'
    emit(trace_ref, canonical)
    evidence_ref = base + '/v5_evidence.json'
    observed = sorted(field for field in mapping if not (cumulative and field in totals))
    unknown = sorted(field for field, value in canonical['observations'][0].items()
                     if value is None)
    sidecar = {
        'format': 'molgap-historical-trace-recovery-v1',
        'trajectory_id': trajectory['trajectory_id'], 'run_id': run_id,
        'record_mode': 'retrospective_partial', 'capability': 'historical_partial',
        'source': source_binding, 'canonical_trace_ref': trace_ref,
        'canonical_trace_sha256': digest(json_bytes(canonical)),
        'terminal_evidence_ref': evidence_ref,
        'terminal_evidence_sha256': digest((ROOT / evidence_ref).read_bytes()),
        'authority_bindings': bindings,
        'observed_fields': observed,
        'recovered_fields': sorted(totals) if cumulative else [], 'unknown_fields': unknown,
        'scientific_reinterpretation': False, 'new_role_access': False,
        'device_time_from_wall_time': False,
        'notes': ['Sequence is serialization order, not a measured axis.',
                  'Epoch labels retain their original zero-based values.',
                  'No checkpoint identity is inferred from best_epoch or improved.',
                  'Wall intervals exclude unrecorded gaps; no cumulative wall or device time is invented.'],
    }
    emit(base + '/historical_recovery/migration.json', sidecar)
    return canonical, sidecar


def manifest(base, canonical, sidecar, identity, *, eligible, contract, reference, role):
    obs = canonical['observations']
    exposures = {key: obs[-1][field] for key, field in
                 [('optimizer_steps', 'optimizer_step'), ('sample_presentations', 'sample_presentations')]}
    emit(base + '/trace_manifest.json', {
        'schema': 'molgap-trace-manifest-v1', 'trajectory_id': canonical['trajectory_id'],
        'run_id': canonical['run_id'], 'contract_ref': contract,
        'model_identity': identity['architecture_identity'], 'reference_id': reference,
        'comparison_role': role, 'x_axis': identity['x_axis_semantics'],
        'presentation_semantics_ref': base + '/historical_recovery/migration.json',
        'weight_semantics': identity['ema_semantics'], 'metric_semantics': 'Gap MAE in eV; live training and explicitly bound development weights',
        'evaluation_role_identity': identity['evaluation_role_identity'],
        'selection_semantics': identity['selection_role_identity'],
        'trace_artifact_ref': sidecar['canonical_trace_ref'],
        'trace_artifact_sha256': sidecar['canonical_trace_sha256'],
        'terminal_evidence_ref': base + '/v5_evidence.json',
        'comparability_identity': identity, 'exposure': exposures,
        'backtest_eligibility': {'eligible': eligible, 'exclusion_reasons': [] if eligible else
                                 ['CONTEXT_ONLY', 'missing_observed_step_and_sample_axes', 'no_matched_reference_recovered']},
        'trace_fields': {f: any(row[f] is not None for row in obs) for f in
                         ['optimizer_step', 'sample_presentations', 'epoch_or_pass', 'learning_rate',
                          'live_train_metric', 'live_dev_metric', 'ema_dev_metric', 'checkpoint_identity']},
    })


def main():
    # These are imported scientific records, not a new prospective experiment.
    original = read(CAND + '/historical_recovery/original_trajectory.json')
    imported = json.loads(json.dumps(original))
    imported['record_mode'] = 'retrospective_partial'
    imported['hypothesis']['expected_native_cost_ref'] = None
    imported['hypothesis']['historical_unknowns'] = [
        'Selected-arm native cost is not separately observed; job-wide cost is not allocated to this arm.',
        'Original prospective record is retained with its source commit; this Desktop import grants no prospective authority.']
    for action in imported['actions']:
        action['cost_event_ids'] = []
    emit(CAND + '/trajectory.json', imported)
    for base in [REF, CAND, BRIDGE, GINE]:
        write(base + '/historical_recovery/.gitattributes', b'* -text\n')
    write(CAND + '/.gitattributes', b'*.json -text\nresults/*.json -text\n')

    shared = read(CAND + '/results/acceptance_v1.json')['arms'][0]['comparison']['shared']
    ref_frozen = read(str(Path(REF_SOURCE).parent / 'frozen_reference.json'))
    cand_frozen = read(str(Path(CAND_SOURCE).parent / 'frozen_reference.json'))
    # Input guards are part of conversion, not new scientific acceptance.
    for field, value in shared.items():
        if ref_frozen[field] != value or cand_frozen[field] != value:
            raise ValueError(f'retained comparison input changed: {field}')
    emit(CAND + '/historical_recovery/import_provenance.json', {
        'source_commit': IMPORT_COMMIT, 'original_owner': original['owner'],
        'original_record_mode': original['record_mode'], 'import_record_mode': 'retrospective_partial',
        'imported_files': ['training_contract.json', 'results/acceptance_v1.json', 'decision.md', 'v5_evidence.json'],
        'source_trajectory_sha256': digest((ROOT / CAND / 'historical_recovery/original_trajectory.json').read_bytes()),
        'original_outcome': original['decision']['outcome'], 'shared_scientific_fields': shared,
        'no_new_role_or_cost_events': True,
        'replay_label_limitation': 'The historical V5 scientific_status is lowercase negative_under_contract. Existing replay label mapping uses uppercase literals. Evidence is preserved; no policy score is promised.',
    })
    mapping = {'epoch_or_pass': 'epoch', 'optimizer_step': 'optimizer_steps',
               'sample_presentations': 'sample_presentations', 'learning_rate': 'learning_rate',
               'live_train_metric': 'train_mae_eV', 'ema_dev_metric': 'development_mae_eV',
               'wall_time_seconds': 'elapsed_seconds'}
    common_identity = {
        'scientific_contract': shared['benchmark_id'] + ':seed42:normalized-gap-l1',
        'dataset_identity': shared['data_role_fingerprint'], 'row_split_identity': shared['row_order_fingerprint'],
        'optimizer_identity': shared['optimizer_fingerprint'], 'lr_schedule_identity': shared['schedule_fingerprint'],
        'target_transform_identity': shared['target_transform_fingerprint'],
        'precision_identity': 'fp32:tf32-disabled:deterministic-algorithms:seed42',
        'ema_semantics': 'ema-decay-0.9999', 'evaluation_role_identity': shared['role_access_fingerprint'],
        'selection_role_identity': shared['selection_fingerprint'], 'x_axis_semantics': 'optimizer_steps',
        'terminal_endpoint_identity': 'optimizer_steps46860:sample_presentations5998080',
        'matched_architecture_required': False,
    }
    sources = []
    for base, source, frozen, role in [(REF, REF_SOURCE, ref_frozen, 'reference'), (CAND, CAND_SOURCE, cand_frozen, 'candidate')]:
        completion_path = str(Path(source).parent / 'completion_manifest.json').replace('\\', '/')
        completion = read(completion_path)
        semantics = git_semantics(base, completion['source_commit'], 'src/molgap/pcqm_gptrans_v4.py',
                                 ['def _optimizer_step', 'optimizer.step()', 'def _evaluate', 'model.load_state_dict(ema.state_dict()',
                                  'for epoch in range(start_epoch', 'for batch_index, batch in enumerate', 'train_count +=',
                                  '"optimizer_steps": BATCHES_PER_EPOCH,', 'trace = list(checkpoint["trace"])'])
        bindings = [snapshot(base, completion_path, 'completion_source.json'),
                    snapshot(base, str(Path(source).parent / 'frozen_reference.json').replace('\\', '/'), 'frozen_reference_source.json'),
                    snapshot(base, REF + '/results/kaggle_acceptance.json' if base == REF else CAND + '/results/acceptance_v1.json', 'acceptance_source.json')]
        rows = read(source)['rows']
        trace, sidecar = generate(base, source, rows, mapping, cumulative=True, ema=True,
                                  roles=('internal_train_100k', shared['role_access_fingerprint']), semantics=semantics, bindings=bindings)
        if (trace['observations'][-1]['optimizer_step'], trace['observations'][-1]['sample_presentations']) != (completion['optimizer_steps'], completion['sample_presentations']):
            raise ValueError('recorded increments do not reconcile with retained completion counters')
        identity = {**common_identity, 'architecture_identity': frozen['architecture_fingerprint']}
        manifest(base, trace, sidecar, identity, eligible=True, contract=base + '/training_contract.json',
                 reference='pcqm-gptrans-t-100k-v4-reference', role=role)
        sources.append(sidecar)

    semantics = git_semantics(BRIDGE, '0c0108e', 'src/molgap/pcqm_gptrans_scale.py',
                              ['def _evaluate', 'model.load_state_dict(ema.state_dict()', '"train_mae_eV":', '"elapsed_s": elapsed'])
    bindings = [snapshot(BRIDGE, BRIDGE + '/results/acceptance.json', 'acceptance_source.json')]
    bridge_mapping = {k: v for k, v in mapping.items() if k not in {'optimizer_step', 'sample_presentations'}}
    bridge_mapping['wall_time_seconds'] = 'elapsed_s'
    bridge = read(BRIDGE_SOURCE)
    trace, sidecar = generate(BRIDGE, BRIDGE_SOURCE, bridge['trace'], bridge_mapping, cumulative=False, ema=True,
                              roles=('internal_train_500k', 'internal_development_500000_549999'), semantics=semantics, bindings=bindings)
    identity = {k: 'unknown_retrospective' for k in common_identity}
    identity.update(scientific_contract='historical_gptrans_500k_context_only', dataset_identity=bridge['cache_aggregate_sha256'],
                    row_split_identity='train[0:500000]-development[500000:550000]', architecture_identity='gptrans-t-core-12x256-pair32',
                    ema_semantics='ema-decay-0.9999', evaluation_role_identity='internal_development_500000_549999',
                    selection_role_identity='best-development-ema', x_axis_semantics='epochs',
                    terminal_endpoint_identity='60-recorded-epochs:step-and-sample-exposure-unknown', matched_architecture_required=True)
    manifest(BRIDGE, trace, sidecar, identity, eligible=False, contract=BRIDGE + '/protocol.md',
             reference='pcqm-gptrans-t-500k-bridge', role='reference')
    sources.append(sidecar)

    semantics = git_semantics(GINE, 'de05855a45507c7330e6d3225731732ac6bff140', 'src/molgap/pcqm_expert.py',
                              ['"dev_mae_eV": f', '"train_mae_eV": f', '"elapsed_seconds": f', 'model, cache_dir, "dev", device'])
    semantics_note = 'Retained package implementation documents metric meaning; exact original GINE source commit is unknown. No exposure or precision is inferred.'
    bindings = [snapshot(GINE, GINE + '/results/local_scaleup_1m_v7_frozen_bn/completion_manifest.json', 'completion_source.json'),
                snapshot(GINE, GINE + '/results/local_scaleup_1m_v7_frozen_bn/metrics.json', 'metrics_source.json')]
    rows = list(csv.DictReader(io.StringIO((ROOT / GINE_SOURCE).read_text(encoding='utf-8-sig'))))
    gine_mapping = {'epoch_or_pass': 'epoch', 'learning_rate': 'learning_rate', 'live_train_metric': 'train_mae_eV',
                    'live_dev_metric': 'dev_mae_eV', 'wall_time_seconds': 'elapsed_seconds'}
    trace, sidecar = generate(GINE, GINE_SOURCE, rows, gine_mapping, cumulative=False, ema=False,
                              roles=('gine_v7_scaffold_train_917746', 'gine_v7_scaffold_dev_82240'), semantics=semantics, bindings=bindings)
    identity = {k: 'unknown_retrospective' for k in common_identity}
    identity.update(scientific_contract='gine_1m_v7_context_only', dataset_identity='pcqm4mv2-local-expanded-1m-v7',
                    architecture_identity='gine-v7-frozen-bn', ema_semantics='live', evaluation_role_identity='gine_v7_scaffold_dev_82240',
                    selection_role_identity='best-scaffold-dev-mae', x_axis_semantics='epochs',
                    terminal_endpoint_identity='retained-v7-log:step-and-sample-exposure-unknown', matched_architecture_required=True)
    manifest(GINE, trace, sidecar, identity, eligible=False,
             contract=GINE + '/results/local_scaleup_1m_v7_frozen_bn/input_contract.json', reference='pcqm-gine-1m-local-specialist-v7', role='reference')
    sources.append(sidecar)
    emit(REF + '/historical_recovery/batch1_report.json', {
        'format': 'molgap-historical-recovery-batch1-v1', 'starting_head': 'de05855a45507c7330e6d3225731732ac6bff140',
        'sources': [{'trajectory_id': s['trajectory_id'], **s['source'],
                     'observed': s['observed_fields'], 'recovered': s['recovered_fields'], 'unknown': s['unknown_fields']} for s in sources],
        'intended_comparable_pair': ['TB-gptrans-t-100k-v4-reference', 'TC-gptrans-flow-ablation-100k-s42'],
        'rebuild_status': 'NOT_RUN', 'acceptance_status': 'NOT_RUN', 'tests': 'NOT_RUN',
        'gine_source_identity_limit': semantics_note,
        'cost_role_policy': 'No new events; no allocation of aggregate two-arm job cost to selected arm; no wall-to-device conversion.',
        'excluded_candidate': {'name': 'pcqm_k1_induced_pair_token', 'reason': 'different K1 optimizer/schedule/selection/exposure contract'},
    })
    print(json.dumps([{'trajectory_id': s['trajectory_id'], 'source': s['source']} for s in sources], indent=2))


if __name__ == '__main__':
    main()
