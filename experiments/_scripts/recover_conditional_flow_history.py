"""Retain historical arm views without modifying or re-finalizing their joint run."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.research_memory.trace import canonicalize_trace, json_bytes

COMMIT = 'ea6240e18028663893587fbfa29cdb8090db9ccb'
JOINT = 'experiments/pcqm_gptrans_conditional_flow_100k'
OUTPUT = 'experiments/pcqm_gptrans_conditional_flow_history'
RAW = 'platforms/_records/kaggle/training/pcqm_gptrans_conditional_flow_s42_v2/raw/gptrans_conditional_flow'
ARMS = ('conditional_pair_readback', 'conditional_pair_recurrence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(REPO_ROOT)
    destination = root / OUTPUT
    if destination.exists():
        raise ValueError('historical output already exists; refusing to overwrite')
    files = {}
    bindings = {}
    sources = {
        'joint_prospective_snapshot.json': 'rml_finalized/prospective_snapshot.json',
        'joint_terminal_trajectory.json': 'rml_finalized/trajectory.json',
        'joint_evidence_snapshot.json': 'rml_finalized/v5_evidence.json',
        'joint_finalization_receipt.json': 'rml_finalized/finalization.json',
        'joint_acceptance.json': 'results/acceptance_v2.json',
        'joint_decision.md': 'decision.md',
        'joint_training_contract.json': 'training_contract.json',
    }
    sha = lambda data: hashlib.sha256(data).hexdigest()
    for name, relative in sources.items():
        pointer = f'{JOINT}/{relative}'
        data = subprocess.check_output(['git', 'show', f'{COMMIT}:{pointer}'], cwd=args.source_root)
        files[f'sources/{name}'] = data
        bindings[name] = {'source_commit': COMMIT, 'source_path': pointer,
                          'retained_ref': f'{OUTPUT}/sources/{name}', 'sha256': sha(data)}
    frozen = json.loads(files['sources/joint_prospective_snapshot.json'])
    terminal = json.loads(files['sources/joint_terminal_trajectory.json'])
    evidence = json.loads(files['sources/joint_evidence_snapshot.json'])
    receipt = json.loads(files['sources/joint_finalization_receipt.json'])
    acceptance = json.loads(files['sources/joint_acceptance.json'])
    run_id = 'nothingnessvoid/molgap-gptrans-conditional-flow-s42:v2'
    joint_id = frozen['trajectory_id']
    for arm in ARMS:
        raw_pointer = f'{RAW}/{arm}/training/trace.json'
        raw = (args.source_root / raw_pointer).read_bytes()
        # Fail closed on source substitution; this is a conversion input guard,
        # not a rerun of acceptance or scientific comparison.
        if sha(raw) != receipt['input_artifact_hashes'][raw_pointer]:
            raise ValueError(f'raw trace differs from immutable joint receipt: {arm}')
        accepted_arm = next(item for item in acceptance['arms'] if item['variant'] == arm)
        if terminal['decision']['outcome'] != 'NEGATIVE_UNDER_CONTRACT' or acceptance['scientific_decision'] != 'NEGATIVE_UNDER_CONTRACT':
            raise ValueError('retained terminal decision differs from this historical adapter')
        trajectory_id = f'TH-gptrans-conditional-flow-100k-s42-{arm}'
        prefix = f'{OUTPUT}/{arm}'
        files[f'{arm}/raw_trace.json'] = raw
        provenance = {
            'relationship': 'derived_from_joint_run', 'arm': arm,
            'joint_trajectory_id': joint_id, 'joint_run_id': run_id,
            'joint_evidence_id': evidence['evidence_id'],
            'joint_finalization_id': receipt['finalization_id'],
            'joint_bindings': bindings,
            'source': raw_pointer, 'sha256': sha(raw),
            'retained_ref': f'{prefix}/raw_trace.json',
            'record_mode': 'retrospective_partial', 'prospective_authority': False,
            'independent_experiment': False, 'independent_finalization': False,
            'run_identity_semantics': 'original joint run ID; arm distinguished by historical trajectory ID and arm field',
            'axis_recovery': 'epoch copied; per-epoch step/sample increments retained only in raw source, not relabeled as cumulative axes',
        }
        rows = json.loads(raw)['rows']
        observations = [{'epoch_or_pass': row['epoch'], 'learning_rate': row['learning_rate'],
                         'live_train_metric': row['train_mae_eV'], 'ema_dev_metric': row['development_mae_eV'],
                         'wall_time_seconds': row['elapsed_seconds']} for row in rows]
        def metric(role, weights):
            return {'metric': 'MAE', 'unit': 'eV', 'target': 'PCQM4Mv2 Gap',
                    'role_identity': role, 'weights': weights, 'direction': 'minimize'}
        trace = canonicalize_trace({
            'trajectory_id': trajectory_id, 'run_id': run_id,
            'metric_semantics': {'live_train_metric': metric('internal_train_100k', 'live'),
                                 'live_dev_metric': None, 'ema_dev_metric': metric('internal_development_50k', 'ema')},
            'observations': observations, 'device_time_semantics': None, 'provenance': [provenance],
        })
        files[f'{arm}/canonical_trace.json'] = json_bytes(trace)
        files[f'{arm}/provenance.json'] = json_bytes(provenance)
        historical = {
            'schema': 'molgap-trajectory-v1', 'trajectory_id': trajectory_id,
            'record_mode': 'retrospective_partial', 'track': frozen['track'], 'owner': 'historical',
            'family_id': frozen['family_id'],
            'question': f'Historical arm view of {arm} from joint trajectory {joint_id}; not an independent prospective experiment.',
            'hypothesis': {'hypothesis_id': f'H-{trajectory_id}', 'supporting_evidence_ids': [],
                           'alternative_explanations': [], 'related_closed_family_ids': [],
                           'historical_unknowns': ['No independently frozen arm-level prospective state exists.',
                                                   'No separately measured arm device cost is available.']},
            'state_at_start': {'source_commit': 'unknown_independent_arm_state_not_frozen',
                               'contract_refs': [bindings['joint_training_contract.json']['retained_ref']],
                               'reference_ids': [], 'parent_trajectory_ids': [], 'prior_evidence_ids': [],
                               'role_snapshot_refs': [], 'budget_snapshot_ref': None},
            'actions': [{'action_id': 'A001', 'type': 'derived_from_joint_run', 'source_commit': COMMIT,
                         'run_ids': [run_id], 'attempt_ids': ['t4x2-v2'], 'cost_event_ids': [],
                         'evidence_refs': [f'{prefix}/provenance.json', bindings['joint_finalization_receipt.json']['retained_ref']]}],
            'result': {'evidence_ids': [],
                       'evidence_refs': [bindings['joint_evidence_snapshot.json']['retained_ref'],
                                         bindings['joint_acceptance.json']['retained_ref'], f'{prefix}/canonical_trace.json'],
                       'provenance': provenance, 'arm_acceptance': accepted_arm},
            'decision': {'decision_ref': bindings['joint_decision.md']['retained_ref'],
                         'outcome': terminal['decision']['outcome'], 'next_allowed_actions': [], 'reopen_conditions': []},
        }
        files[f'{arm}/trajectory.json'] = json_bytes(historical)
    files['.gitattributes'] = b'* -text\n**/* -text\n'
    files['source_bindings.json'] = json_bytes(bindings)
    # All payloads are prepared before creating the new history-only directory.
    for name, data in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(json.dumps({'output': OUTPUT, 'arms': ARMS, 'files': len(files),
                      'record_mode': 'retrospective_partial', 'prospective_authority': False}, indent=2))


if __name__ == '__main__':
    main()
