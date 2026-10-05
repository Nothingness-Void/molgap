"""Capture bounded authoritative run state and the existing immutable receipt."""
import importlib.util
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_launch import build_launch_receipt, canonical_json, write_launch_receipt
from molgap.training_reproducibility import atomic_json

HERE = Path(sys.argv[1]).resolve()
ROOT = HERE.parents[1]
PACKAGE = Path(sys.argv[2]).resolve()
spec = ExperimentSpec.from_json((HERE / 'experiment_spec_kaggle3_v1.json').read_text())
manifest = json.loads((PACKAGE / 'package_manifest.json').read_text())
submission = json.loads((HERE / 'submission_response.json').read_text())
if submission['status'] != 'submitted' or submission['reconciliation_required']:
    raise ValueError('Submission must be reconciled before observing a run')
credential_spec = importlib.util.spec_from_file_location('credentials', ROOT / 'platforms/kaggle/credential_api.py')
module = importlib.util.module_from_spec(credential_spec)
credential_spec.loader.exec_module(module)
api = module.api_for_credentials(Path('D:/下载/Key/kaggle-nv912.json'))
ref = submission['kernel']
status = api.kernels_status(ref)
observed = datetime.now(timezone.utc)
timestamp = observed.isoformat()
directory = HERE / 'platform_observations' / observed.strftime('%Y%m%dT%H%M%S%fZ')
directory.mkdir(parents=True)
status_name = str(getattr(status, 'status', None))
record = {'kernel': ref, 'observed_at_utc': timestamp, 'status': status_name,
          'failure_message': getattr(status, 'failure_message', None),
          'observed_version_number': getattr(status, 'version_number', None),
          'submitted_version_number': submission['version_number'], 'submitted_kernel_id': submission['kernel_id']}
request_class = api.kernels_output.__func__.__globals__['ApiListKernelSessionOutputRequest']
request = request_class()
request.user_name, request.kernel_slug = ref.split('/')
try:
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
    log = getattr(response, 'log', '') or ''
    (directory / 'startup_log.txt').write_text(log[-100000:], encoding='utf-8')
    record['log_characters'] = len(log)
    record['output_file_count'] = len(getattr(response, 'files', []) or [])
except Exception as error:
    record['log_observation_error_type'] = type(error).__name__
atomic_json(directory / 'status.json', record)
receipt_dir = directory / 'launch_receipt'
receipt_dir.mkdir()
binding = build_launch_receipt(spec, PACKAGE, expected_package_identity=manifest['package_identity'])['binding']
fact = lambda value: {'value': value, 'missing_reason': None}
unknown = {'value': None, 'missing_reason': 'not_reported'}
version = str(submission['version_number'])
envelope = {'format': 'molgap-platform-response', 'version': 1, 'mode': 'observed', 'outcome': 'accepted', 'conflict_kind': None, 'binding': binding, 'canonical_platform_reference': fact(ref), 'platform_version': fact(version), 'physical_runs': fact([{'run_identity': str(submission['kernel_id']) + '/v' + version, 'canonical_reference': ref, 'platform_version': fact(version), 'arm_ids': [arm['arm_id'] for arm in spec.to_dict()['arms']]}]), 'timestamp': fact(timestamp), 'monitor_paths': unknown}
receipt = build_launch_receipt(spec, PACKAGE, expected_package_identity=manifest['package_identity'], response_json=canonical_json(envelope))
write_launch_receipt(canonical_json(receipt), receipt_dir, spec, PACKAGE, expected_package_identity=manifest['package_identity'])
atomic_json(HERE / 'latest_platform_observation.json', {'snapshot': str(directory.relative_to(ROOT)), **record})
print(json.dumps(record))
