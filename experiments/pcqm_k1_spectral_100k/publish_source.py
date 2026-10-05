"""Publish the verified frozen source once using the Kaggle account adapter."""
import hashlib
import importlib.util
import json
from pathlib import Path
from datetime import datetime, timezone
from molgap.kaggle_accelerator_push import _verify_release_report
from molgap.training_reproducibility import atomic_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'platforms/_records/local/k1_spectral_100k_night_20261006/release_v1'
SOURCE = BASE / 'source_dataset'
if (HERE / 'source_publication.json').exists():
    raise FileExistsError('Reconcile existing publication rather than create again')
binding = _verify_release_report(BASE / 'release_report.json', BASE / 'kernel/run.py')
pins = {'spectral_cache/spectral_cache.pt': '016c1fecad6461a7be3a66a12c9d33d11cd5049bfe0e487707c19ead302fe1a7', 'spectral_cache/spectral_cache_manifest.json': 'd428fe08e6b715e4357c7ebd6db31b5ecaada740e408cabcbaef7c91987588fd'}
for name, expected in pins.items():
    actual = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'Staged spectral pin mismatch: {name}')
spec = importlib.util.spec_from_file_location('credentials', ROOT / 'platforms/kaggle/credential_api.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
credentials = Path('D:/下载/Key/kaggle-nv912.json')
if json.loads(credentials.read_text())['username'] != 'nvoid912':
    raise ValueError('Wrong account')
api = module.api_for_credentials(credentials)
metadata = json.loads((SOURCE / 'dataset-metadata.json').read_text())
if metadata['id'] != 'nvoid912/molgap-k1-spectral-source-s42-v1':
    raise ValueError('Unexpected dataset')
response = api.dataset_create_new(str(SOURCE), public=False, quiet=True, convert_to_csv=False, dir_mode='zip')
record = {'account': 'nvoid912', 'dataset': metadata['id'], 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'response_url': getattr(response, 'url', None), 'response_error': getattr(response, 'error', None), 'response_type': type(response).__name__, 'private_requested': True, 'release_binding': binding, 'staged_spectral_sha256': pins}
atomic_json(HERE / 'source_publication.json', record)
print(json.dumps(record))
