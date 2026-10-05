"""Reconcile a published private source without creating another version."""
import hashlib
import importlib.util
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'platforms/_records/local/k1_spectral_100k_night_20261006/release_v1/source_dataset'
spec = importlib.util.spec_from_file_location('credentials', ROOT / 'platforms/kaggle/credential_api.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
api = module.api_for_credentials(Path('D:/下载/Key/kaggle-nv912.json'))
ref = 'nvoid912/molgap-k1-spectral-source-s42-v1'
files = []
token = None
while True:
    response = api.dataset_list_files(ref, page_token=token, page_size=100)
    files.extend(response.files)
    token = getattr(response, 'next_page_token', None)
    if not token:
        break
remote = {f.name: f.total_bytes for f in files}
expected = {p.relative_to(SOURCE).as_posix(): p.stat().st_size for p in SOURCE.rglob('*') if p.is_file() and p.name != 'dataset-metadata.json'}
if remote != expected:
    raise ValueError({'missing': sorted(set(expected)-set(remote)), 'extra': sorted(set(remote)-set(expected)), 'size_mismatch': [k for k in expected.keys() & remote.keys() if expected[k] != remote[k]]})
output = HERE / 'published_source_verification'
output.mkdir(exist_ok=True)
hashes = {}
for name in ('source_payload.bin', 'SOURCE_COMMIT.txt', 'experiment_launch.json'):
    destination = output / name.replace('.', '_')
    destination.mkdir(exist_ok=True)
    api.dataset_download_file(ref, name, path=str(destination), force=False, quiet=True)
    received = list(destination.iterdir())
    if len(received) != 1 or not received[0].is_file():
        raise ValueError('Expected exactly one downloaded source file')
    local = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
    downloaded = hashlib.sha256(received[0].read_bytes()).hexdigest()
    if downloaded != local:
        raise ValueError(f'Published bytes differ: {name}')
    hashes[name] = downloaded
record = {'dataset': ref, 'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'authoritative_status': api.dataset_status(ref), 'file_count': len(remote), 'inventory_and_sizes_match': True, 'downloaded_sha256': hashes}
from molgap.training_reproducibility import atomic_json
atomic_json(HERE / 'source_verification.json', record)
print(json.dumps(record))
