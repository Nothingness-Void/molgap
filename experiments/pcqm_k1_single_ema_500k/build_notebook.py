"""Thin private-Drive adapter; model/training/acceptance stay with shared owners."""
import ast
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def bootstrap_acceptance():
    """Reuse the CPU dataset checker from the existing Colab bootstrap."""
    tree = ast.parse((ROOT / "platforms/colab/v5_runtime/build_notebook.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("# 4. Optional fixed-data staging"):
                functions = node.value.split("\nif STAGE_FIXED_DATA:", 1)[0]
                return "def sha256_file" + functions.split("def sha256_file", 1)[1]
    raise ValueError("Colab bootstrap acceptance owner changed")


def cells(binding):
    digest = binding["payload_sha256"]
    name = binding["payload_name"]
    cpu = '''# CPU only: existing immutable cache; no credentials or model execution.
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
DRIVE_ROOT = Path('/content/drive/MyDrive/MolGap')
FIXED_DATA_ROLE = '500k'
EXPECTED = {'ref': 'nvoid912/pcqm4mv2-ogb-fixed-500k-scnet-v1',
 'manifest_sha256': '630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751',
 'train_rows': 500000, 'development_rows': 50000}
'''
    cpu += bootstrap_acceptance()
    cpu += '''
CACHE = DRIVE_ROOT/'V5/fixed_data/500k'
acceptance = accept_fixed_dataset(CACHE, EXPECTED)
receipt = DRIVE_ROOT/'V5/records/k1-single-ema-500k-20261010-cpu-acceptance.json'
receipt.parent.mkdir(parents=True, exist_ok=True)
temporary = receipt.with_suffix('.tmp')
temporary.write_text(json.dumps(acceptance, indent=2))
os.replace(temporary, receipt)
print('CPU_FIXED500K_ACCEPTED', acceptance, flush=True)
'''
    upload = f'''# CPU only: upload the prepared payload ZIP, not data or keys.
from google.colab import files
uploaded = files.upload()
assert set(uploaded) == {{'{name}'}}
raw = uploaded['{name}']
assert hashlib.sha256(raw).hexdigest() == '{digest}'
destination = DRIVE_ROOT/'V5/packages/{name}'
destination.parent.mkdir(parents=True, exist_ok=True)
assert not destination.exists(), 'Reconcile existing payload before replacing anything.'
temporary = destination.with_suffix('.tmp')
temporary.write_bytes(raw)
os.replace(temporary, destination)
del uploaded, raw
print('DURABLE_SOURCE_PAYLOAD_ACCEPTED', destination, flush=True)
'''
    gpu = f'''# Run only after CPU acceptance and an explicitly selected A100 runtime.
# Includes setup/preflight/calibration/evaluation in the original four-hour window.
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
from threading import Timer
from google.colab import drive, runtime
ALLOCATION_STARTED_UNIX = ALLOCATION_START_PLACEHOLDER
DEADLINE = ALLOCATION_STARTED_UNIX + 14400
process = None
def release_at_ceiling():
    if process is not None and process.poll() is None:
        process.kill()
    runtime.unassign()
watchdog = Timer(max(0, DEADLINE-time.time()-15), release_at_ceiling)
watchdog.daemon = True
watchdog.start()
try:
    assert time.time() < DEADLINE-300, 'Allocation preparation exhausted budget.'
    drive.mount('/content/drive')
    base = Path('/content/drive/MyDrive/MolGap')
    archive = base/'V5/packages/{name}'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == '{digest}'
    payload = Path('/content/k1-single-ema-payload')
    payload.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            assert (payload/member.filename).resolve().is_relative_to(payload.resolve())
        handle.extractall(payload)
    retained = json.loads((payload/'payload_manifest.json').read_text())
    for relative, expected in retained['files'].items():
        assert hashlib.sha256((payload/relative).read_bytes()).hexdigest() == expected, relative
    binding = json.loads((payload/'package_binding.json').read_text())
    cache = base/'V5/fixed_data/500k'
    receipt = json.loads((base/'V5/records/k1-single-ema-500k-20261010-cpu-acceptance.json').read_text())
    assert receipt['status'] == 'accepted' and receipt['manifest_sha256'] == binding['manifest_sha256']
    run = base/'V5/runs/k1-single-ema-500k-a100-20261010/attempt-001'
    assert not run.exists(), 'Reconcile existing attempt; no automatic rerun.'
    run.mkdir(parents=True)
    shutil.copytree(payload, run/'frozen_payload')
    # Retained isolated runtime; no replacement of the notebook's own kernel.
    def bounded(command, ceiling):
        remaining = DEADLINE-time.time()-180
        assert remaining > 0
        subprocess.run(command, check=True, timeout=min(ceiling, remaining))
    bounded([sys.executable,'-m','pip','install','uv==0.10.9'], 180)
    envdir = Path('/content/k1-single-ema-python311')
    bounded(['uv','venv','--python','3.11',str(envdir)], 180)
    python = str(envdir/'bin/python')
    bounded(['uv','pip','install','--python',python,'torch==2.4.1',
        '--index-url','https://download.pytorch.org/whl/cu121'], 480)
    bounded(['uv','pip','install','--python',python,'numpy==1.26.4',
        'torch-geometric==2.6.1','ogb==1.3.6'], 300)
    source = Path('/content/k1-single-ema-source')
    source.mkdir(exist_ok=True)
    import tarfile
    with tarfile.open(payload/'source/source.tar.gz') as handle:
        for member in handle.getmembers():
            assert member.isfile() or member.isdir()
            assert (source/member.name).resolve().is_relative_to(source.resolve())
        handle.extractall(source)
    inventory = json.loads((payload/'source/SOURCE_FILES.json').read_text())
    assert inventory['source_commit'] == binding['source']['source_commit']
    for entry in inventory['files']:
        assert hashlib.sha256((source/entry['path']).read_bytes()).hexdigest() == entry['sha256']
    config = {{'format': 'molgap-colab-k1-paired500k-v1',
        'source_commit': binding['source']['source_commit'],
        'source_package_sha256': binding['source']['archive_sha256'],
        'job_id': 'colab:11Ri_0aHTJ35kXBX1ml2JY84bL0Ix4eFq:attempt-001',
        'cpu_accepted_dataset_manifest_sha256': binding['manifest_sha256'],
        'initial_format': 'molgap-k1-single-ema-initial-v1',
        'initial_state_sha256': binding['initial_tensor_sha256'],
        'allocation_started_unix': ALLOCATION_STARTED_UNIX}}
    config_path = run/'parent_runconfig.json'
    config_path.write_text(json.dumps(config, sort_keys=True, indent=2))
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(source/'src'), CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='42')
    command = [python,'-u','-m','molgap.colab_k1_screen','run',
        '--dataset-root',str(cache),'--initial-path',str(payload/'initial_state.pt'),
        '--initial-sha256',binding['initial_file_sha256'],
        '--runconfig-path',str(config_path),'--runconfig-sha256',config_sha,
        '--source-archive',str(payload/'source/source.tar.gz'),'--output',str(run/'worker'),
        '--deadline',str(DEADLINE)]
    print('BOUNDED_A100_PAIR_START', config, 'deadline', DEADLINE, flush=True)
    with (run/'worker.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        returncode = process.wait(timeout=max(1, DEADLINE-time.time()-30))
    observation = {{'returncode': returncode, 'observed_allocation_wall_seconds': time.time()-ALLOCATION_STARTED_UNIX,
        'includes_setup': True, 'billing_tail_seconds': None, 'ceiling_seconds': 14400}}
    (run/'allocation_observation.json').write_text(json.dumps(observation, indent=2))
    print('BOUNDED_ATTEMPT_FINISHED', observation, flush=True)
    if (run/'worker/terminal.json').exists():
        print((run/'worker/terminal.json').read_text(), flush=True)
    assert returncode == 0, 'Worker failed; retain artifacts and reconcile, no successor.'
finally:
    if process is not None and process.poll() is None:
        process.kill()
    # Request release while the hard timer remains active if this call fails.
    runtime.unassign()
    watchdog.cancel()
'''
    return [cpu, upload, gpu]


def main():
    from molgap.training_reproducibility import sha256_file, atomic_json
    binding = json.loads((HERE/'submission/package_binding.json').read_text())
    stage = ROOT/'platforms/_records/colab/staging'/binding['run_id']
    archive = stage/'k1_single_ema_500k_20261010.zip'
    retained = {path.relative_to(stage).as_posix(): sha256_file(path)
                for path in stage.rglob('*') if path.is_file() and path != archive}
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as handle:
        for relative in sorted(retained):
            handle.write(stage/relative, relative)
        handle.writestr('payload_manifest.json', json.dumps({'files': retained}, sort_keys=True))
    notebook_binding = {'payload_name': archive.name, 'payload_sha256': sha256_file(archive)}
    atomic_json(HERE/'submission/upload_binding.json', notebook_binding)
    sources = cells(notebook_binding)
    sources[-1] = sources[-1].replace('ALLOCATION_START_PLACEHOLDER', 'None # Set from the pre-connect allocation receipt')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'colab': {'name': 'MolGap_K1_Single_EMA_500K.ipynb'},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3', 'language': 'python'}},
        'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {},
                   'outputs': [], 'source': value.splitlines(True)} for value in sources]}
    atomic_json(HERE/'MolGap_K1_Single_EMA_500K.ipynb', notebook)
    for index, source in enumerate(sources):
        (stage/f'cell_{index}.py').write_text(source, encoding='utf-8')
    print(json.dumps({'archive': str(archive), **notebook_binding}))


if __name__ == '__main__':
    main()
