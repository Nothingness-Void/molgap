"""Build immutable source and two small Kaggle entry packages."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]

ENTRY = '''"""Bounded matched-500K evidence entry; source supplied by immutable dataset."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ARMS = __ARMS__
EXPECTED_GPU = __GPU__
SOURCE_SHA = __SHA__

def main():
    names = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()
    print("GPU allocation:", names, flush=True)
    if len(names) != 2 or not all(EXPECTED_GPU in n for n in names):
        raise RuntimeError("Unexpected accelerator allocation")
    archives = list(Path("/kaggle/input").rglob("evidence_source.bin"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected unique source archive: {archives}")
    if hashlib.sha256(archives[0].read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError("Source archive hash changed")
    # All imports of Torch happen in new workers after installing the frozen runtime.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cu121"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "torch-geometric==2.6.1", "ogb==1.3.6"])
    runtime = Path("/kaggle/working/runtime")
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(runtime)
    workers = []
    for gpu, arm in enumerate(ARMS):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(runtime / "src"),
                   PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        command = [sys.executable, "-u", "-m", "molgap.pcqm_500k_v4_evidence", "--arm", arm,
                   "--output", f"/kaggle/working/evidence/{arm}", "--source-sha", SOURCE_SHA, "--stage-epochs", "4"]
        workers.append(subprocess.Popen(command, env=env))
    codes = [worker.wait() for worker in workers]
    if any(codes):
        raise RuntimeError(f"Worker failures: {dict(zip(ARMS, codes))}")
    print("All stages published; inspect stage manifests for next_epoch.", flush=True)

if __name__ == "__main__":
    main()
'''


def main():
    output = ROOT / 'platforms' / '_records' / 'kaggle' / 'staging' / 'pcqm_500k_v4_evidence'
    source = output / 'source_raw'
    source.mkdir(parents=True, exist_ok=True)
    names = subprocess.check_output(['git', 'ls-files', 'src'], cwd=ROOT, text=True).splitlines()
    names.append('src/molgap/pcqm_500k_v4_evidence.py')
    with zipfile.ZipFile(source / 'evidence_source.bin', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(names)):
            if name.endswith('.py'):
                item = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(item, (ROOT / name).read_bytes().replace(b'\r\n', b'\n'))
    sha = hashlib.sha256((source / 'evidence_source.bin').read_bytes()).hexdigest()
    slug = 'nothingnessvoid/molgap-500k-v4-source-raw-' + sha[:10]
    (source / 'dataset-metadata.json').write_text(json.dumps({
        'id': slug, 'title': 'MolGap 500K V4 Evidence Source ' + sha[:10],
        'licenses': [{'name': 'other'}]}, indent=2))
    for tag, arms, gpu, machine in (
        ('edge-k1', ['full_gps', 'neural_atom_k1'], 'T4', 'NvidiaTeslaT4'),
        ('gptrans', ['gptrans'], 'T4', 'NvidiaTeslaT4')):
        package = output / tag
        package.mkdir(exist_ok=True)
        (package / 'run.py').write_text(ENTRY.replace('__ARMS__', repr(arms)).replace('__GPU__', repr(gpu)).replace('__SHA__', repr(sha)), encoding='utf-8')
        (package / 'kernel-metadata.json').write_text(json.dumps({
            'id': 'nothingnessvoid/molgap-500k-v4-' + tag + '-s42',
            'title': 'MolGap 500K V4 ' + tag + ' S42', 'code_file': 'run.py',
            'language': 'python', 'kernel_type': 'script', 'is_private': 'true',
            'enable_gpu': 'true', 'enable_internet': 'true', 'machine_shape': machine,
            'dataset_sources': [slug, 'nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1'],
            'competition_sources': [], 'kernel_sources': [], 'model_sources': []}, indent=2))
    print(json.dumps({'directory': str(output), 'source_sha256': sha, 'source_dataset': slug}, indent=2))


if __name__ == '__main__':
    main()
