"""Build immutable source and two small Kaggle entry packages."""
from pathlib import Path
import argparse
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
RESUME_SHA = __RESUME_SHA__
RESUME_EPOCH = __RESUME_EPOCH__

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
    resume_root = None
    if RESUME_SHA is not None:
        inputs = list(Path("/kaggle/input").rglob("stage_resume.bin"))
        if len(inputs) != 1 or hashlib.sha256(inputs[0].read_bytes()).hexdigest() != RESUME_SHA:
            raise RuntimeError("Missing or changed accepted resume archive")
        resume_root = Path("/kaggle/working/accepted_resume")
        with zipfile.ZipFile(inputs[0]) as archive:
            archive.extractall(resume_root)
        for arm in ARMS:
            result = json.loads((resume_root / arm / "stage_manifest.json").read_text())
            if result["arm"] != arm or result["next_epoch"] != RESUME_EPOCH:
                raise RuntimeError("Wrong accepted resume identity")
        print(f"Accepted resume archive verified; continuing epoch {RESUME_EPOCH}.", flush=True)
    workers = []
    for gpu, arm in enumerate(ARMS):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(runtime / "src"),
                   PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        command = [sys.executable, "-u", "-m", "molgap.pcqm_500k_v4_evidence", "--arm", arm,
                   "--output", f"/kaggle/working/evidence/{arm}", "--source-sha", SOURCE_SHA, "--stage-epochs", "4"]
        if resume_root is not None:
            command.extend(["--resume", str(resume_root / arm)])
        workers.append(subprocess.Popen(command, env=env))
    codes = [worker.wait() for worker in workers]
    if any(codes):
        raise RuntimeError(f"Worker failures: {dict(zip(ARMS, codes))}")
    terminal = {}
    for arm in ARMS:
        result = json.loads(Path(f"/kaggle/working/evidence/{arm}/stage_manifest.json").read_text())
        terminal[arm] = {key: result[key] for key in ("status", "next_epoch")}
    status_path = Path("/kaggle/working/kernel_status.json")
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(terminal, indent=2))
    os.replace(temporary, status_path)
    print("All stages published; inspect stage manifests for next_epoch.", flush=True)

if __name__ == "__main__":
    main()
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-stage1', action='store_true')
    parser.add_argument('--resume-stage', type=int)
    args = parser.parse_args()
    output = ROOT / 'platforms' / '_records' / 'kaggle' / 'staging' / 'pcqm_500k_v4_evidence'
    resume_sha = None
    resume_slug = None
    stage = args.resume_stage or (1 if args.resume_stage1 else None)
    if stage is not None:
        from accept_stage import accept
        if stage < 1:
            raise ValueError('Resume stage must be positive')
        record_name = 'submission.json' if stage == 1 else f'stage{stage}_submission.json'
        previous = json.loads((ROOT / 'experiments/pcqm_500k_v4_evidence' / record_name).read_text())
        roots = ROOT / f'platforms/_records/kaggle/training/pcqm_500k_v4_stage{stage}'
        inputs = {}
        for kernel in previous['kernels']:
            tag = 'gptrans' if kernel['arms'] == ['gptrans'] else 'edge-k1'
            for arm in kernel['arms']:
                inputs[arm] = roots / f"{tag}-v{kernel['version']}/evidence/{arm}"
        if set(inputs) != {'full_gps', 'neural_atom_k1', 'gptrans'}:
            raise ValueError('Incomplete arm inventory')
        accepted_inputs = {arm: accept(root) for arm, root in inputs.items()}
        epochs = {item['next_epoch'] for item in accepted_inputs.values()}
        if len(epochs) != 1 or any(item['training_complete'] for item in accepted_inputs.values()):
            raise ValueError('Expected aligned unfinished accepted stages')
        resume_epoch = epochs.pop()
        resume_dir = output / f'resume_stage{stage}'
        resume_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(resume_dir / 'stage_resume.bin', 'w', zipfile.ZIP_DEFLATED) as archive:
            for arm, root in inputs.items():
                manifest = json.loads((root / 'stage_manifest.json').read_text())
                for name in sorted(set(manifest['artifacts']) | {'stage_manifest.json'}):
                    archive.write(root / name, arm + '/' + name)
        resume_sha = hashlib.sha256((resume_dir / 'stage_resume.bin').read_bytes()).hexdigest()
        resume_slug = f'nothingnessvoid/molgap-500k-v4-resume-e{resume_epoch}-' + resume_sha[:10]
        (resume_dir / 'dataset-metadata.json').write_text(json.dumps({
            'id': resume_slug, 'title': f'MolGap 500K V4 Accepted Epoch{resume_epoch} ' + resume_sha[:10],
            'licenses': [{'name': 'other'}]}, indent=2))
        record = json.loads((ROOT / 'experiments/pcqm_500k_v4_evidence/submission.json').read_text())
        sha = record['source_sha256']
        slug = 'nothingnessvoid/molgap-500k-v4-source-raw-' + sha[:10]
        # Resume packaging must not regenerate or change frozen training source.
        for tag, arms in (('edge-k1', ['full_gps', 'neural_atom_k1']), ('gptrans', ['gptrans'])):
            package = output / (tag + f'-stage{stage+1}')
            package.mkdir(exist_ok=True)
            entry = ENTRY.replace('__ARMS__', repr(arms)).replace('__GPU__', repr('T4'))
            entry = entry.replace('__SHA__', repr(sha)).replace('__RESUME_SHA__', repr(resume_sha))
            entry = entry.replace('__RESUME_EPOCH__', str(resume_epoch))
            (package / 'run.py').write_text(entry, encoding='utf-8')
            metadata = json.loads((output / tag / 'kernel-metadata.json').read_text())
            metadata['dataset_sources'] = [slug, 'nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1', resume_slug]
            (package / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps({'resume_directory': str(resume_dir), 'resume_sha256': resume_sha,
            'resume_dataset': resume_slug, 'source_sha256': sha}, indent=2))
        return
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
        (package / 'run.py').write_text(ENTRY.replace('__ARMS__', repr(arms)).replace('__GPU__', repr(gpu)).replace('__SHA__', repr(sha)).replace('__RESUME_SHA__', 'None').replace('__RESUME_EPOCH__', '0'), encoding='utf-8')
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
