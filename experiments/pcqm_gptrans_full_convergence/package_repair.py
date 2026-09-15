"""Package complete tracked runtime and generate guarded continuation jobs."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BASE = '/lustre/home/users/sm2/chou/molgap-k1-gpttrans-full'


def main():
    out = ROOT / 'platforms/_records/ims/convergence_repair_r3'
    out.mkdir(parents=True, exist_ok=True)
    tracked = subprocess.check_output(['git', 'ls-files', 'src'], cwd=ROOT, text=True).splitlines()
    files = {p: (ROOT / p).read_bytes().replace(b'\r\n', b'\n')
             for p in tracked if p.endswith('.py')}
    for family in ('gptrans', 'k1'):
        name = f'experiments/pcqm_{family}_full_convergence/run.py'
        files[name] = (ROOT / name).read_bytes().replace(b'\r\n', b'\n')
    for required in ('pcqm_k1_full_runner', 'pcqm_gptrans_full_runner',
                     'pcqm_k1_convergence', 'pcqm_gptrans_convergence'):
        assert f'src/molgap/{required}.py' in files, required
    for name, data in files.items():
        compile(data, name, 'exec')
    hashes = {p: hashlib.sha256(data).hexdigest() for p, data in files.items()}
    files['runtime_files.json'] = json.dumps(hashes, indent=2).encode()
    verifier = '''import hashlib,json,pathlib
root=pathlib.Path(__file__).resolve().parent
for name, expected in json.loads((root/'runtime_files.json').read_text()).items():
    path=(root/name).resolve()
    path.relative_to(root)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, name
print('Runtime file hashes verified', flush=True)
'''
    files['verify_runtime.py'] = verifier.encode()
    with tarfile.open(out / 'runtime.tar.gz', 'w:gz') as tar:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo('code/' + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    for family, old, new in (
        ('gptrans', 'gptrans_convergence_20260915_r2', 'gptrans_convergence_20260915_r3'),
        ('k1', 'k1_convergence_20260915_r1', 'k1_convergence_20260915_r2'),
    ):
        jobs = ROOT / f'experiments/pcqm_{family}_full_convergence/jobs'
        train = (jobs / 'train.pbs').read_text().replace(old, new)
        pre = (jobs / 'preflight.pbs').read_text().replace(old, new)
        # Same allocation executes preflight then training, avoiding a second GPU queue.
        pre_command = pre.split('cd "$ROOT"\n', 1)[1]
        validation = ('"$PYTHON" "$ROOT/code/verify_runtime.py"\n'
                      f'"$PYTHON" "$ROOT/code/experiments/pcqm_{family}_full_convergence/run.py" --help\n')
        train = train.replace('RESUME=()', validation + pre_command + '\nRESUME=()')
        (out / f'{family}_train.pbs').write_text(train, newline='\n')
        accept = (jobs / 'accept.pbs').read_text().replace(old, new)
        (out / f'{family}_accept.pbs').write_text(accept, newline='\n')
        # Import-only check is safe on login: no dataset reads or model execution.
        environment = pre.split('cd "$ROOT"\n', 1)[0].split('set -euo pipefail\n', 1)[1]
        (out / f'{family}_imports.sh').write_text('set -euo pipefail\n' + environment +
            'cd "$ROOT"\n' + validation, newline='\n')
    print(json.dumps({'archive': str(out / 'runtime.tar.gz'),
        'sha256': hashlib.sha256((out / 'runtime.tar.gz').read_bytes()).hexdigest(),
        'runtime_files': len(hashes)}, indent=2))


if __name__ == '__main__':
    main()
