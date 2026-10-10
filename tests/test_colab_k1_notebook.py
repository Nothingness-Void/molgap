"""Adapter syntax/identity checks only; never connect Colab or consume real roles."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'experiments/pcqm_k1_single_ema_500k'


def builder():
    spec = importlib.util.spec_from_file_location('k1_colab_notebook', HERE/'build_notebook.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_cells_compile_and_budget_is_original_not_reset():
    sources = builder().cells({'payload_sha256': 'b'*64, 'payload_name': 'synthetic.zip'})
    for source in sources:
        compile(source.replace('ALLOCATION_START_PLACEHOLDER', '123.0'), '<cell>', 'exec')
    gpu = sources[-1]
    assert 'DEADLINE = ALLOCATION_STARTED_UNIX + 14400' in gpu
    assert 'runtime.unassign()' in gpu and "process.kill()" in gpu
    assert "--source-archive',str(payload/'source/source.tar.gz')" in gpu
    assert "'initial_format': 'molgap-k1-single-ema-initial-v1'" in gpu
    assert 'KAGGLE_KEY' not in ''.join(sources)


def test_allowlist_paths_exist():
    paths = json.loads((HERE/'source_allowlist.json').read_text())
    assert len(paths) == len(set(paths))
    assert all((ROOT/path).is_file() and path.startswith('src/molgap/') for path in paths)


def test_existing_colab_dataset_acceptance_owner_is_reused():
    source = builder().bootstrap_acceptance()
    assert 'def accept_fixed_dataset(' in source
    assert 'if STAGE_FIXED_DATA:' not in source
    assert 'subprocess.check_call' not in source
