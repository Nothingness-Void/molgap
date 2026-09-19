"""Reuse the existing attribution implementation with recovered exact assets."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
BASE = Path('D:/文档/molgap-500k-v4-evidence')
OLD = Path('D:/文档/molgap-500k-attribution')
source = OLD / 'src/molgap/pcqm_500k_module_attribution.py'
spec = importlib.util.spec_from_file_location('attribution', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
paths = {
    'k1': BASE / 'platforms/_records/kaggle/training/pcqm_500k_v4_stage5/edge-k1-v6/evidence/neural_atom_k1/best_predictions.pt',
    'gptrans_t': BASE / 'platforms/_records/kaggle/training/pcqm_500k_v4_stage5/gptrans-v7/evidence/gptrans/best_predictions.pt',
    'edge_state': BASE / 'platforms/_records/kaggle/training/pcqm_500k_v4_stage6/full-gps-final-epoch-v8/evidence/full_gps/best_predictions.pt',
}
expected = json.loads((HERE.parent / 'pcqm_500k_v4_evidence/local_ablation_analysis.json').read_text())['prediction_sha256']
started = time.perf_counter()
result = module.analyze(
    OLD / 'data/cache/pcqm4mv2_500k_v5_topology/train_shard_0010.pt', paths,
    expected_hashes={k: expected[k] for k in paths},
    expected_graph_hash='3879dc27106034c27c893bac7d182d54fbbe93906efb83108e9dac541e0894ae',
)
result['analyzer_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
result['input_paths'] = {k: str(v) for k, v in paths.items()}
result['local_wall_seconds'] = time.perf_counter() - started
result['scale_comparison_status'] = 'blocked_missing_pairtoken_and_100k_matched_predictions'
(HERE / 'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: result[k] for k in ('mae_eV', 'diagnostic_crossfit') if k in result}))
