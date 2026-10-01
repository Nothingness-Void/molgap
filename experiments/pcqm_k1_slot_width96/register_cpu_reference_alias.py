"""Register the CPU plan's clerical bundle label without changing frozen inputs."""
from pathlib import Path
import json
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.v5_common import validate_reference_bundle

HERE = Path(__file__).resolve().parent

def main():
    source = HERE/'reference_binding/reference_bundle.json'
    retained = json.loads(source.read_text(encoding='utf-8'))
    validate_reference_bundle(retained)
    trajectory = json.loads((HERE/'qualification/trajectory.json').read_text(encoding='utf-8'))
    alias = trajectory['reference_bundle_id']
    if alias != 'k1-v4-192-kaggle3-reference-custody-20261002':
        raise ValueError('Unexpected planned alias')
    output = HERE/'qualification_reference_alias'
    output.mkdir(exist_ok=False)
    original_id = retained['reference_bundle_id']
    retained['reference_bundle_id'] = alias
    validate_reference_bundle(retained)
    atomic_json(output/'reference_bundle.json',retained)
    atomic_json(output/'registration.json',{
        'scope':'Metadata alias registration only; no model execution or change to frozen CPU inputs',
        'reason':'CPU prospective planner used a dated custody alias rather than the existing s42-v1 bundle ID',
        'original_reference_bundle_id':original_id,'registered_alias':alias,
        'original_bundle_path':'experiments/pcqm_k1_slot_width96/reference_binding/reference_bundle.json',
        'original_bundle_sha256':sha256_file(source),
        'all_fields_except_bundle_id_preserved':True})

if __name__=='__main__':main()
