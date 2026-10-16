"""Exact authorized P scope over the retained native graph proof; CPU only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from molgap.pcqm_molecular_identity import (
    TWO_P_MOLECULE_RULE, AUTHORIZED_P_SOURCE_INDICES, load_fallback_keys, molecular_identity,
)
from molgap.pcqm_unified_roles import prepare_unified_roles,validate_rule,choose_roles,validate_role_members
from molgap.pcqm_fixed_datasets import sha256_file
from molgap.research_memory.trace import json_bytes
from test_pcqm_unified_roles import source_fixture,rule,database
from test_pcqm_molecular_identity import compatible_rule


PROOF=Path(__file__).resolve().parents[1]/'platforms/_records/ims/pcqm_unified_roles_v2/identity_repair/explicit_h_v3/exception_graph_audit.json'


def p_ledger(tmp_path):
    proof=tmp_path/'exception_graph_audit.json';proof.write_bytes(PROOF.read_bytes())
    data=json.loads(proof.read_text())
    authorization={'scope':'EXACT_TWO_GRAPH_VERIFIED_P_STRUCTURES_CPU_PREPARATION_ONLY',
        'source_indices':list(AUTHORIZED_P_SOURCE_INDICES),'graph_proof_sha256':sha256_file(proof),'user_approved':True}
    auth=tmp_path/'phosphorus_authorization.json';auth.write_bytes(json_bytes(authorization))
    payload={'format':'molgap-pcqm-unified-accepted-si-two-p-ledger-v1',
        'entries':[{k:e[k] for k in ('source_idx','smiles')} for e in data['entries']],
        **{k:data[k] for k in ('source_manifest_sha256','source_acceptance_sha256','identity_audit_sha256','all_hydrogens_explicit')},
        'graph_proof_sha256':sha256_file(proof),'phosphorus_authorization_sha256':sha256_file(auth)}
    ledger=tmp_path/'fallback_ledger.json';ledger.write_bytes(json_bytes(payload))
    return ledger,payload,auth,authorization


def test_only_exact_two_p_structures_and_aliases_are_admitted(tmp_path):
    ledger,payload,_,_=p_ledger(tmp_path)
    keys=load_fallback_keys(ledger,sha256_file(ledger),explicit_hydrogens=True,allow_phosphorus=True)
    assert len(keys)==33
    proof=json.loads(PROOF.read_text())
    for entry in proof['entries']:
        key,evidence=molecular_identity(entry['smiles'],fallback_keys=keys,explicit_hydrogens=True,allow_phosphorus=True)
        assert key==entry['canonical_smiles'] and evidence['status']=='KNOWN_FALLBACK'
    for entry in proof['entries']:
        if entry['source_idx'] in AUTHORIZED_P_SOURCE_INDICES:
            alias=entry['canonical_smiles']
            assert molecular_identity(alias,fallback_keys=keys,explicit_hydrogens=True,allow_phosphorus=True)[0]==alias
            assert molecular_identity(alias,fallback_keys=keys,explicit_hydrogens=True)[0] is None
    assert molecular_identity('CP(C)(C)(C)(C)C',fallback_keys=keys,explicit_hydrogens=True,allow_phosphorus=True)[0] is None
    assert molecular_identity('CN(C)(C)C',fallback_keys=keys,explicit_hydrogens=True,allow_phosphorus=True)[0] is None
    assert molecular_identity('CCO',fallback_keys=keys,explicit_hydrogens=True,allow_phosphorus=True)==('CCO',None)
    with pytest.raises(ValueError,match='separately authorized rule'):
        load_fallback_keys(ledger,sha256_file(ledger),explicit_hydrogens=True)


@pytest.mark.parametrize('kind',['approval','extra_p','proof','features','missing_p'])
def test_p_scope_and_graph_authority_tamper_never_passes(tmp_path,kind):
    ledger,payload,auth,authorization=p_ledger(tmp_path)
    if kind=='approval':authorization['user_approved']=False
    elif kind=='extra_p':authorization['source_indices'].append(3000000)
    elif kind=='proof':payload['graph_proof_sha256']='0'*64
    else:
        proof=tmp_path/'exception_graph_audit.json';data=json.loads(proof.read_text())
        if kind=='features':
            next(e for e in data['entries'] if e['source_idx']==2539039)['feature_matches']['x']=False
        else:data['entries']=[e for e in data['entries'] if e['source_idx']!=2980839]
        proof.write_bytes(json_bytes(data));payload['graph_proof_sha256']=sha256_file(proof)
        authorization['graph_proof_sha256']=sha256_file(proof)
    auth.write_bytes(json_bytes(authorization));payload['phosphorus_authorization_sha256']=sha256_file(auth)
    ledger.write_bytes(json_bytes(payload))
    with pytest.raises(ValueError,match='ROLE_BLOCKED'):
        load_fallback_keys(ledger,sha256_file(ledger),explicit_hydrogens=True,allow_phosphorus=True)


def test_v4_rule_identity_is_distinct_and_synthetic_preparation_reuses_owner(tmp_path):
    fixed,r=source_fixture(tmp_path)
    ledger,payload,_,_=p_ledger(tmp_path)
    r=compatible_rule(r);r['molecule_rule']=TWO_P_MOLECULE_RULE;r['fallback_ledger_sha256']=sha256_file(ledger)
    assert validate_rule(r)!=validate_rule(rule())
    # The native proof cannot be applied to a different synthetic source asset.
    with pytest.raises(ValueError,match='different source assets'):
        prepare_unified_roles(boundary=tmp_path,fixed_root=fixed,output_root=tmp_path/'out',rule=r,fallback_ledger=ledger)


def test_full_complement_cannot_silently_drop_an_unheld_training_row():
    db=database();roles=choose_roles(db,rule())
    omit=next(i for i in roles['full_minus_holdout'] if i>=6)
    roles['full_minus_holdout'].remove(omit)
    with pytest.raises(ValueError,match='exact held-molecule complement'):
        validate_role_members(db,roles,rule())


def test_v4_whole_cpu_fixture_with_both_p_rows_and_resume(tmp_path):
    import gzip
    import torch
    from torch_geometric.data import Data,InMemoryDataset
    from molgap.packed_graph_dataset import PackedGraphDataset
    from molgap.pcqm_k1_scale_cache import _unsanitized_ogb_payload
    fixed,r=source_fixture(tmp_path)
    ledger,payload,auth,authorization=p_ledger(tmp_path)
    p_entries=[e for e in json.loads(PROOF.read_text())['entries'] if e['source_idx'] in AUTHORIZED_P_SOURCE_INDICES]
    smiles={i:e['smiles'] for i,e in enumerate(p_entries)}
    shard=PackedGraphDataset(fixed/'train.pt');graphs=[shard.get(i) for i in range(30)]
    for i,smi in smiles.items():
        p=_unsanitized_ogb_payload(smi)
        graphs[i]=Data(x=torch.as_tensor(p['node_feat']),edge_index=torch.as_tensor(p['edge_index']),
            edge_attr=torch.as_tensor(p['edge_feat']),random_walk_pe=torch.zeros((p['num_nodes'],16)),
            y=torch.tensor([i/10]),source_idx=torch.tensor([i]))
    torch.save(InMemoryDataset.collate(graphs),fixed/'train.pt')
    with gzip.open(fixed/'rows.csv.gz','wt') as f:
        f.write('idx,smiles,homolumogap\n')
        for i in range(30):f.write(f"{i},{smiles.get(i,'C'*(i+1))},{i/10}\n")
        f.write('30,DO_NOT_READ_VALIDATION,not-a-label\n')
    manifest=json.loads((fixed/'manifest.json').read_text())
    manifest['store_assets']['source'][0]['sha256']=sha256_file(fixed/'rows.csv.gz')
    manifest['store_assets']['topology'][0]['sha256']=sha256_file(fixed/'train.pt')
    (fixed/'manifest.json').write_bytes(json_bytes(manifest));r['source_manifest_sha256']=sha256_file(fixed/'manifest.json')
    # A synthetic source-authority envelope only: never claim it as native proof.
    proof=tmp_path/'exception_graph_audit.json';data=json.loads(proof.read_text())
    for k in ('source_manifest_sha256','source_acceptance_sha256'):
        data[k]=payload[k]=r[k]
    proof.write_bytes(json_bytes(data));payload['graph_proof_sha256']=sha256_file(proof)
    authorization['graph_proof_sha256']=sha256_file(proof);auth.write_bytes(json_bytes(authorization))
    payload['phosphorus_authorization_sha256']=sha256_file(auth);ledger.write_bytes(json_bytes(payload))
    r=compatible_rule(r);r.update(molecule_rule=TWO_P_MOLECULE_RULE,fallback_ledger_sha256=sha256_file(ledger))
    kwargs=dict(boundary=tmp_path,fixed_root=fixed,output_root=tmp_path/'out',rule=r,fallback_ledger=ledger)
    report=prepare_unified_roles(**kwargs)
    assert report['status']=='SYNTHETIC_DATA_ROLES_VERIFIED'
    assert prepare_unified_roles(**kwargs)==report
    audit=json.loads((tmp_path/'out/molecular_identity_audit.json').read_text())
    assert audit['known_fallback_rows']==2 and audit['blocked_rows']==0 and audit['rows_skipped_or_replaced']==0


def test_source_topology_probe_is_observation_only_and_readonly(tmp_path):
    from molgap.pcqm_unified_roles import inspect_unified_source_topology
    fixed,r=source_fixture(tmp_path);before={p.name:sha256_file(p) for p in fixed.iterdir()}
    result=inspect_unified_source_topology(boundary=tmp_path,fixed_root=fixed,output_root=tmp_path/'probe',rule=r)
    assert result['observed_rows']==30 and result['expected_rows']==30
    assert not result['data_role_accepted'] and not result['graphs_rebuilt']
    assert {p.name:sha256_file(p) for p in fixed.iterdir()}==before


def test_only_known_provenance_is_projected_without_model_tensor_changes(tmp_path):
    import torch
    from molgap.pcqm_unified_roles import _project_source_graph
    from molgap.packed_graph_dataset import PackedGraphDataset
    fixed,r=source_fixture(tmp_path);g=PackedGraphDataset(fixed/'train.pt').get(0)
    g.smiles_sanitized=torch.tensor([1],dtype=torch.int8);g.split_code=torch.tensor([0],dtype=torch.int8)
    clean=_project_source_graph(g)
    assert not {'smiles_sanitized','split_code'}&set(clean.keys())
    assert {'smiles_sanitized','split_code'}<=set(g.keys())
    assert all(getattr(clean,k) is getattr(g,k) for k in ('x','edge_index','edge_attr','random_walk_pe','y','source_idx'))
    g.split_code=torch.tensor([3],dtype=torch.int8)
    with pytest.raises(ValueError,match='provenance'):_project_source_graph(g)
    g.split_code=torch.tensor([0],dtype=torch.int8);g.pos=torch.zeros((1,3))
    with pytest.raises(ValueError,match='Non-2D'):_project_source_graph(g)


def identity_parent(tmp_path):
    fixed,r=source_fixture(tmp_path);r=compatible_rule(r)
    ledger=tmp_path/'old_ledger.json';ledger.write_bytes(Path(__file__).resolve().parents[1].joinpath('experiments/pcqm_k1_scale500k/parser_fallback_audit_v4.json').read_bytes())
    parent=tmp_path/'parent'
    prepare_unified_roles(boundary=tmp_path,fixed_root=fixed,output_root=parent,rule=r,fallback_ledger=ledger,identity_only=True)
    reuse={'root':parent,'audit_sha256':sha256_file(parent/'molecular_identity_audit.json'),
           'database_sha256':sha256_file(parent/'private_identity.sqlite')}
    return fixed,r,ledger,parent,reuse


def test_hash_bound_complete_identity_reuse_skips_parsing_and_preserves_parent(tmp_path,monkeypatch):
    import molgap.pcqm_unified_roles as owner
    fixed,r,ledger,parent,reuse=identity_parent(tmp_path)
    before={p.name:sha256_file(p) for p in parent.iterdir()}
    r=deepcopy(r);r['source_commit']='e'*40;r['generator_sha256']='b'*64
    monkeypatch.setattr(owner,'inventory_rows',lambda *a,**k:(_ for _ in ()).throw(AssertionError('must reuse complete identities')))
    kwargs=dict(boundary=tmp_path,fixed_root=fixed,output_root=tmp_path/'reused',rule=r,fallback_ledger=ledger,identity_reuse=reuse)
    result=prepare_unified_roles(**kwargs)
    assert result['status']=='SYNTHETIC_DATA_ROLES_VERIFIED'
    assert prepare_unified_roles(**kwargs)==result
    assert {p.name:sha256_file(p) for p in parent.iterdir()}==before


@pytest.mark.parametrize('tamper',['seed','audit_sha','database_sha','wal'])
def test_identity_reuse_cannot_accept_changed_contract_or_unclosed_bytes(tmp_path,tamper):
    fixed,r,ledger,parent,reuse=identity_parent(tmp_path)
    if tamper=='seed':r['seed']=43
    elif tamper=='audit_sha':reuse['audit_sha256']='0'*64
    elif tamper=='database_sha':reuse['database_sha256']='0'*64
    else:(parent/'private_identity.sqlite-wal').write_bytes(b'not-closed')
    with pytest.raises(ValueError,match='Identity reuse'):
        prepare_unified_roles(boundary=tmp_path,fixed_root=fixed,output_root=tmp_path/'reused',rule=r,fallback_ledger=ledger,identity_reuse=reuse)
