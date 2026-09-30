"""Atom-aligned local reconstruction targets on the frozen PCQM training role."""
from pathlib import Path
import json
import numpy as np
import torch
from .pcqm_gptrans_v4 import MANIFEST_SHA256, validate_fixed_assets, _load_datasets
from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file
from .qm9_local_hierarchy import _functional_group_labels, _smarts_sha256


def build_cache(archive: Path, fixed_root: Path, output: Path):
    import pandas as pd
    from rdkit import Chem, rdBase
    from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
    from .pcqm_official_edge_state import load_official_splits, _open_official_csv
    if sha256_file(archive) != '628ae612a3de752160929d93d1584a75257ae156e7b13d91b133d8db62e6d701':
        raise ValueError('Official archive identity changed')
    assets = validate_fixed_assets(fixed_root, fixed_root/'manifest.json', verify_content=True)
    if not np.isin(np.arange(100000), load_official_splits(archive)['train']).all():
        raise ValueError('Requested prefix is not entirely official training')
    bundle, compressed, stream = _open_official_csv(archive)
    try:
        # Do not materialize Gap labels or any row outside the authorized train role.
        frame = pd.read_csv(stream, nrows=100000, usecols=['idx','smiles'])
    finally:
        stream.close(); compressed.close(); bundle.close()
    if not np.array_equal(frame.idx.values, np.arange(100000)):
        raise ValueError('Source rows changed')
    graphs, _ = _load_datasets(assets.train_paths)
    labels, offsets = [], [0]
    for i, smiles in enumerate(frame.smiles):
        graph = graphs[i]
        molecule=Chem.MolFromSmiles(smiles)
        if molecule is None:
            # Identical fallback to the owning official graph builder; no row drop.
            molecule=Chem.MolFromSmiles(smiles,sanitize=False)
            if molecule is None: raise ValueError(f'Unparseable training row: {i}')
            molecule.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(molecule)
        edges,features=[],[]
        for bond in molecule.GetBonds():
            a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
            edges.extend([(a,b),(b,a)])
            features.extend([bond_to_feature_vector(bond)]*2)
        expected={'node_feat':torch.tensor([atom_to_feature_vector(a) for a in molecule.GetAtoms()]),
                  'edge_index':torch.tensor(edges,dtype=torch.long).reshape(-1,2).t(),
                  'edge_feat':torch.tensor(features,dtype=torch.long).reshape(-1,3)}
        if int(graph.source_idx.item()) != i:
            raise ValueError(f'Source alignment failed: {i}')
        for key, value in [('x','node_feat'),('edge_index','edge_index'),('edge_attr','edge_feat')]:
            if not torch.equal(graph[key].cpu(), torch.as_tensor(expected[value])):
                raise ValueError(f'Atom/bond alignment failed at {i}: {key}')
        label = _functional_group_labels(molecule).to(torch.uint8)
        labels.append(label); offsets.append(offsets[-1] + len(label))
        if (i+1)%10000 == 0: print(f'aligned hierarchy labels {i+1}/100000',flush=True)
    output.mkdir(parents=True, exist_ok=True)
    path=output/'hierarchy_labels.pt'
    atomic_torch_save(path, {'labels':torch.cat(labels), 'offsets':torch.tensor(offsets),
                            'source_idx':torch.arange(100000)})
    manifest={'format':'molgap-local-hierarchy-fixed100k-v1','rows':100000,
              'graph_manifest_sha256':MANIFEST_SHA256,'labels_sha256':sha256_file(path),
              'smarts_sha256':_smarts_sha256(),'rdkit_version':rdBase.rdkitVersion,
              'archive_sha256':sha256_file(archive),'gap_labels_read':False,
              'protected_roles_read':False,'atom_and_bond_features_exact':True}
    atomic_json(output/'hierarchy_manifest.json',manifest)
    return manifest


class LabelledTrainingGraphs:
    def __init__(self, graphs, root: Path, expected_manifest_sha256: str):
        manifest_path=root/'hierarchy_manifest.json'
        if sha256_file(manifest_path)!=expected_manifest_sha256:
            raise ValueError('Hierarchy manifest identity changed')
        manifest=json.loads(manifest_path.read_text())
        if (manifest['graph_manifest_sha256']!=MANIFEST_SHA256 or manifest['rows']!=100000
                or manifest['smarts_sha256']!=_smarts_sha256()
                or sha256_file(root/'hierarchy_labels.pt')!=manifest['labels_sha256']):
            raise ValueError('Hierarchy parent or label identity changed')
        self.payload=torch.load(root/'hierarchy_labels.pt',map_location='cpu',weights_only=False)
        self.graphs=graphs
        if len(graphs)!=100000 or not torch.equal(self.payload['source_idx'],torch.arange(100000)):
            raise ValueError('Hierarchy training membership changed')

    def __len__(self): return len(self.graphs)

    def __getitem__(self, index):
        graph=self.graphs[index].clone()
        if int(graph.source_idx.item())!=index: raise ValueError('Hierarchy row identity changed')
        offsets=self.payload['offsets']
        graph.functional_group_y=self.payload['labels'][offsets[index]:offsets[index+1]]
        if graph.functional_group_y.shape!=(graph.num_nodes,12): raise ValueError('Atom count changed')
        return graph
