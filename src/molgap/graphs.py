"""Unified 3D graph building from SMILES via ETKDG conformers."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .utils import require_rdkit, compute_gasteiger_charges

if TYPE_CHECKING:
    from torch_geometric.data import Data


def smiles_to_pyg(
    smiles: str,
    *,
    use_charges: bool = True,
    mmff_iters: int = 200,
    max_embed_attempts: int = 2,
) -> Data | None:
    """Convert a SMILES string to a PyG Data object with ETKDG 3D coordinates.

    Returns None if parsing or embedding fails.
    """
    require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_h = AllChem.AddHs(mol)

    for _ in range(max_embed_attempts):
        if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) == 0:
            break
    else:
        return None

    if mmff_iters > 0:
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=mmff_iters)
        except Exception:
            pass

    n = mol_h.GetNumAtoms()
    if n == 0:
        return None

    conf = mol_h.GetConformer()
    z = torch.tensor(
        [mol_h.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)],
        dtype=torch.long,
    )
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)

    data = Data(z=z, pos=pos)
    if use_charges:
        charges = compute_gasteiger_charges(mol_h)
        data.charges = torch.tensor(charges, dtype=torch.float32)
    return data


def smiles_to_pyg_ensemble(
    smiles: str,
    k: int = 8,
    *,
    use_charges: bool = True,
    mmff_iters: int = 200,
    random_seed: int | None = None,
) -> list[Data]:
    """Generate k ETKDG conformers for one SMILES, return list of PyG Data."""
    require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol_h = AllChem.AddHs(mol)

    results = []
    for i in range(k):
        mol_copy = Chem.RWMol(mol_h)
        params = AllChem.ETKDGv3()
        if random_seed is not None:
            params.randomSeed = int(random_seed) + i
        if AllChem.EmbedMolecule(mol_copy, params) != 0:
            continue
        if mmff_iters > 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol_copy, maxIters=mmff_iters)
            except Exception:
                pass
        n = mol_copy.GetNumAtoms()
        if n == 0:
            continue
        conf = mol_copy.GetConformer()
        z = torch.tensor(
            [mol_copy.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)],
            dtype=torch.long,
        )
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
        data = Data(z=z, pos=pos)
        if use_charges:
            charges = compute_gasteiger_charges(mol_copy)
            data.charges = torch.tensor(charges, dtype=torch.float32)
        results.append(data)
    return results


def smiles_list_to_pyg(
    smiles_list: list[str],
    *,
    use_charges: bool = True,
    mmff_iters: int = 200,
    show_progress: bool = True,
) -> tuple[list[Data], list[int]]:
    """Convert a list of SMILES to PyG graphs. Returns (graphs, valid_indices)."""
    from tqdm import tqdm

    graphs = []
    valid_idx = []
    it = tqdm(smiles_list, desc="Building graphs", disable=not show_progress)
    for i, smi in enumerate(it):
        data = smiles_to_pyg(smi, use_charges=use_charges, mmff_iters=mmff_iters)
        if data is not None:
            graphs.append(data)
            valid_idx.append(i)
    return graphs, valid_idx


def _bond_type_map():
    from rdkit import Chem
    return {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }


def smiles_to_2d_pyg(smiles: str) -> Data | None:
    """Convert SMILES to a 2D bond-topology PyG graph (no 3D coords needed).

    Node features (9-dim): [atomic_num_onehot(6), degree, formal_charge,
                             is_aromatic]
    Edge features (4-dim): [bond_type_onehot(4)]
    """
    require_rdkit()
    from rdkit import Chem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    ATOM_LIST = [6, 7, 8, 9, 16, 17]  # C N O F S Cl

    node_feats = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        onehot = [1.0 if z == a else 0.0 for a in ATOM_LIST]
        feat = onehot + [
            atom.GetDegree() / 4.0,
            atom.GetFormalCharge() / 2.0,
            float(atom.GetIsAromatic()),
        ]
        node_feats.append(feat)

    if len(node_feats) == 0:
        return None

    x = torch.tensor(node_feats, dtype=torch.float32)

    rows, cols, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = _bond_type_map().get(bond.GetBondType(), 0)
        onehot = [1.0 if bt == k else 0.0 for k in range(4)]
        rows += [i, j]
        cols += [j, i]
        edge_feats += [onehot, onehot]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32) if edge_feats else torch.zeros((0, 4), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _build_one_labeled(args: tuple) -> Data | None:
    """Worker function for parallel graph building. Must be top-level for pickle."""
    smi, target = args
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from torch_geometric.data import Data

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol_h = AllChem.AddHs(mol)
        for _ in range(2):
            if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) == 0:
                break
        else:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
        n = mol_h.GetNumAtoms()
        if n == 0:
            return None
        conf = mol_h.GetConformer()
        z = torch.tensor([mol_h.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)], dtype=torch.long)
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
        AllChem.ComputeGasteigerCharges(mol_h)
        charges = []
        for atom in mol_h.GetAtoms():
            c = atom.GetDoubleProp('_GasteigerCharge')
            charges.append(0.0 if (c != c) or abs(c) > 1e6 else c)
        data = Data(
            z=z, pos=pos,
            charges=torch.tensor(charges, dtype=torch.float32),
            y=torch.tensor(target, dtype=torch.float32).unsqueeze(0),
        )
        return data
    except Exception:
        return None


def build_labeled_graphs(
    smiles_list: list[str],
    targets: np.ndarray,
    *,
    use_charges: bool = True,
    mmff_iters: int = 200,
    show_progress: bool = True,
    n_jobs: int | None = None,
) -> list[Data]:
    """Build PyG graphs with y labels attached. Skips failed molecules.

    Uses multiprocessing when n_jobs != 1. Default: all cores - 1.
    """
    from tqdm import tqdm

    if n_jobs == 1:
        graphs = []
        it = tqdm(smiles_list, desc="Building labeled graphs", disable=not show_progress)
        for i, smi in enumerate(it):
            data = smiles_to_pyg(smi, use_charges=use_charges, mmff_iters=mmff_iters)
            if data is not None:
                data.y = torch.tensor(targets[i], dtype=torch.float32).unsqueeze(0)
                graphs.append(data)
        return graphs

    import multiprocessing as mp
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    work = list(zip(smiles_list, targets.tolist()))
    graphs = []
    with mp.Pool(n_jobs) as pool:
        for result in tqdm(
            pool.imap(_build_one_labeled, work, chunksize=500),
            total=len(work), desc=f"Building graphs ({n_jobs} workers)",
            disable=not show_progress,
        ):
            if result is not None:
                graphs.append(result)
    return graphs
