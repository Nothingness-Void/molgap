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
    for _ in range(k):
        mol_copy = Chem.RWMol(mol_h)
        if AllChem.EmbedMolecule(mol_copy, AllChem.ETKDGv3()) != 0:
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


def build_labeled_graphs(
    smiles_list: list[str],
    targets: np.ndarray,
    *,
    use_charges: bool = True,
    mmff_iters: int = 200,
    show_progress: bool = True,
) -> list[Data]:
    """Build PyG graphs with y labels attached. Skips failed molecules."""
    from tqdm import tqdm

    graphs = []
    it = tqdm(smiles_list, desc="Building labeled graphs", disable=not show_progress)
    for i, smi in enumerate(it):
        data = smiles_to_pyg(smi, use_charges=use_charges, mmff_iters=mmff_iters)
        if data is not None:
            data.y = torch.tensor(targets[i], dtype=torch.float32).unsqueeze(0)
            graphs.append(data)
    return graphs
