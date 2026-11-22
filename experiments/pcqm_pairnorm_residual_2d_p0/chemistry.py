"""Fixed, coordinate-free chemistry and grouping for the P0 probe."""
from hashlib import md5

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from molgap.utils import calc_rdkit_descriptors

NAMES = [n for n, _ in Descriptors.descList]
COUNTS = ["conjugated_bonds", "conjugated_atoms", "aromatic_atoms", "double_bonds",
          "triple_bonds", "radical_electrons"]
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def bucket(key, n):
    return int(md5(key.encode("utf8")).hexdigest(), 16) % n


def keys(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return "!INVALID", ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False), Chem.MolToSmiles(m)


def feature_row(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.full(len(NAMES) + 6 + 1024, np.nan, np.float32), [-1] * 6
    bonds, atoms = list(m.GetBonds()), list(m.GetAtoms())
    conj = [b for b in bonds if b.GetIsConjugated()]
    counts = [len(conj), len({i for b in conj for i in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())}),
              sum(a.GetIsAromatic() for a in atoms), sum(b.GetBondTypeAsDouble() == 2 for b in bonds),
              sum(b.GetBondTypeAsDouble() == 3 for b in bonds), sum(a.GetNumRadicalElectrons() for a in atoms)]
    desc = calc_rdkit_descriptors(m)
    row = np.concatenate(([desc["desc_" + n] for n in NAMES], counts, FP.GetCountFingerprintAsNumPy(m)))
    elements = {a.GetAtomicNum() for a in atoms}
    groups = [counts[-1] > 0, all(b.GetBondTypeAsDouble() == 1 for b in bonds),
              m.GetRingInfo().NumRings(), elements <= {1, 6, 7, 8},
              bool(elements & {9, 17, 35, 53}), bool(elements & {15, 16})]
    return row.astype(np.float32), groups
