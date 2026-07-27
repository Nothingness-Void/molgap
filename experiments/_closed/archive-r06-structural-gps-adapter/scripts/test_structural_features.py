import unittest

import numpy as np

import torch

from molgap.archive.phase8_r06_structural_gps_adapter.structural_features import (
    structural_summary_from_mol,
    structural_topology_from_mol,
)
from molgap.utils import safe_mol


class StructuralFeaturesTest(unittest.TestCase):
    def test_ring_and_conjugation_features_are_deterministic(self):
        mol = safe_mol("c1ccc2ccccc2c1")
        first = structural_topology_from_mol(mol)
        second = structural_topology_from_mol(mol)

        self.assertTrue(np.array_equal(first.atom_features, second.atom_features))
        self.assertTrue(np.array_equal(first.edge_features, second.edge_features))
        self.assertEqual(first.atom_features.shape, (10, 4))
        self.assertEqual(first.edge_features.shape, (22, 2))
        self.assertTrue(np.isfinite(first.atom_features).all())
        self.assertTrue(np.isfinite(first.edge_features).all())
        self.assertEqual(structural_summary_from_mol(mol)["has_fused_ring_atom"], 1.0)

    def test_acyclic_molecule_has_finite_zero_ring_features(self):
        mol = safe_mol("CCO")
        summary = structural_summary_from_mol(mol)

        self.assertEqual(summary["has_ring"], 0.0)
        self.assertEqual(summary["has_fused_ring_atom"], 0.0)
        self.assertTrue(np.isfinite(list(summary.values())).all())

    def test_feature_shapes_are_compatible_with_graph_tensor_storage(self):
        topology = structural_topology_from_mol(safe_mol("c1ccccc1"))
        atom = torch.from_numpy(topology.atom_features)
        edge = torch.from_numpy(topology.edge_features)

        self.assertEqual(atom.shape, (6, 4))
        self.assertEqual(edge.shape, (12, 2))
        self.assertEqual(atom.dtype, torch.float32)
        self.assertEqual(edge.dtype, torch.float32)
