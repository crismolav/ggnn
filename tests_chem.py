import unittest
import chem_tensorflow_dense as chem_tfd
import to_graph as tg
import numpy as np
from pdb import set_trace

class ChemTests(unittest.TestCase):
    def test_adj_mat_to_target__simple(self):

        adj_mat = np.array(
            [[[0., 0., 0., 0., 0.],
              [0., 0., 0., 1., 0.],
              [0., 0., 0., 1., 0.],
              [0., 0., 0., 0., 0.]],

             [[0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0.],
              [1., 0., 0., 0., 0.]]])

        result = chem_tfd.adj_mat_to_target(
            adj_mat=adj_mat)

        expected = [[3, 1], [3, 1], [0, 2]]

        self.assertEqual(expected, result)

    def test_graph_to_adj_mat(self):
        graph = [[3, 1, 1], [3, 1, 2], [0, 2, 3]]

        result = chem_tfd.graph_to_adj_mat_dir(
            graph=graph, max_n_vertices=4,
            num_edge_types=2)

        expected = np.array(
            [[[0., 0., 0., 0.],
              [0., 0., 0., 1.],
              [0., 0., 0., 1.],
              [0., 0., 0., 0.]],

             [[0., 0., 0., 0.],
              [0., 0., 0., 0.],
              [0., 0., 0., 0.],
              [1., 0., 0., 0.]]])


        self.assertTrue((expected == result).all())