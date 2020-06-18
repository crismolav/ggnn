import unittest
import to_graph as tg

class ToGraphTests(unittest.TestCase):

    def test_adj_mat_to_target__simple(self):
        sentence_list = [
            '1\tit\t_\tPRP\tPRP\t_\t3\tnsubj\t_\t_\n',
            '2\twas\t_\tVBD\tVBD\t_\t3\tcop\t_\t_\n',
            '3\tMonday\t_\tNNP\tNNP\t_\t0\tROOT\t_\t_\n']

        result = tg.process_sentence(
            sentence_list, problem='identity')

        expected = [[3, 12, 1], [3, 26, 2], [0, 9, 3]]

        self.assertEqual(expected, result['graph'])