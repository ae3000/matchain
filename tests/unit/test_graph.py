from unittest.mock import mock_open, patch

import matchain.graph
from tests.utils_for_tests import TestBase


class TestGraph(TestBase):

    def test_create_graph_for_node2vec(self):
        idx2token = {0: [40, 41], 1: [41, 42, 43]}

        mock = mock_open()
        with patch('builtins.open', mock):
            matchain.graph.create_graph_for_node2vec(
                idx2token, file_graph='D://tmp3/testgraph.txt')

        count = 0
        for mock_call in mock.mock_calls:
            if 'write' in repr(mock_call):
                count += 1
        self.assertEqual(count, 5)
