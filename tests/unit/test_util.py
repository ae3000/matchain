import numpy as np
import pandas as pd

import matchain.util
from tests.utils_for_tests import TestBase


class TestUtil(TestBase):

    def test_index_formatting(self):
        formatter = matchain.util.IndexFormatter
        idx = 10
        idx_as_token = formatter.format(idx)
        self.assertEqual(formatter.as_int(idx_as_token), idx)
        self.assertTrue(formatter.is_index(idx_as_token, 20))

    def test_advanced_indexing(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        list_idx = [0, 2, 0]
        act = matchain.util.advanced_indexing(arr, list_idx)
        exp = [1, 2, 3, 7, 8, 9, 1, 2, 3]
        self.assertListEqual(list(np.ravel(act)), exp)

        ser_idx = pd.Series(list_idx)
        act = matchain.util.advanced_indexing(arr, ser_idx)
        self.assertListEqual(list(np.ravel(act)), exp)

        np_idx = np.array(list_idx, dtype=float)
        act = matchain.util.advanced_indexing(arr, np_idx)
        self.assertListEqual(list(np.ravel(act)), exp)

    def test_advanced_indexing_masked_not_None(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        ser_idx = pd.Series([0, 2, None, 0])
        ser_idx = ser_idx[ser_idx.notnull()]
        act = matchain.util.advanced_indexing(arr, ser_idx)
        exp = [1, 2, 3, 7, 8, 9, 1, 2, 3]
        self.assertListEqual(list(np.ravel(act)), exp)

        np_idx = ser_idx.to_numpy()
        act = matchain.util.advanced_indexing(arr, np_idx)
        self.assertListEqual(list(np.ravel(act)), exp)
