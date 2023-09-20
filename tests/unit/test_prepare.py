import collections

import matchain.prepare
from tests.utils_for_tests import TestBase


class TestPrepare(TestBase):

    def test_run(self):

        config = collections.defaultdict(dict)
        config['prepare']['seed'] = 10
        config['prepare']['dir_experiments'] = 'any'
        main_dir = self.get_test_dir_dg_small()
        config['dataset']['data_1'] = f'{main_dir}/tableA.csv'
        config['dataset']['data_2'] = f'{main_dir}/tableB.csv'

        df_data, size_1, size_2 = matchain.prepare.run(config)

        size = size_1 + size_2
        self.assertEqual(len(df_data), size)
