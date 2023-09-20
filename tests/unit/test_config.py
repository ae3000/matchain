from unittest.mock import MagicMock, mock_open, patch

import matchain.config
from tests.utils_for_tests import TestBase


class TestConfig(TestBase):

    def get_nested_yaml_data_as_string(self):
        return '''
        dataset1:
          name: DG
          data_1: "{dir1}/{name}/tableA.csv"
          data_2: "{dir1}/{name}/tableB.csv"
        log_file: "{dir2}/{name}_log.txt"
        misc:
          seed: 12
          misc:
            seed: 14
            misc:
              seed: 16
        args:
          - "{dir1}/results"
          - False
          - 13
        test1:
          test2:
            test3: "seed_{seed}"
        variables:
          dir1: C:/tmp1
          dir2: C:/tmp2
        '''

    def get_yaml_data_as_string_for_multiple_variable_resolution(self):
        return '''
        dataset:
          dataset_name: DG
        chain:
          chain_name: chain_1
        my_time: "{current_time}"
        dir_experiments: "./tmp_{my_time}"
        prepare:
          sub_dir: "{dir_experiments}/{chain_name}"
          log_file: "{sub_dir}/{dataset_name}_log.txt"
        '''

    def get_yaml_data_with_two_configured_datasets_as_string(self):
        return '''
        dir_data: D:/data
        FZ:
          type: dataset
          data_1: "{dir_data}/{dataset_name}/tableA.csv"
          data_2: "{dir_data}/{dataset_name}/tableB.csv"
        DG:
          type: dataset
          data_1: "{dir_data}/{dataset_name}/tableA.csv"
          data_2: "{dir_data}/{dataset_name}/tableB.csv"
        select_datasets: [DG]
        seed: 36
        dir_experiments: D:/experiments
        log_file: "{dir_experiments}/{dataset_name}_log.txt"
        '''

    def test_collect_variables(self):
        data = self.get_nested_yaml_data_as_string()
        mock = mock_open(read_data=data)
        with patch('builtins.open', mock):
            config = matchain.config.read_yaml('some_file')

        act = {}
        matchain.config._collect_variables(variables=act, dictionary=config)
        exp = {'name': 'DG', 'seed': 16, 'dir1': 'C:/tmp1', 'dir2': 'C:/tmp2'}
        self.assertDictEqual(act, exp)

    def test_replace_list_of_strings(self):
        strings = ['{dir1}/file1.txt', '{dir2}/file2.txt']
        variables = {'dir1': 'C:/tmp1', 'dir2': 'C:/tmp2'}
        act = matchain.config._replace_variables(variables, strings)
        exp = ['C:/tmp1/file1.txt', 'C:/tmp2/file2.txt']
        self.assertListEqual(act, exp)

    def test_replace_single_string(self):
        strings = '{dir1}/file1.txt'
        variables = {'dir1': 'C:/tmp1', 'dir2': 'C:/tmp2'}
        act = matchain.config._replace_variables(variables, strings)
        exp = 'C:/tmp1/file1.txt'
        self.assertEqual(act, exp)

    def test_replace_list_of_mixed_objects(self):
        strings = ['{dir1}/file1.txt', 4, True]
        variables = {'dir1': 'C:/tmp1', 'dir2': 'C:/tmp2'}
        act = matchain.config._replace_variables(variables, strings)
        exp = ['C:/tmp1/file1.txt', 4, True]
        self.assertListEqual(act, exp)

    def test_resolve_variables(self):
        data = self.get_nested_yaml_data_as_string()
        mock = mock_open(read_data=data)
        with patch('builtins.open', mock):
            config = matchain.config.read_yaml('some_file')

        act = matchain.config.resolve_config(config)
        print(act)

        exp = {
            'dataset1': {
                'name': 'DG',
                'data_1': 'C:/tmp1/DG/tableA.csv',
                'data_2': 'C:/tmp1/DG/tableB.csv'
            },
            'log_file': 'C:/tmp2/DG_log.txt',
            'misc': {
                'seed': 12,
                'misc': {
                    'seed': 14,
                    'misc': {
                        'seed': 16
                    }
                }
            },
            'args': ['C:/tmp1/results', False, 13],
            'test1': {
                'test2': {
                    'test3': 'seed_16'
                }
            },
            'variables': {
                'dir1': 'C:/tmp1',
                'dir2': 'C:/tmp2'
            }
        }
        self.assertDictEqual(exp, act)

    def test_resolve_variables_multiple_times(self):
        data = self.get_yaml_data_as_string_for_multiple_variable_resolution()
        mock = mock_open(read_data=data)
        with patch('builtins.open', mock):
            config = matchain.config.read_yaml('some_file')

        mock = MagicMock(return_value='250101_083005')
        with patch('matchain.config._get_current_time', mock):
            act = matchain.config.resolve_config(config)
        print(act)

        exp = {
            'dataset': {
                'dataset_name': 'DG'
            },
            'chain': {
                'chain_name': 'chain_1'
            },
            'my_time': '250101_083005',
            'dir_experiments': './tmp_250101_083005',
            'prepare': {
                'sub_dir': './tmp_250101_083005/chain_1',
                'log_file': './tmp_250101_083005/chain_1/DG_log.txt'
            }
        }
        self.assertDictEqual(exp, act)

    def test_split_configs(self):
        data = self.get_yaml_data_with_two_configured_datasets_as_string()
        mock = mock_open(read_data=data)
        with patch('builtins.open', mock):
            config = matchain.config.read_yaml('some_file')

        act = matchain.config.split_config(config)
        # only one dataset is selected
        self.assertEqual(1, len(act))

        exp = {
            'dir_data': 'D:/data',
            'seed': 36,
            'dir_experiments': 'D:/experiments',
            'log_file': 'D:/experiments/DG_log.txt',
            'dataset': {
                'data_1': 'D:/data/DG/tableA.csv',
                'data_2': 'D:/data/DG/tableB.csv',
                'dataset_name': 'DG'
            }
        }

        self.assertDictEqual(exp, act[0])

    def test_get_data_paths(self):
        for dataset_name in ['fz', 'dg', 'kg', 'ag', 'da', 'ds']:
            path_data_1, path_data_2, data_dir = matchain.config.DefaultDataPaths.get_data_paths(
                dataset_name)

            self.assertEqual(path_data_1.index(data_dir), 0)
            self.assertEqual(path_data_2.index(data_dir), 0)
