import logging
import unittest
from typing import Optional, Tuple

import pandas as pd

import matchain.api
import matchain.config
import matchain.mtoken
import matchain.prepare
import matchain.util


class DefaultDataPaths:
    """Default paths for data and config files"""

    config_datasets: dict = {}

    @staticmethod
    def get_dir_main_experiments():
        """Returns default directory for experiments"""
        return './experiments'

    @staticmethod
    def get_dir_main_config():
        """Returns default directory for config files"""
        return './tests/resources/config'

    @staticmethod
    def get_file_config_chains():
        """Returns default config file"""
        main_dir = DefaultDataPaths.get_dir_main_config()
        return f'{main_dir}/mccommands.yaml'

    @staticmethod
    def get_file_datasets():
        """Returns default config file for specifying datasets"""
        main_dir = DefaultDataPaths.get_dir_main_config()
        return f'{main_dir}/mcdatasets.yaml'

    @staticmethod
    def get_data_paths(dataset_name: str) -> Tuple[str, str, str]:
        """Returns paths to data files and true matches file for a given dataset"""
        if not DefaultDataPaths.config_datasets:
            path = DefaultDataPaths.get_file_datasets()
            config = matchain.config.read_yaml(path)
            DefaultDataPaths.config_datasets = matchain.config.resolve_config(config)
        dataset = DefaultDataPaths.config_datasets[dataset_name]
        return dataset['data_1'], dataset['data_2'], dataset['file_matches']


class TestBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        matchain.util.init_console_logging_only()

    def get_test_dir_dg_small(self):
        return './tests/resources/dg_small'

    def load(
        self,
        path_data_1: str,
        path_data_2: str,
        dir_matches: str,
        matches_offset: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, int, int, pd.MultiIndex]:
        df1 = pd.read_csv(path_data_1)
        df2 = pd.read_csv(path_data_2)
        size_1 = len(df1)
        size_2 = len(df2)
        if matches_offset is None:
            matches_offset = size_1
        matches = matchain.util.read_matches(dir_matches,
                                           offset=matches_offset,
                                           apply_format=False)
        return df1, df2, size_1, size_2, matches

    def load_test_data_dg(
        self,
        matches_offset: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int,
               pd.MultiIndex]:
        main_dir = self.get_test_dir_dg_small()
        path_data_1 = f'{main_dir}/tableA.csv'
        path_data_2 = f'{main_dir}/tableB.csv'
        file_matches = f'{main_dir}/matches.csv'
        df1, df2, size_1, size_2, matches = self.load(path_data_1, path_data_2,
                                                      file_matches,
                                                      matches_offset)
        df_data = matchain.prepare.concat_data(df1, df2)
        return df_data, df1, df2, size_1, size_2, matches

    def create_index_dg(
        self,
        tfidf_props: Optional[list] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int,
               pd.MultiIndex, pd.DataFrame]:
        df_data, df1, df2, size_1, size_2, matches = self.load_test_data_dg()
        # sort matches
        matches = matchain.util.sort_pairs(matches)
        blocking_props = ['name', 'owner']
        if tfidf_props is None:
            tfidf_props = []
        tokenize_fct = matchain.mtoken.tokenize
        df_index = matchain.mtoken.TokenIndexCreator.create_index(
            df_data, size_1, blocking_props, tfidf_props, tokenize_fct)
        return df_data, df1, df2, size_1, size_2, matches, df_index

    def get_config(self, dataset_name: str) -> dict:
        file_commands = DefaultDataPaths.get_file_config_chains()
        return matchain.config.get_config(dataset_name, file_commands)

    def get_config_and_matches(self, dataset_name: str):
        config = self.get_config(dataset_name)

        df_data, size_1, size_2 = matchain.prepare.run(config)

        file_matches = config['dataset']['file_matches']
        matches = matchain.util.read_matches(file_matches,
                                           offset=size_1,
                                           apply_format=False)
        #print('size_1=', size_1, 'size_2=', size_2, 'matches=', len(matches))
        return config, df_data, size_1, size_2, matches


class TestBaseIntegration(TestBase):

    def assertf1(self, mat: matchain.api.MatChain, expectedf1) -> None:
        actual = mat.evaluate()
        logging.debug('evaluation result=%s', actual)
        self.assertEqual(actual['union_set']['estimated']['f1'], expectedf1)

    def assertAlmostf1(self, actual, expectedf1, delta) -> None:
        self.assertAlmostEqual(actual['union_set']['estimated']['f1'],
                               expectedf1,
                               delta=delta)

    def load_data(
        self, dataset_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, int, int, pd.MultiIndex]:
        path_data_1, path_data_2, dir_data = DefaultDataPaths.get_data_paths(
            dataset_name)
        return self.load(path_data_1, path_data_2, dir_data)

    def get_config_w2v(self, dataset_name: str, epochs: int,
                   evaluation: bool) -> dict:

        config = self.get_config(dataset_name)

        config['w2vpytorch']['epochs'] = epochs
        if evaluation:
            config['chain'] = [
                'prepare', 'mtoken', 'randomwalk', 'w2vpytorch', 'predict',
                'evaluate'
            ]
        else:
            config['chain'] = [
                'prepare', 'mtoken', 'randomwalk', 'w2vpytorch', 'predict'
            ]

        return config

    def prepare(self,
                dataset_name: str,
                use_similarity=True) -> matchain.api.MatChain:
        config = self.get_config(dataset_name)
        config['chain'] = None
        mat = matchain.api.MatChain(config=config)
        if use_similarity:
            mat.board.config['mtoken']['tfidf_index'] = True
        return mat
