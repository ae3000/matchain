import collections
import logging

import pandas as pd

import matchain.alg.autocal.autocal
import matchain.api
import matchain.similarity
import matchain.util
from tests.utils_for_tests import TestBase


class TestAutoCal(TestBase):

    def test_autocal_complete_dg_small_property_name_only_equal(self):
        data_dir = self.get_test_dir_dg_small()
        df1 = f'{data_dir}/tableA.csv'
        df2 = f'{data_dir}/tableB.csv'

        mat = matchain.api.MatChain(df1, df2)
        mat.property('name', simfct='equal')
        # "fuel" generate 62 candidate pairs
        mat.token(blocking_props=['fuel'])

        mat.autocal()
        predicted_matches = mat.predict()
        logging.debug('predicted matches=%s', len(predicted_matches))

        actual = mat.evaluate(matches=data_dir)
        logging.debug('evaluation result=%s', actual)

        expected = {
            't': 1.0,
            'f1': 0.85714,
            'p': 1.0,
            'r': 0.75,
            'tpos': 3,
            'fpos': 0,
            'fneg': 1
        }

        self.assertDictEqual(actual['union_set']['estimated'], expected)

    def test_autocal_complete_dg_small_property_owner_only_equal(self):
        data_dir = self.get_test_dir_dg_small()
        df1 = f'{data_dir}/tableA.csv'
        df2 = f'{data_dir}/tableB.csv'

        mat = matchain.api.MatChain(df1, df2)
        mat.property('owner', simfct='equal')
        # "owner" generates 24 candidate pairs
        mat.token(blocking_props=['owner'])

        mat.autocal()
        predicted_matches = mat.predict()
        logging.debug('predicted matches=%s', len(predicted_matches))

        actual = mat.evaluate(matches=data_dir)
        logging.debug('evaluation result=%s', actual)

        expected = {
            't': 1.0,
            'f1': 0.33333,
            'p': 0.5,
            'r': 0.25,
            'tpos': 1,
            'fpos': 1,
            'fneg': 3
        }

        self.assertDictEqual(actual['union_set']['estimated'], expected)

    def get_candidate_pairs_for_blocking_property_fuel(self):
        '''
        returns candidate_pairs due to blocking property fuel for
        test case df_small created when runnning
        test_autocal_complete_dg_small_property_name_only_equal()
        '''
        return [(0, 24), (0, 27), (1, 24), (1, 27), (2, 24), (2, 27), (3, 24),
                (3, 27), (4, 24), (4, 27), (5, 24), (5, 27), (6, 24), (6, 27),
                (7, 24), (7, 27), (8, 24), (8, 27), (9, 24), (9, 27), (10, 39),
                (11, 24), (11, 27), (12, 24), (12, 27), (13, 24), (13, 27),
                (14, 24), (14, 27), (15, 24), (15, 27), (16, 21), (16, 22),
                (16, 26), (16, 28), (16, 29), (16, 30), (16, 31), (16, 32),
                (16, 34), (16, 35), (17, 21), (17, 22), (17, 26), (17, 28),
                (17, 29), (17, 30), (17, 31), (17, 32), (17, 34), (17, 35),
                (18, 21), (18, 22), (18, 26), (18, 28), (18, 29), (18, 30),
                (18, 31), (18, 32), (18, 34), (18, 35), (19, 36)]

    def test_run_for_given_similarity_scores(self):
        candidate_pairs = self.get_candidate_pairs_for_blocking_property_fuel()
        assert len(candidate_pairs) == 62
        # candidate_pairs contains 3 out of 4 true matches
        matching_candidate_pairs = [(16, 31), (18, 34), (19, 36)]

        # when using similarity function "equal" for property "name"
        # only candidate pairs in matches_within_index have a property
        # score of 1. All others have property score 0.
        column_data = {}
        for pair in candidate_pairs:
            score = 1. if pair in matching_candidate_pairs else 0.
            column_data[pair] = score

        data = {'0': column_data}
        df_sim = pd.DataFrame(data=data)
        df_sim.index.names = ['idx_1', 'idx_2']
        matchain.util.sort_pairs(df_sim)

        config = collections.defaultdict(dict)
        config['autocal']['delta'] = 0.025
        config['autocal']['threshold'] = 'estimated'
        size_1 = 20
        params_mapping = {'name': 'equal'}
        property_mapping = matchain.similarity.create_property_mapping(
            params_mapping, maxidf=30)

        predicted_matches, _, _ = matchain.alg.autocal.autocal.run(
            config, size_1, df_sim, property_mapping)

        expected = []
        for idx_1, idx_2 in matching_candidate_pairs:
            expected.append((idx_1, idx_2 - size_1))
        self.assertListEqual(list(predicted_matches), expected)

    def test_autocal_complete_dg_small_property_name_only_equal_threshold_method_majority(
            self):
        data_dir = self.get_test_dir_dg_small()
        df1 = f'{data_dir}/tableA.csv'
        df2 = f'{data_dir}/tableB.csv'

        mat = matchain.api.MatChain(df1, df2)
        mat.property('name', simfct='equal')
        # "fuel" generates 62 candidate pairs
        mat.token(blocking_props=['fuel'])

        mat.autocal(threshold_method='majority')
        predicted_matches = mat.predict()
        logging.debug('predicted matches=%s', len(predicted_matches))

        actual = mat.evaluate(matches=data_dir)
        logging.debug('evaluation result=%s', actual)

        expected = {
            'f1': 0.57143,
            'p': 0.4,
            'r': 1.0,
            'tpos': 4,
            'fpos': 6,
            'fneg': 0
        }

        self.assertDictEqual(actual['union_set']['estimated'], expected)
