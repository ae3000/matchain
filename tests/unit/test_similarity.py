import logging

import pandas as pd

import matchain.mtoken
import matchain.similarity
import matchain.util
from tests.utils_for_tests import TestBase


class TestSimilarity(TestBase):

    def create_params_mapping(self) -> dict:
        params_mapping = {}
        function_names = matchain.similarity.get_similarity_function_names()
        for fct in function_names:
            prop = f'prop_{fct}'
            params_mapping[prop] = fct
        return params_mapping

    def test_create_property_mapping(self):
        params_mapping = self.create_params_mapping()
        maxidf = 20
        mapping = matchain.similarity.create_property_mapping(
            params_mapping, maxidf)

        function_names = matchain.similarity.get_similarity_function_names()
        self.assertEqual(len(mapping), len(function_names))

        for propmap in mapping:
            if propmap['sim_fct_name'] == 'tfidf':
                act_maxidf = propmap['sim_fct_params']['maxidf']
                self.assertEqual(act_maxidf, maxidf)

    def test_get_tfidf_props(self):
        params_mapping = self.create_params_mapping()
        props = matchain.similarity.get_tfidf_props(params_mapping)
        self.assertEqual(len(props), 1)


class TestVectorizedSimilarity(TestBase):

    def test_similarity_function_absolute(self):
        sim_fct = matchain.similarity.VectorizedSimilarity.create_similarity_function_from_params(
            fct_name='absolute', cut_off_value=10)

        act = sim_fct(10, 5)
        self.assertEqual(act, 0.5)

        act = sim_fct(10, 20)
        self.assertEqual(act, 0.)

    def test_similarity_function_relative(self):
        sim_fct = matchain.similarity.VectorizedSimilarity.create_similarity_function_from_params(
            fct_name='relative', cut_off_value=1)

        act = sim_fct(10, 8)
        self.assertEqual(act, 0.8)

        act = sim_fct(10, 20)
        self.assertEqual(act, 0.5)

        act = sim_fct(10, 40)
        self.assertEqual(act, 0.25)

    def test_similarity_function_equal(self):
        sim_fct = matchain.similarity.VectorizedSimilarity.create_similarity_function_from_params(
            fct_name='equal', cut_off_value=1)

        act = sim_fct(10, 10.)
        self.assertEqual(act, 1)

        act = sim_fct(10, 11)
        self.assertEqual(act, 0)

        act = sim_fct('any name', 'any name')
        self.assertEqual(act, 1)

        act = sim_fct('any name', 'another name')
        self.assertEqual(act, 0)


class TestSimilarityManager(TestBase):

    def test_compute_vectorized_similarity(self):
        df_data, _, _, size_1, _, matches, _ = self.create_index_dg()

        params_mapping = {'capacity': 'absolute'}
        mapping = matchain.similarity.create_property_mapping(params_mapping,
                                                            maxidf=30)

        manager = matchain.similarity.SimilarityManager(
            df_data,
            size_1,
            candidate_pairs=matches,
            property_mapping=mapping,
            embedding_batch_size=-1,
            embedding_model='',
            embedding_device='',
            df_token_index=pd.DataFrame())
        manager.calculate_similarities()
        df_sim = manager.get_similarity_values()

        logging.debug('df_sim=\n%s', df_sim)

        def sim_abs(x, y) -> float:
            return 1 - min(10, abs(x - y)) / 10

        def actual(index) -> float:
            return float(df_sim.loc[index]['0'])

        self.assertAlmostEqual(actual((16, 31)), sim_abs(24.7, 24.66717), 4)
        self.assertAlmostEqual(actual((17, 21)), sim_abs(6.0, 5.9), 4)
        self.assertAlmostEqual(actual((18, 34)), sim_abs(7.7, 7.68222), 4)
        self.assertAlmostEqual(actual((19, 36)), sim_abs(1559.0, 1586.0), 4)

    def test_compute_pseudo_vectorized_similarity_fuzzy(self):
        df_data, _, _, size_1, _, matches, _ = self.create_index_dg()

        params_mapping = {'name': 'fuzzy'}
        mapping = matchain.similarity.create_property_mapping(params_mapping)

        manager = matchain.similarity.SimilarityManager(
            df_data,
            size_1,
            candidate_pairs=matches,
            property_mapping=mapping,
            embedding_batch_size=-1,
            embedding_model='',
            embedding_device='',
            df_token_index=pd.DataFrame())
        manager.calculate_similarities()
        df_sim = manager.get_similarity_values()

        logging.debug('df_sim=\n%s', df_sim)

        def actual(index) -> float:
            return float(df_sim.loc[index]['0'])

        self.assertAlmostEqual(actual((16, 31)), 1.0, 4)
        self.assertAlmostEqual(actual((17, 21)), 1.0, 4)
        self.assertAlmostEqual(actual((18, 34)), 1.0, 4)
        self.assertAlmostEqual(actual((19, 36)), 1.0, 4)

    def test_compute_pseudo_vectorized_similarity_tfidf_sklearn(self):
        df_data, _, _, size_1, _, matches, _ = self.create_index_dg()

        params_mapping = {'name': 'tfidf_sklearn'}
        mapping = matchain.similarity.create_property_mapping(params_mapping)

        manager = matchain.similarity.SimilarityManager(
            df_data,
            size_1,
            candidate_pairs=matches,
            property_mapping=mapping,
            embedding_batch_size=-1,
            embedding_model='',
            embedding_device='',
            df_token_index=pd.DataFrame())
        manager.calculate_similarities()
        df_sim = manager.get_similarity_values()

        logging.debug('df_sim=\n%s', df_sim)

        def actual(index) -> float:
            return float(df_sim.loc[index]['0'])

        self.assertAlmostEqual(actual((16, 31)), 1.0, 4)
        self.assertAlmostEqual(actual((17, 21)), 0.533918, 4)
        self.assertAlmostEqual(actual((18, 34)), 1.0, 4)
        self.assertAlmostEqual(actual((19, 36)), 1.0, 4)

    def test_compute_tfidf_similarity(self):
        df_data, _, _, size_1, _, matches, df_index = self.create_index_dg(
            tfidf_props=['name'])

        params_mapping = {'name': 'tfidf'}
        mapping = matchain.similarity.create_property_mapping(params_mapping,
                                                            maxidf=30)

        manager = matchain.similarity.SimilarityManager(df_data,
                                                      size_1,
                                                      candidate_pairs=matches,
                                                      property_mapping=mapping,
                                                      embedding_batch_size=-1,
                                                      embedding_model='',
                                                      embedding_device='',
                                                      df_token_index=df_index)
        manager.calculate_similarities()
        df_sim = manager.get_similarity_values()

        logging.debug('df_sim=\n%s', df_sim)

        def actual(index) -> float:
            return float(df_sim.loc[index]['0'])

        self.assertAlmostEqual(actual((16, 31)), 1.0, 4)
        self.assertAlmostEqual(actual((17, 21)), 0.528706, 4)
        self.assertAlmostEqual(actual((18, 34)), 1.0, 4)
        self.assertAlmostEqual(actual((19, 36)), 1.0, 4)

    def test_compute_embedding_similarity(self):
        df_data, _, _, size_1, _, matches, _ = self.create_index_dg()

        params_mapping = {'name': 'embedding'}
        mapping = matchain.similarity.create_property_mapping(params_mapping,
                                                            maxidf=30)

        manager = matchain.similarity.SimilarityManager(
            df_data,
            size_1,
            candidate_pairs=matches,
            property_mapping=mapping,
            embedding_batch_size=16,
            embedding_model='stsb-roberta-base',
            embedding_device='cpu',
            df_token_index=pd.DataFrame())
        manager.calculate_similarities()
        df_sim = manager.get_similarity_values()

        logging.debug('df_sim=\n%s', df_sim)

        def actual(index) -> float:
            return float(df_sim.loc[index]['0'])

        self.assertAlmostEqual(actual((16, 31)), 1.0, 4)
        self.assertAlmostEqual(actual((17, 21)), 0.768752, 4)
        self.assertAlmostEqual(actual((18, 34)), 1.0, 4)
        self.assertAlmostEqual(actual((19, 36)), 1.0, 4)

    def test_compute_embedding_similarity_empty_values(self):

        data = [{
            'id': 0,
            'name': 'Abbots Ripton'
        }, {
            'id': 1,
            'name': ''
        }, {
            'id': 2,
            'name': None
        }, {
            'id': 3
        }, {
            'id': 4,
            'name': 'Abbots Ripton'
        }, {
            'id': 5,
            'name': ''
        }, {
            'id': 6,
            'name': None
        }, {
            'id': 7
        }]
        df_data = pd.DataFrame(data)
        size_1 = 4
        matches = pd.MultiIndex.from_tuples([(0, 4), (0, 5), (0, 6), (1, 5),
                                             (2, 6), (3, 7)])

        params_mapping = {'name': 'embedding'}
        mapping = matchain.similarity.create_property_mapping(params_mapping,
                                                            maxidf=30)

        manager = matchain.similarity.SimilarityManager(
            df_data,
            size_1,
            candidate_pairs=matches,
            property_mapping=mapping,
            embedding_batch_size=16,
            embedding_model='stsb-roberta-base',
            embedding_device='cpu',
            df_token_index=pd.DataFrame())
        manager.calculate_similarities()
        df_sim = manager.get_similarity_values()
        logging.debug('df_sim=\n%s', df_sim)

        def actual(index) -> float:
            return float(df_sim.loc[index]['0'])

        # Remark: pd.read_csv() sets empty strings to nan
        # similarity value for 'Abbots Ripton' and ''
        self.assertAlmostEqual(actual((0, 5)), 0.1999, 4)
        # similarity value for '' and ''
        self.assertAlmostEqual(actual((1, 5)), 1.0, 4)
        # no similarity value for None
        self.assertTrue(pd.isnull(df_sim.loc[0, 6]['0']))
        self.assertTrue(pd.isnull(df_sim.loc[3, 7]['0']))
