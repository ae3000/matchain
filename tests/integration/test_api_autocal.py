import logging

import pandas as pd

import matchain.api
import matchain.config
import matchain.mtoken
import matchain.prepare
import matchain.util
from tests.utils_for_tests import DefaultDataPaths, TestBaseIntegration


class TestAPIAutoCal(TestBaseIntegration):

    def _autocal_dg_set_properties(self):
        data_dir = './data/Structured/DUKES-GPPDgbr'
        df1 = pd.read_csv(f'{data_dir}/tableA.csv')
        df2 = pd.read_csv(f'{data_dir}/tableB.csv')

        config = DefaultDataPaths.get_file_config_chains()
        mat = matchain.api.MatChain(df1, df2, config=config)

        mat.property('capacity', simfct='relative')
        mat.property('fuel', simfct='equal')
        mat.property('name', simfct='tfidf')
        mat.property('name', simfct='embedding')
        mat.property('owner', simfct='tfidf')
        mat.property('owner', simfct='embedding')

        return mat, data_dir

    def test_autocal_dg_token_complete_including_eval_with_minimum_config(
            self):

        mat, data_dir = self._autocal_dg_set_properties()
        mat.token(blocking_props=['name', 'owner'])
        mat.autocal()
        predicted_matches = mat.predict()
        logging.debug('predicted matches=%s', len(predicted_matches))

        actual = mat.evaluate(matches=data_dir)
        logging.debug('evaluation result=%s', actual)

        expected = {
            't': 0.3,
            'f1': 0.87636,
            'p': 0.9038,
            'r': 0.85053,
            'tpos': 808,
            'fpos': 86,
            'fneg': 142
        }

        self.assertDictEqual(actual['union_set']['estimated'], expected)

    def test_autocal_dg_blocking_sklearn_shingle_complete_including_eval(self):

        mat, data_dir = self._autocal_dg_set_properties()
        mat.blocking(name='sklearn',
                     blocking_props=['name', 'owner'],
                     vector_type='shingle_tfidf',
                     query_strategy='smaller')
        mat.autocal()
        predicted_matches = mat.predict()
        logging.debug('predicted matches=%s', len(predicted_matches))

        actual = mat.evaluate(matches=data_dir)
        logging.debug('evaluation result=%s', actual)

        expected = {
            't': 0.35,
            'f1': 0.89069,
            'p': 0.91648,
            'r': 0.86632,
            'tpos': 823,
            'fpos': 75,
            'fneg': 127
        }

        self.assertDictEqual(actual['union_set']['estimated'], expected)

    def test_autocal_dg_blocking_bruteforce_shingle_complete_including_eval(
            self):

        mat, data_dir = self._autocal_dg_set_properties()
        mat.blocking(name='bruteforce',
                     blocking_props=['name', 'owner'],
                     vector_type='shingle_tfidf',
                     query_strategy='larger')
        mat.autocal()
        predicted_matches = mat.predict()
        logging.debug('predicted matches=%s', len(predicted_matches))

        actual = mat.evaluate(matches=data_dir)
        logging.debug('evaluation result=%s', actual)

        expected = {
            't': 0.275,
            'f1': 0.88199,
            'p': 0.86762,
            'r': 0.89684,
            'tpos': 852,
            'fpos': 130,
            'fneg': 98
        }

        self.assertDictEqual(actual['union_set']['estimated'], expected)

    def test_autocal_dg_blocking_sparsedottopn_shingle_complete_including_eval(
            self):

        mat, data_dir = self._autocal_dg_set_properties()
        mat.blocking(name='sparsedottopn',
                     blocking_props=['name', 'owner'],
                     vector_type='shingle_tfidf',
                     query_strategy='first')
        mat.autocal()
        predicted_matches = mat.predict()
        logging.debug('predicted matches=%s', len(predicted_matches))

        actual = mat.evaluate(matches=data_dir)
        logging.debug('evaluation result=%s', actual)

        expected = {
            't': 0.3,
            'f1': 0.88817,
            'p': 0.90769,
            'r': 0.86947,
            'tpos': 826,
            'fpos': 84,
            'fneg': 124
        }

        # sparse_dot_topn does not allow to set the random seed.
        # Therefore, the results are not reproducible and checking exact equality is not possible.
        # Instead, we check that the f1-score is within a certain range.
        #self.assertDictEqual(actual['union_set']['estimated'], expected)
        act_t = actual['union_set']['estimated']['t']
        self.assertEqual(act_t, 0.3)
        self.assertAlmostf1(actual, expected['f1'], delta=0.05)

    def test_autocal_dg_blocking_nmslib_shingle_complete_including_eval(self):

        mat, data_dir = self._autocal_dg_set_properties()
        mat.blocking(name='nmslib',
                     blocking_props=['name', 'owner'],
                     vector_type='shingle_tfidf',
                     query_strategy='second')
        mat.autocal()
        predicted_matches = mat.predict()
        logging.debug('predicted matches=%s', len(predicted_matches))

        actual = mat.evaluate(matches=data_dir)
        logging.debug('evaluation result=%s', actual)

        expected_0275 = {
            't': 0.275,
            'f1': 0.87584,
            'p': 0.86462,
            'r': 0.88737,
            'tpos': 843,
            'fpos': 132,
            'fneg': 107
        }

        expected_0300 = {
            't': 0.3,
            'f1': 0.88613,
            'p': 0.90919,
            'r': 0.86421,
            'tpos': 821,
            'fpos': 82,
            'fneg': 129
        }

        # The Python bindings of nmslib do not allow to set the random seed.
        # Therefore, the results are not reproducible and checking exact equality is not possible.
        # Instead, we check that the f1-score is within a certain range.
        #self.assertDictEqual(actual['union_set']['estimated'], expected)
        act_t = actual['union_set']['estimated']['t']
        if act_t == 0.275:
            self.assertAlmostf1(actual, expected_0275['f1'], delta=0.05)
        elif act_t == 0.3:
            self.assertAlmostf1(actual, expected_0300['f1'], delta=0.05)
        else:
            raise AssertionError(f'Unexpected threshold: {act_t}')

    def test_autocal_dg_blocking_bruteforce_embedding_complete_including_eval(
            self):

        mat, data_dir = self._autocal_dg_set_properties()
        mat.blocking(name='bruteforce',
                     blocking_props=['name', 'owner'],
                     vector_type='embedding',
                     query_strategy='first')
        mat.autocal()
        predicted_matches = mat.predict()
        logging.debug('predicted matches=%s', len(predicted_matches))

        actual = mat.evaluate(matches=data_dir)
        logging.debug('evaluation result=%s', actual)

        expected = {'t': 0.25, 'f1': 0.86633, 'p': 0.93317, 'r': 0.80842, 'tpos': 768, 'fpos': 55,
                    'fneg': 182}

        self.assertDictEqual(actual['union_set']['estimated'], expected)

    def test_autocal_dg_blocking_faiss_embedding_complete_including_eval(self):

        mat, data_dir = self._autocal_dg_set_properties()
        mat.blocking(name='faiss',
                     blocking_props=['name', 'owner'],
                     vector_type='embedding',
                     query_strategy='first')
        mat.autocal()
        predicted_matches = mat.predict()
        logging.debug('predicted matches=%s', len(predicted_matches))

        actual = mat.evaluate(matches=data_dir)
        logging.debug('evaluation result=%s', actual)

        expected = {'t': 0.25, 'f1': 0.8649, 'p': 0.93407, 'r': 0.80526, 'tpos': 765, 'fpos': 54,
                    'fneg': 185}

        self.assertDictEqual(actual['union_set']['estimated'], expected)

    def test_autocal_ag_token_configuration_with_data_paths_property_mapping(
            self):
        path_data_1, path_data_2, data_dir = DefaultDataPaths.get_data_paths('ag')
        mat = matchain.api.MatChain(path_data_1, path_data_2)

        mat.property('price', simfct='relative')
        mat.property('title', simfct='tfidf')
        mat.property('title', simfct='embedding')
        mat.property('manufacturer', simfct='tfidf')
        mat.property('manufacturer', simfct='embedding')

        mat.token(blocking_props=['title', 'manufacturer'])

        # assert three different property names
        props_sim = mat.board.config['dataset']['props_sim']
        self.assertEqual(len(props_sim), 3)
        # assert five similarity functions
        count = 0
        for _, simfcts in props_sim.items():
            count += 1 if isinstance(simfcts, str) else len(simfcts)
        self.assertEqual(count, 5)

        mat.autocal()
        actual = mat.evaluate(matches=data_dir)
        logging.debug('evaluation result=%s', actual)

        expected = {
            't': 0.175,
            'f1': 0.61611,
            'p': 0.61611,
            'r': 0.61611,
            'tpos': 719,
            'fpos': 448,
            'fneg': 448
        }

        self.assertDictEqual(actual['union_set']['estimated'], expected)

    def test_autocal_dg_token_mtf_50(self):
        mat = self.prepare('dg')

        token_manager = mat.token(maximum_token_frequency=50,
                                  minimum_token_length=3)
        candidate_pairs = mat.blocking(name='token')
        self.assertEqual(len(token_manager.token_index_unpruned), 3890)
        self.assertEqual(len(token_manager.token_index), 3719)
        self.assertEqual(len(candidate_pairs), 13509)

        mat.autocal()
        self.assertf1(mat, 0.87636)

    def test_autocal_dg_token_mtf_20(self):
        mat = self.prepare('dg')

        token_manager = mat.token(maximum_token_frequency=20)
        mat.blocking(name='token')
        self.assertEqual(len(token_manager.token_index_unpruned), 3890)
        self.assertEqual(len(token_manager.token_index), 3670)
        self.assertEqual(len(mat.board.candidate_pairs), 4804)

        mat.autocal()
        self.assertf1(mat, 0.89215)

    def test_autocal_dg_token_readable(self):
        mat = self.prepare('dg')
        mat.token(readable=True)
        mat.autocal()
        self.assertf1(mat, 0.87636)

    def test_autocal_fz_token_similarity_number_of_candidates(self):
        mat = self.prepare('fz')
        df_sim = mat.similarity()
        n_candidates = len(mat.board.candidate_pairs)
        self.assertEqual(len(df_sim), n_candidates)

    def test_autocal_ag_token_similarity_modified_embedding_model_cuda(self):
        mat = self.prepare('ag')
        mat.similarity(embedding_model='all-MiniLM-L6-v2',
                       embedding_batch_size=16,
                       embedding_device='cuda')
        mat.autocal()
        self.assertf1(mat, 0.60852)

    def test_autocal_ag_token_similarity_modified_embedding_model_cpu(self):
        mat = self.prepare('ag')
        mat.similarity(embedding_model='all-MiniLM-L6-v2',
                       embedding_device='cpu')
        mat.autocal()
        self.assertf1(mat, 0.60852)

    def test_autocal_dg_token_similarity_modified_tfidf_maxidf(self):
        mat = self.prepare('dg')
        mat.similarity(tfidf_maxidf=10)
        mat.autocal()
        self.assertf1(mat, 0.87731)
