import logging
from typing import Callable

import numpy as np
import pandas as pd
import torch

import matchain.blocking
import matchain.config
import matchain.mtoken
import matchain.prepare
import matchain.similarity
import matchain.util
from tests.utils_for_tests import TestBase


class TestBlocking(TestBase):

    def test_get_candidate_pairs_for_token_blocking(self):
        _, _, _, _, _, _, df_index = self.create_index_dg()

        df_index = matchain.mtoken.TokenIndexCreator.prune_token_index(
            df_index, maximum_token_frequency=2, min_token_length=3)

        candidates = matchain.blocking.TokenBlocking.get_candidate_pairs(
            df_index)

        self.assertEqual(len(candidates), 6)

    def test_run_and_read_candidate_pairs(self):
        _, _, _, size_1, _, _, df_index = self.create_index_dg()
        main_dir = self.get_test_dir_dg_small()
        file_candidates = f'{main_dir}/matches.csv'
        config = {
            'blocking': {
                'name': 'token'
            },
            'autocal': {
                'file_candidates': file_candidates
            }
        }
        candidates, _, _ = matchain.blocking.run(config, None, size_1, df_index)

        self.assertEqual(len(candidates), 4)

    def get_config_blocking_sparse_vectors(self, blocking_name: str, blocking_threshold: float):
        config = {
            'blocking': {
                'name': blocking_name,
                'vector_type': 'shingle_tfidf',
                'shingle_size': 3,
                'query_strategy': 'first',
                'njobs': 1,
                'ntop': 10,
                'blocking_threshold': blocking_threshold
            },
            'dataset': {
                'blocking_props': ['name']
            }
        }
        return config

    def get_candidates_for_test_data_dg(self, blocking_name: str, blocking_threshold: float):
        df_data, _, _, size_1, _, _ = self.load_test_data_dg()
        config = {
            'blocking': {
                'name': blocking_name,
                'vector_type': 'shingle_tfidf',
                'shingle_size': 3,
                'query_strategy': 'first',
                'njobs': 1,
                'ntop': 10,
                'blocking_threshold': blocking_threshold
            },
            'dataset': {
                'blocking_props': ['name']
            }
        }
        candidates, _, _ = matchain.blocking.run(config, df_data, size_1, None)

        for idx_1, idx_2 in candidates:
            print(idx_1, idx_2, df_data.iloc[idx_1]['name'],
                  df_data.iloc[idx_2]['name'])

        return candidates

    def test_run_and_bruteforce_blocking(self):
        candidates = self.get_candidates_for_test_data_dg('bruteforce', 0.5)
        self.assertEqual(len(candidates), 6)

    def test_run_and_bruteforce_blocking_higher_threshold(self):
        candidates = self.get_candidates_for_test_data_dg('bruteforce', 0.9)
        self.assertEqual(len(candidates), 3)

    def test_run_and_sklearn_bruteforce_blocking(self):
        candidates = self.get_candidates_for_test_data_dg('sklearn', 0.5)
        self.assertEqual(len(candidates), 6)

    def test_run_and_sklearn_bruteforce_blocking_higher_threshold(self):
        candidates = self.get_candidates_for_test_data_dg('sklearn', 0.9)
        self.assertEqual(len(candidates), 3)

    def test_run_and_sparsedottopn_blocking(self):
        candidates = self.get_candidates_for_test_data_dg('sparsedottopn', 0.5)
        self.assertEqual(len(candidates), 6)

    def test_run_and_sparsedottopn_blocking_higher_threshold(self):
        candidates = self.get_candidates_for_test_data_dg('sparsedottopn', 0.9)
        self.assertEqual(len(candidates), 3)

    def test_run_and_nmslib_blocking(self):
        # nmslib uses its own random seed
        candidates = self.get_candidates_for_test_data_dg('nmslib', 0.5)
        self.assertEqual(len(candidates), 6)

    def test_run_and_nmslib_blocking_higher_threshold(self):
        # nmslib uses its own random seed
        candidates = self.get_candidates_for_test_data_dg('nmslib', 0.9)
        self.assertEqual(len(candidates), 3)

    def generate_data(self, first_comp_1: int, first_comp_2: int):

        vectors_1 = [
            (first_comp_1, 20, 20, 20, 20),
            (first_comp_1, 20, 20, 20, 30),
            None,
            (first_comp_1, 20, 20, 20, 40),
            None,
            None,
            (first_comp_1, 20, 20, 20, 50),
        ]

        vectors_2 = [(first_comp_1, 20, 20, 20, 20),
                     (first_comp_2, 20, 20, 20, 20),
                     (first_comp_2, 20, 20, 20, 31),
                     (first_comp_2, 20, 20, 20, 32), None,
                     (first_comp_2, 20, 20, 20, 60),
                     (first_comp_2, 20, 20, 20, 70), None]

        vectors = vectors_1.copy()
        vectors.extend(vectors_2)
        return vectors, len(vectors_1)

    def generate_dataframe(self):
        vectors_prop1, size_1 = self.generate_data(1, 2)
        vectors_prop2, _ = self.generate_data(1001, 1002)
        df_data = pd.DataFrame(data={
            'prop1': vectors_prop1,
            'prop2': vectors_prop2
        })
        df_data.index.name = 'idx'
        return size_1, df_data

    def generate_vectors_from_values(self, values: pd.Series):
        vectors = []
        for value in values:
            vectors.append(list(value))
        return np.array(vectors)

    def normalize_vectors(self, a: np.ndarray) -> np.ndarray:
        normalized_a = []
        for i in range(len(a)):
            v = a[i]
            norm = np.linalg.norm(v, ord=2)
            normalized_a.append(v / norm)
        return np.array(normalized_a)

    def convert_search_result(self, result: pd.DataFrame):
        return result['id'], result['score']

    def test_ann_brute_force(self):
        index_vectors = [[1, 2, 3], [7, 8, 9], [4, 5, 6], [10, 11, 12],
                         [13, 14, 15]]
        index_vectors = [[1, 2, 3], [7, 0, 9], [4, 5, 6], [10, 11, 1],
                         [1, 14, 15]]
        index_vectors = self.normalize_vectors(np.array(index_vectors))

        query_vectors = [[4, 4, 6], [4, 5, 6]]
        query_vectors = self.normalize_vectors(np.array(query_vectors))

        bruteforce = matchain.blocking.NNBruteForce(threshold=0)
        df_search = bruteforce.nearest_neighbours(index_vectors, query_vectors,
                                                  3)
        search_index, search_score = self.convert_search_result(df_search)

        exp = [2, 0, 4, 2, 0, 4]
        self.assertListEqual(list(search_index.ravel()), exp)

        expected = [0.995, 0.972, 0.885, 1., 0.974, 0.909]
        for act, exp in zip(search_score.ravel(), expected):
            self.assertAlmostEqual(act, exp, places=2)

    def test_ann_sklearn_brute_force(self):
        index_vectors = [[1, 2, 3], [7, 0, 9], [4, 5, 6], [10, 11, 1],
                         [1, 14, 15]]
        index_vectors = self.normalize_vectors(np.array(index_vectors))

        query_vectors = [[4, 4, 6], [4, 5, 6]]
        query_vectors = self.normalize_vectors(np.array(query_vectors))

        bruteforce = matchain.blocking.NNWrapperSklearn(threshold=0)
        df_search = bruteforce.nearest_neighbours(index_vectors, query_vectors,
                                                  3)
        search_index, search_score = self.convert_search_result(df_search)

        exp = [2, 0, 4, 2, 0, 4]
        self.assertListEqual(list(search_index.ravel()), exp)

        expected = [
            1 - 0.995, 1 - 0.972, 1 - 0.885, 1 - 1., 1 - 0.974, 1 - 0.909
        ]
        for act, exp in zip(search_score.ravel(), expected):
            self.assertAlmostEqual(act, exp, places=2)

    def test_create_value_index_array(self):
        size_1, df_data = self.generate_dataframe()
        df_data = df_data[['prop1']]
        _, df_index_array = matchain.blocking.NearestNeighbourBlocking._create_value_index_array(
            df_data)

        act = list(df_index_array[0])
        exp = [0, 1, None, 2, None, None, 3, 0, 4, 5, 6, None, 7, 8, None]
        self.assertListEqual(exp, act)

    def test_generate_vectors_and_apply_advanced_indexing(self):
        size_1, df_data = self.generate_dataframe()
        df_data = df_data[['prop1', 'prop2']]
        values, df_index_array = matchain.blocking.NearestNeighbourBlocking._create_value_index_array(
            df_data)

        exp = 9 * 2  # number of non-null unique values per prop * 2 props
        self.assertEqual(exp, len(values))

        vectors = self.generate_vectors_from_values(values)
        self.assertEqual(exp, len(vectors))

        mask = df_index_array[0].notnull()
        index = df_index_array[0][mask]
        fancy_vectors = matchain.util.advanced_indexing(vectors, index)
        exp = 10  # 9 non-null unique values plus the first value duplicated
        self.assertEqual(exp, len(fancy_vectors))
        self.assertListEqual(list(fancy_vectors[0]), list(fancy_vectors[4]))

    def test_get_candidate_pairs_brute_force_strategy_smaller(self):
        size_1, df_data = self.generate_dataframe()
        #df_data = df_data[['prop1', 'prop2']]
        blocking_props = ['prop1']
        generate_vectors = self.generate_vectors_from_values
        nearest_neighbours = matchain.blocking.NNBruteForce(
            threshold=0).nearest_neighbours
        act, _, _ = matchain.blocking.NearestNeighbourBlocking.get_candidate_pairs(
            df_data,
            size_1,
            blocking_props,
            generate_vectors,
            nearest_neighbours,
            ntop=3,
            query_strategy='smaller')

        exp = set()
        for idx1 in [1, 3, 6]:
            for idx2 in [7, 8, 9, 10, 12, 13]:
                exp.add((idx1, idx2))
        #print('exp=', exp)
        self.assertSetEqual(set(act), exp)

        return act

    def create_dataframe_from_faiss_search_result(self, search_index,
                                                  search_score):
        n_query_vectors, ntop = search_index.shape
        vec_s = np.ravel(search_score)
        vec_i = np.ravel(search_index)
        vec_query_ids = np.empty(len(vec_i), dtype=int)
        for j in range(n_query_vectors):
            vec_query_ids[j * ntop:(j + 1) * ntop] = j
        df = pd.DataFrame({
            'id': vec_i,
            'query_id': vec_query_ids,
            'score': vec_s
        })
        df = df[df['id'] >= 0]
        df.set_index(['id', 'query_id'], inplace=True)
        #logging.debug('created df=%s\n%s', len(df), df)
        return df

    def generate_random_normalized_vectors(self, dim: int, n_index: int,
                                           n_duplicates: int, n_query: int):
        np.random.seed(1)
        vectors = []
        for _ in range(n_index):
            v = np.random.random(dim)
            # normalize vectors and query when using cosine similarity
            norm = np.linalg.norm(v, ord=2)
            v = np.array(v) / norm
            vectors.append(v)

        v0 = vectors[0]
        for _ in range(n_duplicates):
            vectors.append(v0)
        index_vectors = np.array(vectors, dtype='float32')
        query_vectors = np.array([v0 for _ in range(n_query)], dtype='float32')
        logging.debug('index_vectors=%s, queery_vectors=%s',
                      index_vectors.shape, query_vectors.shape)
        return index_vectors, query_vectors

    def test_faiss_wrapper_cosine_similarity_duplicated_vectors(self):
        '''Generate 5 random normalized 10-dim vectors and duplicate the first vector twice.
            Use the 7 vectors as index for NNWrapperFaiss.
            Use the first vector and duplicate it once.
            Query ntop=20 for these two vectors.
            '''
        index_vectors, query_vectors = self.generate_random_normalized_vectors(
            dim=10, n_index=5, n_duplicates=2, n_query=2)

        wrapper = matchain.blocking.NNWrapperFaiss(threshold=0)
        df = wrapper.nearest_neighbours(index_vectors, query_vectors, ntop=20)

        #search_index, search_score = self.convert_search_result(df_search)

        #logging.debug('search_index = \n%s', search_index)
        #logging.debug('search_score = \n%s', search_score)

        #df = self.create_dataframe_from_faiss_search_result(
        #    search_index, search_score)
        # check that all index vectors are returned (since ntop=20 > 7)
        self.assertEqual(len(df), 2 * 7)

        # check that cosine similarity is used by default and that v0 is returned
        # three times (both for the first and the second query vector)
        df = df[df['score'] > 0.99]
        self.assertEqual(len(df), 2 * 3)

        df.set_index(['id', 'query_id'], inplace=True)
        act = set(df.index.to_list())
        exp = {(0, 0), (5, 0), (6, 0), (0, 1), (5, 1), (6, 1)}
        self.assertSetEqual(act, exp)

    def test_faiss_wrapper_get_candidates_cosine_similarity(self):
        '''Generate 20 random normalized 100-dim vectors.
        Use the 50 vectors as index for NNWrapperFaiss.
        Use the first vector as query with ntop=5.
        Blocking threshold is 0.5.
        '''
        index_vectors, query_vectors = self.generate_random_normalized_vectors(
            dim=100, n_index=20, n_duplicates=0, n_query=1)

        wrapper = matchain.blocking.NNWrapperFaiss(threshold=0.5)
        df = wrapper.nearest_neighbours(index_vectors, query_vectors, ntop=20)

        # 20 vectors are returned for threshold=0.5 for the single query vector
        df.set_index(['id', 'query_id'], inplace=True)
        act = set(df.index.to_list())
        self.assertEqual(len(act), 20)

    def test_faiss_wrapper_get_candidates_cosine_similarity_higher_threshold(self):
        '''Generate 20 random normalized 100-dim vectors.
        Use the 50 vectors as index for NNWrapperFaiss.
        Use the first vector as query with ntop=5.
        Blocking threshold is 0.8.
        '''
        index_vectors, query_vectors = self.generate_random_normalized_vectors(
            dim=100, n_index=20, n_duplicates=0, n_query=1)

        wrapper = matchain.blocking.NNWrapperFaiss(threshold=0.8)
        df = wrapper.nearest_neighbours(index_vectors, query_vectors, ntop=20)

        # three vectors are returned for threshold=0.8 for the single query vector
        df.set_index(['id', 'query_id'], inplace=True)
        act = set(df.index.to_list())
        exp = {(6, 0), (3, 0), (0, 0)}
        self.assertSetEqual(act, exp)

    def get_config(self):
        config_file = './config/mccommands.yaml'
        dataset_name = 'ag'
        config = {}
        conf = matchain.config.read_yaml(config_file)
        split_configs = matchain.config.split_config(conf)
        for conf in split_configs:
            if conf['dataset']['dataset_name'] == dataset_name:
                config = conf
                break

        df_data, size_1, size_2 = matchain.prepare.run(config)

        file_matches = config['dataset']['file_matches']
        matches = matchain.util.read_matches(file_matches,
                                           offset=size_1,
                                           apply_format=False)
        #print('size_1=', size_1, 'size_2=', size_2, 'matches=', len(matches))
        return config, df_data, size_1, size_2, matches

    def create_fct_sentence_transformer(self, config):
        model = config['similarity']['embedding_model']
        device = config['similarity'].get('embedding_device')
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = config['similarity']['embedding_batch_size']
        wrapper = matchain.similarity.SentenceTransformerWrapper(
            model, device, batch_size)
        return wrapper.generate_vectors

    def get_candidates(self, generate_vectors: Callable,
                       nearest_neighbours: Callable, query_strategy: str):
        prop1_1 = [
            'London', None, None, 'New York', None, 'Paris a', 'Paris b',
            'Tokio', None, None
        ]
        prop1_2 = [
            None, 'New York a', 'Paris b', 'Berlin', None, 'Paris', 'Rome',
            'Buenes Aires', None, 'Cairo', 'Tokio 1', 'London 1', None,
            'New York b', 'Tokio 2', 'London 2'
        ]
        prop1 = prop1_1.copy()
        prop1.extend(prop1_2)
        size_1 = len(prop1_1)
        df_data = pd.DataFrame(data={'prop1': prop1})
        df_data.index.name = 'idx'
        blocking_props = ['prop1']
        act, _, _ = matchain.blocking.NearestNeighbourBlocking.get_candidate_pairs(
            df_data,
            size_1,
            blocking_props,
            generate_vectors,
            nearest_neighbours,
            ntop=2,
            query_strategy=query_strategy)

        exp = set([(0, 21), (0, 25), (3, 11), (3, 23), (5, 12), (5, 15),
                   (6, 12), (6, 15), (7, 20), (7, 24)])
        #print('exp=', exp)
        self.assertSetEqual(set(act), exp)

        return act

    def test_candidates_sentence_transformer_brute_force_strategy_larger(self):
        config, _, _, _, _ = self.get_config()
        generate_vectors = self.create_fct_sentence_transformer(config)
        nearest_neighbours = matchain.blocking.NNBruteForce(
            threshold=0).nearest_neighbours

        self.get_candidates(generate_vectors,
                            nearest_neighbours,
                            query_strategy='larger')

    def test_candidates_sentence_transformer_faiss_strategy_larger(self):
        config, _, _, _, _ = self.get_config()
        generate_vectors = self.create_fct_sentence_transformer(config)
        nearest_neighbours = matchain.blocking.NNWrapperFaiss(
            threshold=0).nearest_neighbours

        self.get_candidates(generate_vectors,
                            nearest_neighbours,
                            query_strategy='larger')
