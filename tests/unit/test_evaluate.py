import logging

import pandas as pd

import matchain.evaluate
from tests.utils_for_tests import TestBase


class TestEvaluate(TestBase):

    def test_compute_metrics_true_matches_are_given(self):
        true_matches = [(0, 10), (1, 11), (2, 12), (3, 13)]
        true_matches = pd.MultiIndex.from_tuples(true_matches)
        predicted_matches = [(2, 12), (3, 13), (4, 14), (5, 15)]
        predicted_matches = pd.MultiIndex.from_tuples(predicted_matches)

        act = matchain.evaluate.compute_metrics(predicted_matches, true_matches)
        exp = {'p': 0.5, 'r': 0.5, 'f1': 0.5, 'tpos': 2, 'fpos': 2, 'fneg': 2}
        self.assertDictEqual(act, exp)

    def test_compute_metrics_true_matches_are_given_zero_true_matches(self):
        true_matches = [(0, 10), (1, 11), (2, 12), (3, 13)]
        true_matches = pd.MultiIndex.from_tuples(true_matches)
        predicted_matches = [(4, 14), (5, 15)]
        predicted_matches = pd.MultiIndex.from_tuples(predicted_matches)

        act = matchain.evaluate.compute_metrics(predicted_matches, true_matches)
        exp = {'p': 0.0, 'r': 0.0, 'f1': 0.0, 'tpos': 0, 'fpos': 2, 'fneg': 4}
        self.assertDictEqual(act, exp)

    def test_compute_metrics_true_matches_are_given_zero_false_matches(self):
        true_matches = [(0, 10), (1, 11), (2, 12), (3, 13)]
        true_matches = pd.MultiIndex.from_tuples(true_matches)
        predicted_matches = [(3, 13)]
        predicted_matches = pd.MultiIndex.from_tuples(predicted_matches)

        act = matchain.evaluate.compute_metrics(predicted_matches, true_matches)
        exp = {'p': 1.0, 'r': 0.25, 'f1': 0.4, 'tpos': 1, 'fpos': 0, 'fneg': 3}
        self.assertDictEqual(act, exp)

    def test_compute_estimated_and_maximum_f1(self):
        _, _, _, size_1, size_2, matches = self.load_test_data_dg()
        predicted_matches = matches[:3]

        logging.debug('matches=%s, predicted matches=%s', len(matches),
                      len(predicted_matches))

        config = {'evaluate': {}}
        act = matchain.evaluate.run(config,
                                  size_1,
                                  size_2,
                                  predicted_matches=predicted_matches,
                                  true_matches=matches)
        exp = {
            'estimated': {
                'f1': 0.85714,
                'p': 1.0,
                'r': 0.75,
                'tpos': 3,
                'fpos': 0,
                'fneg': 1
            }
        }
        self.assertDictEqual(act, exp)

    def test_evaluate(self):
        _, _, _, size_1, size_2, matches = self.load_test_data_dg(
            matches_offset=0)
        predicted_matches = matches[1:]
        main_dir = self.get_test_dir_dg_small()
        config = {
            'evaluate': {
                'compute_max_f1': True
            },
            'dataset': {
                'file_matches': main_dir
            }
        }

        act = matchain.evaluate.run(config,
                                  size_1,
                                  size_2,
                                  predicted_matches=predicted_matches,
                                  threshold=0.1,
                                  true_matches=None)

        exp = {
            't': 0.1,
            'f1': 1.0,
            'p': 1.0,
            'r': 1.0,
            'tpos': 1,
            'fpos': 0,
            'fneg': 0
        }
        self.assertDictEqual(act['test_set']['estimated'], exp)

        exp = {
            't': 0.1,
            'f1': 0.85714,
            'p': 1.0,
            'r': 0.75,
            'tpos': 3,
            'fpos': 0,
            'fneg': 1
        }
        self.assertDictEqual(act['union_set']['estimated'], exp)

    def test_compute_match_frequencies(self):
        _, _, _, size_1, size_2, matches = self.load_test_data_dg(
            matches_offset=0)

        matches = [(16, 11), (17, 1), (18, 14), (19, 16), (16, 1), (16, 2)]
        matches = pd.MultiIndex.from_tuples(matches)

        act = matchain.evaluate.compute_match_frequencies(
            size_1, size_2, matches)

        exp = {
            'match_frequencies_1_to_2': {
                0: 16,
                1: 3,
                3: 1
            },
            'match_frequencies_2_to_1': {
                0: 15,
                1: 4,
                2: 1
            }
        }

        self.assertDictEqual(act, exp)
