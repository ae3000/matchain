import matchain.chain
import matchain.config
import matchain.mtoken
import matchain.prepare
import matchain.util
from tests.utils_for_tests import TestBaseIntegration


class TestConfigAutoCal(TestBaseIntegration):

    def test_config_autocal_dg_with_similarity_fuzzy(self):

        config = matchain.config.get_config('dg')
        props = config['dataset']['props_sim']
        props['name'] = 'fuzzy'
        props['owner'] = 'fuzzy'

        board = matchain.chain.run(config)

        expected = {
            't': 0.3,
            'f1': 0.87248,
            'p': 0.81961,
            'r': 0.93263,
            'tpos': 886,
            'fpos': 195,
            'fneg': 64
        }
        actual = board.evaluation_metrics['union_set']['estimated']
        self.assertDictEqual(actual, expected)

    def test_config_autocal_dg_with_similarity_tfidf_sklearn(self):

        config = matchain.config.get_config('dg')
        props = config['dataset']['props_sim']
        props['name'] = 'tfidf_sklearn'
        props['owner'] = 'tfidf_sklearn'

        board = matchain.chain.run(config)

        expected = {
            't': 0.275,
            'f1': 0.85328,
            'p': 0.81375,
            'r': 0.89684,
            'tpos': 852,
            'fpos': 195,
            'fneg': 98
        }
        actual = board.evaluation_metrics['union_set']['estimated']
        self.assertDictEqual(actual, expected)
