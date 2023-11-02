import matchain.chain
import matchain.config
import matchain.mtoken
import matchain.prepare
import matchain.util
from tests.utils_for_tests import TestBaseIntegration


class TestConfigAutoCal(TestBaseIntegration):

    def test_config_autocal_dg_with_similarity_fuzzy(self):

        config = self.get_config('dg')
        props = config['dataset']['props_sim']
        props['name'] = 'fuzzy'
        props['owner'] = 'fuzzy'

        board = matchain.chain.run(config)

        expected = {'t': 0.3, 'f1': 0.87205, 'p': 0.81885, 'r': 0.93263, 'tpos': 886, 'fpos': 196,
                    'fneg': 64}
        actual = board.evaluation_metrics['union_set']['estimated']
        self.assertDictEqual(actual, expected)


        config = self.get_config('dg')
        props = config['dataset']['props_sim']
        props['name'] = 'tfidf_sklearn'
        props['owner'] = 'tfidf_sklearn'

        board = matchain.chain.run(config)

        expected = {'t': 0.275, 'f1': 0.84496, 'p': 0.80729, 'r': 0.88632, 'tpos': 842, 'fpos': 201,
                    'fneg': 108}
        actual = board.evaluation_metrics['union_set']['estimated']
        self.assertDictEqual(actual, expected)
