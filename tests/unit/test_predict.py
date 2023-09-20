import logging
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import matchain.predict
from tests.utils_for_tests import TestBase


class TestPredict(TestBase):

    def test_predict(self):

        file_emb = 'any'
        file_emb_indices = 'any'
        ntop = 30
        size_1 = 20
        size_2 = 20
        matches = [(17, 1), (16, 11), (18, 14), (19, 16)]

        def most_similar(token: str) -> List[Tuple[str, float]]:
            # function 'predict' should always ignore index '10' from dataset 1
            # index '30' from dataset 2 has a low similarity score
            sim_list = [('10', 0.8), ('30', 0.3)]
            for idx_1, idx_2 in matches:
                int_idx_1 = str(idx_1)
                int_idx_2 = str(idx_2 + size_1)
                if token == int_idx_1:
                    return [(int_idx_2, 0.95)] + sim_list
                if token == int_idx_2:
                    return [(int_idx_1, 0.95)] + sim_list
            return []

        mock = MagicMock(return_value=most_similar)
        with patch('matchain.predict._create_most_similar_function', mock):
            predicted_matches, _ = matchain.predict.predict(
                size_1, size_2, file_emb, file_emb_indices, ntop)

        act = predicted_matches.to_list()
        logging.debug(act)
        self.assertListEqual(sorted(act), sorted(matches))
