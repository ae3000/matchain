import pandas as pd

import matchain.mtoken
import matchain.util
from tests.utils_for_tests import TestBase


class TestTokenIndexCreator(TestBase):

    def test_tokenize(self):
        tokens = matchain.mtoken.tokenize('ABC_12   DEF')
        self.assertListEqual(tokens, ['abc', '12', 'def'])

    def test_create_index_without_tfidf_properties(self):
        _, _, _, _, _, _, df_index = self.create_index_dg()

        self.assertEqual(len(df_index), 125)
        # assert not columns for tfidf props are created
        self.assertEqual(len(df_index.columns), 5)

        mask = (df_index['count_1'] == 1) & (df_index['count_2'] == 0)
        actual_tokens = df_index[mask].index
        self.assertEqual(len(actual_tokens), 41)
        for token in ['bp', 'temporis', 'res', 'e', 'ii', 'upper', 'holton']:
            self.assertIn(token, actual_tokens)

        mask = (df_index['count_1'] > 1) & (df_index['count_2'] > 1)
        actual_tokens = list(df_index[mask].index)
        self.assertListEqual(sorted(actual_tokens),
                             ['energy', 'npower', 'rwe'])
        row = df_index.loc['rwe']
        self.assertEqual(row['count_1'], 4)
        self.assertEqual(row['count_2'], 3)
        self.assertListEqual(sorted(row['links_1']), [9, 10, 13, 19])
        self.assertListEqual(sorted(row['links_2']), [36, 37, 38])

    def test_create_index_with_tfidf_properties(self):
        _, _, _, _, _, _, df_index = self.create_index_dg(tfidf_props=['name'])

        # assert not columns for tfidf props are created
        self.assertEqual(len(df_index.columns), 7)

        row = df_index.loc['finvoy']
        self.assertListEqual(sorted(row['links_1_name']), [17])
        self.assertListEqual(sorted(row['links_2_name']), [21])

        for token in ['ii', 'upper', 'holton']:
            links = df_index.loc[token]['links_1_name']
            self.assertEqual(len(links), 1)
        for token in ['bp', 'temporis', 'res']:
            links = df_index.loc[token]['links_1_name']
            act_len = len(links) if matchain.util.notnull(links) else 0
            self.assertEqual(act_len, 0)

    def test_prune_token_index_by_maximum_token_frequency(self):
        _, _, _, _, _, _, df_index = self.create_index_dg()

        self.assertEqual(len(df_index), 125)
        df_index_pruned = matchain.mtoken.TokenIndexCreator.prune_token_index(
            df_index, maximum_token_frequency=1, min_token_length=0)

        self.assertEqual(len(df_index_pruned), 103)

    def test_prune_token_index_by_min_token_length(self):
        _, _, _, _, _, _, df_index = self.create_index_dg()

        self.assertEqual(len(df_index), 125)
        df_index_pruned = matchain.mtoken.TokenIndexCreator.prune_token_index(
            df_index, maximum_token_frequency=1000, min_token_length=3)

        self.assertEqual(len(df_index_pruned), 109)


class TestTokenManager(TestBase):

    def test_replace_with_tints(self):
        token2int = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
        src = {'a': {'b', 'c'}, 'd': {'e'}}
        exp = {0: [1, 2], 3: [4]}
        act = matchain.mtoken.TokenManager._replace_with_tints(src, token2int)
        self.assertDictEqual(exp, act)

    def test_add_tokens_readable_true(self):

        df_data, _, _, _, _, _ = self.load_test_data_dg()

        token_manager: matchain.mtoken.TokenManager = matchain.mtoken.TokenManager(
            pd.DataFrame(), pd.DataFrame(), readable=True)

        token_formatter = {
            'name': 'name',
            'owner': 'o',
            'capacity': ['capa', 'float', 3],
            'fuel': 'fuel'
        }
        token_manager.add_tokens(df_data, token_formatter)

        # check int2token
        d = token_manager.int2token
        self.assertEqual(len(d), 142)
        # check that the first tokens correspond to the position of all data records
        last_pos = len(df_data) - 1
        self.assertEqual(d[last_pos], last_pos)
        # check that column values are formatted correctly if readable == True
        self.assertEqual(d[40], '40__name')
        self.assertEqual(d[41], '41__o')
        self.assertEqual(d[42], '42__capa__10.0')
        self.assertEqual(d[43], '43__fuel')
        self.assertEqual(d[121], '121__capa__24.667')
        # check that same column values are converted into same tokens
        fuel_tokens = []
        for _, value in d.items():
            if str(value).endswith('__fuel'):
                fuel_tokens.append(value)
        self.assertEqual(len(fuel_tokens), 6)

        # check token2int
        d = token_manager.token2int
        self.assertEqual(d[last_pos], last_pos)
        self.assertEqual(d['40__name'], 40)
        self.assertEqual(d['121__capa__24.667'], 121)

        # check token2idx
        d = token_manager.token2idx
        token_sse = token_manager.get_token('owner', 'SSE')
        self.assertListEqual(sorted(d[token_sse]), [0, 15])
        token_coal = token_manager.get_token('fuel', 'Coal')
        self.assertListEqual(sorted(d[token_coal]), [19, 36])

        # check idx2token
        d = token_manager.idx2token
        self.assertIn(token_sse, d[0])
        self.assertIn(token_sse, d[15])
        self.assertIn(token_coal, d[19])
        self.assertIn(token_coal, d[36])

    def test_add_tokens_from_index_readable_true(self):

        _, _, _, _, _, _, df_index = self.create_index_dg()
        token_manager = matchain.mtoken.TokenManager(pd.DataFrame(),
                                                   df_index,
                                                   readable=True)

        token_manager.add_tokens_from_index()

        def create_formatted_tokens(tokens):
            formatted_tokens = []
            for tok in tokens:
                pos = df_index.index.get_loc(tok)
                formatted_tokens.append(f'{pos}__tt__{tok}')
            return formatted_tokens

        d = token_manager.token2idx
        token_finvoy = token_manager.get_token(None, 'finvoy')
        token_expected = create_formatted_tokens(['finvoy'])[0]
        self.assertEqual(token_finvoy, token_expected)
        self.assertListEqual(sorted(d[token_finvoy]), [17, 21])

        d = token_manager.idx2token
        tokens_expected = create_formatted_tokens(
            ['finvoy', 'lightsource', 'bp'])
        self.assertListEqual(sorted(d[17]), sorted(tokens_expected))
        tokens_expected = create_formatted_tokens(
            ['finvoy', 'renewable', 'lightsource', 'energy', 'road', '289'])
        self.assertListEqual(sorted(d[21]), sorted(tokens_expected))

    def test_add_tokens_from_index_readable_false(self):

        _, _, _, _, _, _, df_index = self.create_index_dg()
        token_manager = matchain.mtoken.TokenManager(pd.DataFrame(),
                                                   df_index,
                                                   readable=False)

        token_manager.add_tokens_from_index()

        def create_positions(tokens):
            positions = []
            for tok in tokens:
                pos = df_index.index.get_loc(tok)
                positions.append(str(pos))
            return positions

        d = token_manager.token2idx
        token_finvoy = token_manager.get_token(None, 'finvoy')
        exp = create_positions(['finvoy'])[0]
        self.assertEqual(token_finvoy, exp)
        self.assertListEqual(sorted(d[token_finvoy]), [17, 21])

        d = token_manager.idx2token
        exp = create_positions(['finvoy', 'lightsource', 'bp'])
        self.assertListEqual(sorted(d[17]), sorted(exp))
        exp = create_positions(
            ['finvoy', 'renewable', 'lightsource', 'energy', 'road', '289'])
        self.assertListEqual(sorted(d[21]), sorted(exp))

    def test_is_token_from_token_index(self):

        df_data, _, _, _, _, _, df_index = self.create_index_dg()
        token_manager = matchain.mtoken.TokenManager(pd.DataFrame(),
                                                   df_index,
                                                   readable=True)

        token_formatter = {'name': 'name'}
        token_manager.add_tokens(df_data, token_formatter)
        idx2token = token_manager.get_idx2token()
        self.assertEqual(len(idx2token), 40)
        token2idx = token_manager.get_token2idx()
        self.assertEqual(len(token2idx), 37)

        token_manager.add_tokens_from_index()
        tokens = list(token_manager.get_token2idx().keys())
        self.assertEqual(len(tokens), 162)
        n_tokens = token_manager.get_number_of_tokens()
        self.assertEqual(n_tokens, 40 + 162)

        act = token_manager.is_token_from_token_index(tokens[0])
        self.assertEqual(act, False)
        act = token_manager.is_token_from_token_index(tokens[36])
        self.assertEqual(act, False)
        act = token_manager.is_token_from_token_index(tokens[37])
        self.assertEqual(act, True)
        act = token_manager.is_token_from_token_index(tokens[161])
        self.assertEqual(act, True)
