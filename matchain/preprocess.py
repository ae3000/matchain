import logging
import os
from typing import Tuple

import matchain.util
import pandas as pd

COLUMN_IDX_1 = 'ltable_id'
COLUMN_IDX_2 = 'rtable_id'
COLUMN_LABEL = 'label'
COLUMN_ID = 'id'


class MagellanMovies():

    def __init__(self):
        file_a = 'D:/data/magellan/movies/imdb.csv'
        file_b = 'D:/data/magellan/movies/omdb.csv'
        main_tgt_dir = './tmp_data/magellan'

        df1 = pd.read_csv(file_a, encoding='utf-8', on_bad_lines='warn')
        df1 = df1[['imdbid', 'title', 'year', 'director', 'cast']]
        df1 = self.transform(df1)
        df1['title'] = df1['title'].apply(lambda s: self.split_title(s)[1])
        df1['cast'] = df1['cast'].apply(self.clean_cast)

        df2 = pd.read_csv(file_b,
                          encoding='ansi',
                          on_bad_lines='warn',
                          dtype={'Year': str})
        column_map = {
            'id': 'imdbid',
            'Title': 'title',
            'Year': 'year',
            'Director': 'director',
            'Cast': 'cast'
        }
        df2.rename(columns=column_map, inplace=True)
        df2 = self.transform(df2)

        tgt_dir = f'{main_tgt_dir}/movies_1940'
        df1_masked = df1[df1['year'] == '1940'].copy()
        df2_masked = df2[df2['year'] == '1940'].copy()
        df1_masked, df2_masked, matches = self.sort_and_find_matches(
            df1_masked, df2_masked)
        self.save(df1_masked, df2_masked, matches, tgt_dir)

        tgt_dir = f'{main_tgt_dir}/movies_2010'
        df1_masked = df1[df1['year'] == '2010'].copy()
        df2_masked = df2[df2['year'] == '2010'].copy()
        df1_masked, df2_masked, matches = self.sort_and_find_matches(
            df1_masked, df2_masked)
        self.save(df1_masked, df2_masked, matches, tgt_dir)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info('columns=%s', df.columns)
        df = df[['imdbid', 'title', 'year', 'director', 'cast']].copy()
        logging.info('df=%s, year nan=%s', len(df),
                     len(df[df['year'].isnull()]))
        for col in ['title', 'year', 'director', 'cast']:
            df[col] = df[col].fillna('').astype(str)
        return df

    def remove_brackets(self, s):
        s = s.strip()
        if s.endswith(')'):
            pos = s.rfind('(')
            if pos > 0:
                s = s[:pos].strip()
        return s

    def split_title(self, s):
        s1 = ''
        s2 = ''
        s = s.strip()
        if s.startswith('"'):
            if s.startswith('""'):
                s1 = s[2:]
                i = s1.find('""')
                if i >= 0:
                    s2 = s1[i + 2:]
                    s1 = s1[:i]
            else:
                s1 = s[1:]
                i = s1.find('"')
                if i >= 0:
                    s2 = s1[i + 1:]
                    s1 = s1[:i]
        else:
            s2 = s
        s1 = self.remove_brackets(s1.strip()).strip()
        s2 = self.remove_brackets(s2.strip()).strip()
        return s1, s2

    def clean_cast(self, s: str):
        cleaned_tokens = []
        for token in s.split(','):
            token = self.remove_brackets(token)
            cleaned_tokens.append(token)
        return ', '.join(cleaned_tokens)

    def sort_and_find_matches(
            self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.MultiIndex]:
        size_1 = len(df1)
        df1.sort_values(by='imdbid', inplace=True, ascending=True)
        df1.reset_index(inplace=True, drop=True)
        ser1 = df1['imdbid']
        df1.drop(columns=['imdbid'], inplace=True)

        size_2 = len(df2)
        df2.sort_values(by='imdbid', inplace=True, ascending=True)
        df2.reset_index(inplace=True, drop=True)
        ser2 = df2['imdbid']
        df2.drop(columns=['imdbid'], inplace=True)

        # both df1 / ser1 and df2 / ser2 are sorted by imdbid
        # sort-merge join helps to find all matches efficiently
        # see https://en.wikipedia.org/wiki/Sort-merge_join
        matches = []
        idx_1 = 0
        idx_2 = 0
        while (idx_1 < size_1 and idx_2 < size_2):
            imdbid_1 = ser1[idx_1]
            imdbid_2 = ser2[idx_2]
            if imdbid_1 == imdbid_2:
                pair = (idx_1, idx_2)
                matches.append(pair)
                if idx_1 == size_1 - 1:
                    idx_2 += 1
                else:
                    idx_1 += 1
            elif imdbid_1 < imdbid_2:
                idx_1 += 1
            else:
                idx_2 += 1

        logging.debug('matches=%s', len(matches))
        return df1, df2, pd.MultiIndex.from_tuples(matches)

    def save(self, df1: pd.DataFrame, df2: pd.DataFrame,
             matches: pd.MultiIndex, tgt_dir: str) -> None:
        os.makedirs(tgt_dir, exist_ok=True)

        df1.index.name = COLUMN_ID
        file1 = f'{tgt_dir}/tableA.csv'
        logging.info('saving %s=%s', file1, len(df1))
        df1.to_csv(file1, index=True, encoding='utf-8')

        df2.index.name = COLUMN_ID
        file2 = f'{tgt_dir}/tableB.csv'
        logging.info('saving %s=%s', file2, len(df2))
        df2.to_csv(file2, index=True, encoding='utf-8')

        file_matches = f'{tgt_dir}/matches.csv'
        logging.info('saving %s=%s', file_matches, len(matches))
        matches = matchain.util.sort_pairs(matches)
        df_matches = pd.DataFrame(index=matches)
        df_matches.index.set_names([COLUMN_IDX_1, COLUMN_IDX_2], inplace=True)
        df_matches[COLUMN_LABEL] = 1
        df_matches.to_csv(file_matches, index=True, encoding='utf-8')


if __name__ == "__main__":
    matchain.util.init_console_logging_only()
    MagellanMovies()
