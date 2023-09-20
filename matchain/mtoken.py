"""This module contains classes to create and manage a token index.
A token index is an inverted index from tokens to the set of instance ids
that contain the token. Tokens are created from the values of the properties
such as words in string values but also (rounded) numeric values.
The token index is used for token blocking, tf-idf similarity computation and
for the generation of graphs consisting of instance and token nodes and edges in between.
"""
import logging
import pickle
from collections import defaultdict
from typing import Any, Callable, Dict, List, Set, Union

import matchain.similarity
import matchain.util
import pandas as pd
from tqdm import tqdm


class TokenIndexCreator():
    """
    This class creates a token index from a dataframe.
    The token index can be used for token blocking and tf-idf similarity computation.
    """

    @staticmethod
    def create_index(df_data: pd.DataFrame, size_1: int,
                     blocking_properties: List[str],
                     tfidf_properties: List[str],
                     tokenize_fct: Callable[[str], List[str]]) -> pd.DataFrame:
        """Creates a token index from the given dataframe and returns the token index
        as dataframe where the tokens build the index and the columns contain lists of
        instance ids as values. For each given tfidf property, its values are tokenized and
        the tokens are added to the index. The ids of instances in datasets 1 and 2 containing
        a token are stored in separate columns (e.g. name_links_1 and name_links_2 for the
        property 'name'). Concerning the blocking properties, the referred instance ids are
        not stored property-wise but are merged and stored in the two columns
        'links_1' and 'links_2'.

        :param df_data: The dataframe containing the data of dataset 1 and 2
        :type df_data: pd.DataFrame
        :param size_1: The number of instances in dataset 1
        :type size_1: int
        :param blocking_properties: The properties used for blocking
        :type blocking_properties: List[str]
        :param tfidf_properties: The properties used for tf-idf similarity computation
        :type tfidf_properties: List[str]
        :param tokenize_fct: The function used to tokenize the values
            of the blocking and tfidf properties
        :type tokenize_fct: Callable[[str], List[str]]
        :return: The token index as dataframe
        :rtype: pd.DataFrame
        """
        logging.info(
            'creating index, blocking_properties=%s, tfidf_properties=%s',
            blocking_properties, tfidf_properties)

        if not tfidf_properties:
            tfidf_properties = []

        # convert dataframes to plain dictionaries to fasten iteration
        df_data_1 = df_data.iloc[:size_1]
        data_1 = df_data_1.to_dict(orient='index')
        columns_1 = [str(col) for col in df_data_1.columns]
        df_data_2 = df_data.iloc[size_1:]
        data_2 = df_data_2.to_dict(orient='index')
        columns_2 = [str(col) for col in df_data_2.columns]

        # create complete index for token-blocking and generation of candidate pairs
        # if tfidf properties are given, also create token index for each tfidf property
        complete_index = defaultdict(defaultdict)
        TokenIndexCreator._create_index_from_data(data_1, columns_1, data_2,
                                                  columns_2, complete_index,
                                                  blocking_properties,
                                                  tfidf_properties,
                                                  tokenize_fct)

        df_index = pd.DataFrame.from_dict(complete_index, orient='index')
        # reorder columns
        columns = ['len', 'count_1', 'count_2', 'links_1', 'links_2']
        for prop in tfidf_properties:
            for col in [f'links_1_{prop}', f'links_2_{prop}']:
                if col in df_index.columns:
                    columns.append(col)
        df_index = df_index[columns]

        logging.info('finished creating index for token blocking, columns=%s',
                     [str(c) for c in df_index.columns])

        return df_index

    @staticmethod
    def _create_index_from_data(
            data_1: dict, columns_1: list, data_2: dict, columns_2: list,
            index: dict, blocking_properties: List[str],
            tfidf_properties: List[str],
            tokenize_fct: Callable[[str], List[str]]) -> None:

        columns_list = [columns_1, columns_2]
        blocking_links_list = ['links_1', 'links_2']
        for j, data in enumerate([data_1, data_2]):

            columns = columns_list[j]
            blocking_link = blocking_links_list[j]
            logging.info('columns dataset = %s', [c for c in columns])
            all_props = set(blocking_properties)
            all_props.update(tfidf_properties)

            for prop in all_props:
                if not prop in columns:
                    message = f'blocking property is not a column, property={prop}'
                    logging.warning(message)
                    continue

                is_blocking_prop = prop in blocking_properties
                is_tfidf_prop = prop in tfidf_properties
                TokenIndexCreator._create_index_from_property(
                    data, index, prop, blocking_link, is_blocking_prop,
                    is_tfidf_prop, tokenize_fct)

        for tok, row in index.items():
            index[tok]['len'] = len(tok)
            links = row.get('links_1')
            count = len(links) if links else 0
            index[tok]['count_1'] = count
            links = row.get('links_2')
            count = len(links) if links else 0
            index[tok]['count_2'] = count

    @staticmethod
    def _create_index_from_property(
            data: dict, index: dict, prop: str, blocking_link: str,
            is_blocking_prop: bool, is_tfidf_prop: bool,
            tokenize_fct: Callable[[str], List[str]]) -> None:

        def add(key1, key2, element):
            myset = index[key1].get(key2)
            if myset is None:
                index[key1][key2] = {element}
            else:
                myset.add(element)

        prop_links = f'{blocking_link}_{prop}'
        for idx, row in tqdm(data.items()):
            tokens = tokenize_fct(row[prop])
            for tok in tokens:
                if is_blocking_prop:
                    add(tok, blocking_link, idx)
                if is_tfidf_prop:
                    add(tok, prop_links, idx)

        logging.info('tokens=%s after processing property %s, %s', len(index),
                     prop, blocking_link)

    @staticmethod
    def prune_token_index(df_index: pd.DataFrame, maximum_token_frequency: int,
                          min_token_length: int) -> pd.DataFrame:
        """This method prunes the token index by removing tokens that occur too often or
        are too short.

        :param df_index: the index to prune
        :type df_index: pd.DataFrame
        :param maximum_token_frequency: the maximum number of times a token can occur
        :type maximum_token_frequency: int
        :param min_token_length: the minimum length of a token
        :type min_token_length: int
        :return: the pruned token index
        :rtype: pd.DataFrame
        """
        mask = (df_index['len'] >= min_token_length) & (
            (df_index['count_1'] <= maximum_token_frequency) &
            (df_index['count_2'] <= maximum_token_frequency))
        df_index_tokens_pruned = df_index[mask].copy()

        logging.debug('number of tokens in pruned token index=%s',
                      len(df_index_tokens_pruned))

        mask = ((df_index['len'] >= min_token_length) &
                (df_index['count_1'] <= maximum_token_frequency) &
                (df_index['count_1'] > 0) &
                (df_index['count_2'] <= maximum_token_frequency) &
                (df_index['count_2'] > 0))

        logging.debug(
            'original number of tokens in pruned token index \
            (not used, for comparison with OntoMatch only)=%s',
            len(df_index[mask]))

        return df_index_tokens_pruned


class TokenManager:
    """This class extends and manages tokens from the token index in a different way
    and adds functionality that supports the generation of graphs consisting of instance
    and token nodes and edges in between.
    """

    def __init__(self, token_index_unpruned: pd.DataFrame,
                 token_index_pruned: pd.DataFrame, readable: bool) -> None:
        """Constructor

        :param token_index_unpruned: the original token index
        :type token_index_unpruned: pd.DataFrame
        :param token_index_pruned: the pruned token index
        :type token_index_pruned: pd.DataFrame
        :param readable: whether the token and instance are represented as human-readable strings
            or as integers, useful for debugging
        :type readable: bool
        """
        self.readable = readable
        self.token_index_unpruned = token_index_unpruned
        self.token_index = token_index_pruned
        self.token_index_min = -1
        self.token_index_max = -1

        #TODO-AE 230219 samevaluesametoken is fixed to True; configurable? remove?
        self.samevaluesametoken = True
        self.propertyvalue2token = defaultdict(list)

        # map tokens to set of indices
        self.token2idx = defaultdict(set)
        self.token2idx_int = None
        # map indices to set of tokens
        self.idx2token = defaultdict(set)
        self.idx2token_int = None

        self.int2token = {}
        self.token2int = {}

    def get_idx2token(self) -> Dict[str, Set[str]]:
        """Returns a dictionary mapping an instance string id to a set
        of related tokens.

        :return: a dictionary mapping instance string ids to tokens
        :rtype: Dict[str, Set[str]]
        """
        return self.idx2token

    def get_idx2token_int(self) -> Dict[int, List[int]]:
        """Returns a dictionary mapping an instance integer id to the
        integer ids of related tokens.

        :return: a dictionary mapping instance integer ids to token ids
        :rtype: Dict[int, List[int]]
        """
        return TokenManager._replace_with_tints(self.idx2token, self.token2int)

    def get_token2idx(self) -> Dict[str, Set[str]]:
        """Returns a dictionary mapping a token to a set of related
        instance string ids.

        :return: a dictionary mapping tokens to instance string ids
        :rtype: Dict[str, Set[str]]
        """
        return self.token2idx

    def get_token2idx_int(self) -> Dict[int, List[int]]:
        """Returns a dictionary mapping the integer id of a token
        to a set of integer ids of related instances.

        :return: a dictionary mapping token ids to instance integer ids
        :rtype: Dict[int, List[int]]
        """
        return TokenManager._replace_with_tints(self.token2idx, self.token2int)

    @staticmethod
    def _create_formatter(prop: str, args: list) -> Callable[[Any], Any]:

        def format_int(value: Any):
            val = str(round(float(value)))
            return f'{prop_alias}__{val}'

        def format_float(value: Any):
            val = str(round(float(value), ndigits))
            return f'{prop_alias}__{val}'

        def format_text(value: str):
            return prop_alias

        prop_alias = args[0]
        if len(args) == 1:
            return format_text
        if args[1] == 'int':
            return format_int
        if args[1] == 'float':
            ndigits = args[2]
            return format_float

        message = f'unknown formatting for {prop=} and {args=}'
        raise ValueError(message)

    def add_tokens(self, df_data: pd.DataFrame, token_formatter: dict) -> None:
        """Adds tokens created from values in the given dataframe to the token index.
        This may include non-tokenized string values and (rounded) numeric values.
        token_formatter describes which properties are used and how values are transformed
        into tokens.

        :param df_data: the dataframe
        :type df_data: pd.DataFrame
        :param token_formatter: a dictionary mapping properties to a list of arguments describing
            how values of the property are transformed into tokens
        :type token_formatter: dict
        """
        token_formatter_internal = []
        props = []
        for pos, (prop, args) in enumerate(token_formatter.items()):
            if isinstance(args, str):
                args = [args]
            formatter = TokenManager._create_formatter(prop, args)
            token_formatter_internal.append((prop, pos + 1, formatter))
            props.append(prop)

        # add dataframe indices as tokens
        for idx in df_data.index:
            i = matchain.util.IndexFormatter.as_int(idx)
            self.int2token[i] = idx
            self.token2int[idx] = i

        df_tmp = df_data[props]
        for tup in df_tmp.itertuples(index=True, name=None):
            idx = tup[0]
            i2tset = self.idx2token[idx]

            for prop, pos, formatter in token_formatter_internal:
                value = tup[pos]
                if matchain.util.notnull(value):
                    key = (prop, value)
                    token_list = self.propertyvalue2token.get(key)
                    if (not token_list) or (not self.samevaluesametoken):
                        i = len(self.int2token)
                        if self.readable:
                            val = formatter(value)
                            token = f'{i}__{val}'
                        else:
                            token = str(i)
                        self.int2token[i] = token
                        self.token2int[token] = i
                        self.propertyvalue2token[key].append(token)
                    else:
                        assert len(token_list) == 1
                        token = token_list[0]

                    assert matchain.util.notnull(token)
                    i2tset.add(token)
                    self.token2idx[token].add(idx)

        logging.info(
            'added tokens from data, \
            int2token:%s, idx2token:%s, token2idx:%s, propertyvalue2token:%s',
            len(self.int2token), len(self.idx2token), len(self.token2idx),
            len(self.propertyvalue2token))

    def add_tokens_from_index(self) -> None:
        """The original token index contains words extracted from the
        string-valued blocking properties of the original data. These
        words are added as tokens to tokens created by method add_tokens
        (such as complete string values and numeric values).
        """
        self.token_index_min = len(self.int2token)

        # add only tokens extracted from string-valued properties
        df_tmp = self.token_index[['links_1', 'links_2']]
        for tok, links_1, links_2 in df_tmp.itertuples(index=True, name=None):
            key = (None, tok)
            i = len(self.int2token)
            if self.readable:
                token = f'{i}__tt__{tok}'
            else:
                token = str(i)
            self.int2token[i] = token
            self.token2int[token] = i
            self.propertyvalue2token[key].append(token)

            t2iset = self.token2idx[token]
            if matchain.util.notnull(links_1):
                for idx in links_1:
                    t2iset.add(idx)
                    self.idx2token[idx].add(token)
            if matchain.util.notnull(links_2):
                for idx in links_2:
                    t2iset.add(idx)
                    self.idx2token[idx].add(token)

        self.token_index_max = len(self.int2token) - 1

        added = 1 + self.token_index_max - self.token_index_min
        logging.info(
            'added tokens from (pruned) token index, \
            int2token:%s, idx2token:%s, token2idx:%s, propertyvalue2token:%s',
            len(self.int2token), len(self.idx2token), len(self.token2idx),
            len(self.propertyvalue2token))
        logging.debug('token_index:%s, added:%s, min:%s, max:%s',
                      len(self.token_index), added, self.token_index_min,
                      self.token_index_max)

    def get_token(self, prop: str, value: Any) -> Union[str, int]:
        """Returns the token for the given property and value.

        :param prop: property name
        :type prop: str
        :param value: property value
        :type value: Any
        :raises RuntimeError: if samevaluesametoken is False
        :return: the token as string (if readable is True) or int (otherwise
        :rtype: Union[str, int]
        """
        if not self.samevaluesametoken:
            raise RuntimeError(
                f'calling get_token() is not allowed for {self.samevaluesametoken=}'
            )

        assert matchain.util.notnull(value)
        key = (prop, value)
        token_list = self.propertyvalue2token[key]
        assert len(token_list) == 1
        return token_list[0]

    def get_token_list(self, prop: str,
                       value: Any) -> Union[List[str], List[int]]:
        """Returns the list of tokens for the given property and value.

        :param prop: property name
        :type prop: str
        :param value: property value
        :type value: Any
        :raises RuntimeError: if samevaluesametoken is True
        :return: the list of tokens as strings (if readable is True) or integers (otherwise)
        :rtype: Union[List[str], List[int]]
        """
        if self.samevaluesametoken:
            raise RuntimeError(
                f'calling get_token_list() is not allowed for {self.samevaluesametoken=}'
            )

        assert matchain.util.notnull(value)
        key = (prop, value)
        token_list = self.propertyvalue2token[key]
        assert len(token_list) >= 1
        return token_list

    def is_token_from_token_index(self, token: Union[int, str]) -> bool:
        """Checks whether the given token represents a word from original token index.

        :param token: token as string (if readable is True) or int (otherwise)
        :type token: Union[int, str]
        :return: True if the token represents a word from original token index
        :rtype: bool
        """
        if isinstance(token, str):
            i = int(token.split('__')[0])
        else:
            i = token
        return self.token_index_min <= i <= self.token_index_max

    def save_idx2token(self, file_name: str) -> None:
        """Dumps the dictionary idx2token to a pickle file.

        :param file_name: file name
        :type file_name: str
        """
        with open(file_name, mode='wb') as file:
            pickle.dump(self.idx2token, file)

    def get_number_of_tokens(self) -> int:
        """Returns the number of tokens including word tokens from
        the original token index."""
        return self.token_index_max + 1

    @staticmethod
    def _replace_with_tints(src: Dict[str, Set[str]],
                            token2int: Dict[str, int]) -> Dict[int, List[int]]:
        """Converts src dictionary into an integer-valued dictionary
        by replacing each string from the src dictionary according to token2int.

        :param src: source dictionary
        :type src: Dict[str, List[str]]
        :param token2int: string to integer mapping
        :type token2int: Dict[str, int]
        :return: converted dictionary
        :rtype: Dict[int, List[int]]
        """
        res = {}
        for tok, tokens in src.items():
            tint = token2int[tok]
            tints = [token2int[t] for t in tokens]
            res[tint] = sorted(tints)
        return res


def preprocess_string(s: str) -> str:
    """Replace special characters with space and convert to lower case.

    :param s: input string
    :type s: str
    :return: preprocessed string
    :rtype: str
    """
    #unicodes'\\xa0'
    char_list = [' ']
    char_list.extend(
        ['.', '-', '(', ')', ',', "'", '_', '/', '     ', '    ', '   ', '  '])
    for c in char_list:
        s = s.replace(c, ' ')
    return s.lower().strip()


def tokenize(s: str) -> List[str]:
    """Preprocesses and tokenizes the given string.

    :param s: input string
    :type s: str
    :return: list of tokens
    :rtype: List[str]
    """
    if isinstance(s, str) and s != '':
        s = preprocess_string(s)
        return s.split(' ')
    return []


def run(config, df_data, size_1):
    """Entry point as part of the command chain
    for creating, extending and managing the token index.
    """
    blocking_props = config['dataset']['blocking_props']
    maximum_token_frequency = config['mtoken']['maximum_token_frequency']
    min_token_length = config['mtoken']['minimum_token_length']
    readable = config['mtoken']['readable']

    # check for which properties an individual token index has to be created
    tfidf_index = config['mtoken'].get('tfidf_index')
    command_sim = (config.get('chain') and 'similarity' in config['chain'])
    if tfidf_index or command_sim:
        # compute the token index for each property configured for tfidf cosine similarity
        params_mapping = config['dataset']['props_sim']
        tfidf_props = matchain.similarity.get_tfidf_props(params_mapping)
    else:
        tfidf_props = []

    tokenize_fct = tokenize
    df_index_tokens_unpruned = TokenIndexCreator.create_index(
        df_data, size_1, blocking_props, tfidf_props, tokenize_fct)
    df_index_tokens_pruned = TokenIndexCreator.prune_token_index(
        df_index_tokens_unpruned, maximum_token_frequency, min_token_length)

    token_manager: TokenManager = TokenManager(df_index_tokens_unpruned,
                                               df_index_tokens_pruned,
                                               readable)

    chain = config.get('chain')
    if chain is None or 'autocal' not in chain:
        token_formatter = config['dataset'].get('props')
        if token_formatter:
            token_manager.add_tokens(df_data, token_formatter)
            token_manager.add_tokens_from_index()

    return token_manager
