"""This module provides the functionality to predict matches between two datasets
if their entities are represented as embedding vectors.
"""
import collections
import logging
from typing import Callable, List, Tuple, cast

import gensim.models
import pandas as pd
from tqdm import tqdm

from matchain.util import IndexFormatter


def _filter_word2vec_model(file_src: str, file_tgt: str, size: int) -> None:
    """The sources file contains embedding vectors for all entities but may also
    contain embeddings for tokens or values. This function filters the embeddings
    for entities and writes them to the target file.

    :param file_src: the source file containing the embeddings
    :type file_src: str
    :param file_tgt: the target file to write the filtered embeddings
    :type file_tgt: str
    :param size: the number of entities
    :type size: int
    """

    with open(file_src, 'r', encoding='utf-8') as file:
        first_line = file.readline()
        n_lines, embedding_dimension = first_line.strip().split(' ')
        filtered_lines = []
        for line in file:
            token, _ = line.split(' ', maxsplit=1)
            if IndexFormatter.is_index(token, size):
                filtered_lines.append(line)

    with open(file_tgt, 'w', encoding='utf-8') as file:
        header = f'{len(filtered_lines)} {embedding_dimension}\n'
        file.write(header)
        for line in filtered_lines:
            file.write(line)

    logging.info(
        'wrote filtered embeddings vectors, before=%s, after=%s, src=%s, tgt=%s',
        n_lines, len(filtered_lines), file_src, file_tgt)


def _load_word2vec_model(
        file_model: str) -> gensim.models.keyedvectors.Word2VecKeyedVectors:
    model = gensim.models.KeyedVectors.load_word2vec_format(
        file_model)  #, unicode_errors='ignore')
    logging.info('loaded word2vec model, vocab:%s', len(model))
    return model


def _create_most_similar_function(file_emb: str, file_emb_indices: str,
                                  size: int,
                                  ntop: int) -> Callable[[str], list]:

    def most_similar(token: str) -> List[Tuple[str, float]]:
        return model.most_similar(token, topn=ntop)

    _filter_word2vec_model(file_emb, file_emb_indices, size)
    model = _load_word2vec_model(file_emb_indices)
    return most_similar


def compute_similarities(size_1: int,
                         size_2: int,
                         most_similar: Callable[[str], list],
                         ignore_unknown_idx: bool = False) -> pd.DataFrame:
    """For each entity of the first dataset, this function computes the most similar
    entities of the second dataset. Likewise, for each entity of the second dataset.
    The number of most similar entities may zero, one or more. The function returns
    a dataframe with all found entity pairs as index. The columns are the
    similarity value and the rank of the entity in the ordered list of most similar entities.

    :param size_1: size of the first dataset
    :type size_1: int
    :param size_2: size of the second dataset
    :type size_2: int
    :param most_similar: a function that returns the most similar entities for a given entity
    :type most_similar: Callable[[str], list]
    :param ignore_unknown_idx: if True, entities with no most similar entities are ignored,
        defaults to False
    :type ignore_unknown_idx: bool, optional
    :raises exp: if ignore_unknown_idx is False and an entity has no most similar entities
    :return: a dataframe with all found entity pairs as index and their similarity value and rank
    :rtype: pd.DataFrame
    """
    # rows is a dictionary with the following structure:
    # { (idx_1, idx_2) : { column1: value1, column2: value2, ...}, ... }
    rows = collections.defaultdict(collections.defaultdict)
    count_unknown_idx = 0
    indices = range(size_1 + size_2)
    for pos, idx in tqdm(enumerate(indices), desc='comp sim'):
        try:
            result = most_similar(str(idx))
        except KeyError as exp:
            # idx not in vocabulary
            if ignore_unknown_idx:
                count_unknown_idx += 1
                continue
            else:
                raise exp

        rank_idx_only = 0
        for token, cosine_sim in result:
            pos_token = cast(int, IndexFormatter.as_int(token))
            if (pos < size_1) and (pos_token >= size_1):
                rank_idx_only += 1
                rows[(idx, pos_token)].update({
                    'sim': cosine_sim,
                    'rank_1': rank_idx_only
                })
            elif (pos >= size_1) and (pos_token < size_1):
                rank_idx_only += 1
                rows[(pos_token, idx)].update({
                    'sim': cosine_sim,
                    'rank_2': rank_idx_only
                })

    # ignore Pylance typing problem
    # since the solution index = list(rows.keys) doesn't work for multi-indices
    # and the line data.index.names = ... raises
    # ValueError: Length of new names must be 1, got 2
    data = pd.DataFrame(
        data=rows.values(),  # type: ignore
        index=rows.keys(),  # type: ignore
        columns=['sim', 'rank_1', 'rank_2'])
    data.index.names = ['idx_1', 'idx_2']

    logging.info('computed similarities, df=%s, unknown indices=%s', len(data),
                 count_unknown_idx)

    return data


def predict(size_1: int, size_2: int, file_emb: str, file_emb_indices: str,
            ntop: int) -> Tuple[pd.MultiIndex, float]:
    """Reads the embedding vectors for entities from the given file and computes
    for each entity, the most similar entities from the other dataset.
    As similarity measure, the cosine similarity is used. The function predicts
    a pairs (a, b) of entities as matches if b is most simlar to a and vice versa.
    The function returns the predicted matches.

    :param size_1: size of the first dataset
    :type size_1: int
    :param size_2: size of the second dataset
    :type size_2: int
    :param file_emb: source file containing the embedding vectors
    :type file_emb: str
    :param file_emb_indices: target file containing the embedding vectors for entities only
    :type file_emb_indices: str
    :param ntop: number of most similar entities to be computed for each entity
    :type ntop: int
    :return: the predicted matches
    :rtype: Tuple[pd.MultiIndex, float]
    """
    size = size_1 + size_2
    most_similar_fct = _create_most_similar_function(file_emb,
                                                     file_emb_indices, size,
                                                     ntop)
    df_scores = compute_similarities(size_1,
                                     size_2,
                                     most_similar_fct,
                                     ignore_unknown_idx=True)

    # Predicted matches are those candidate pairs having a score larger
    # than a given or estimated threshold. Since the index of the second
    # dataset in df_scores starts at size_1, we need to shift the index
    # by - size_1. This ensures that the indices of the predicted matches
    # are in coincidence with the original datasets.
    df_scores.reset_index(inplace=True)
    df_scores['idx_2'] = df_scores['idx_2'] - size_1
    df_scores.set_index(['idx_1', 'idx_2'], inplace=True)

    threshold = 0.
    mask = (df_scores['rank_1'] == 1) & (df_scores['rank_2'] == 1)
    predicted_matches = cast(pd.MultiIndex, df_scores[mask].index)
    logging.info('predicted matches=%s, threshold=%s', len(predicted_matches),
                 threshold)
    return predicted_matches, threshold


def run(config: dict, size_1: int, size_2: int) -> Tuple[pd.MultiIndex, float]:
    """Entry point as part of the command chain
    for predicting matches if entities are represented by embedding vectors.
    """
    # TODO-AE 230818 the nearest neighbor search for each entity should be combined
    # with blocking approaches for vector representations of property values
    file_emb = config['output']['file_embedding']
    file_emb_indices = config['output']['file_embedding_id']
    ntop = config['predict']['ntop']
    return predict(size_1, size_2, file_emb, file_emb_indices, ntop)
