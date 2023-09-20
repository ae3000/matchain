"""This module contains the logic to create a graph from the data.
Some of the implemented algorithms require a graph."""
import logging
import os

from matchain.mtoken import TokenManager


def create_graph_for_node2vec(idx2token: dict, file_graph: str) -> None:
    """Creates a graph for node2vec. Each idx and each token is
    represented as node. A token could be a string value or a part of a string value
    or any other datatype value. The graph connects an idx node to those nodes that
    represent the tokens of the idx.

    :param idx2token: A dictionary that maps an idx to a list of tokens.
    :type idx2token: dict
    :param file_graph: The file path to store the graph.
    :type file_graph: str
    """
    dir_graph = os.path.dirname(file_graph)
    os.makedirs(dir_graph, exist_ok=True)
    with open(file_graph, 'w', encoding='utf-8') as file:
        count = 0
        for idx, tokens in idx2token.items():
            count += len(tokens)
            for tok in tokens:
                line = f'{idx} {tok}\n'
                file.write(line)
    logging.info('store edgelist to %s, lines=%s', file_graph, count)


def run(config: dict, token_manager: TokenManager) -> None:
    """Entry point as part of the command chain
    to create a graph from the data.
    """
    idx2token = token_manager.get_idx2token()
    file_graph = config['graph']['file_graph']
    create_graph_for_node2vec(idx2token, file_graph)
