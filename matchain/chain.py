"""
Entry point for running a command chain
which specifies preprocessing steps and how to match a dataset pair.
"""
import argparse
import logging
import os
import time
from typing import List, Optional

import matchain.base
import matchain.blocking
import matchain.config
import matchain.evaluate
import matchain.graph
import matchain.mtoken
import matchain.predict
import matchain.prepare
import matchain.similarity
import matchain.util
import matchain.word2vec

# check which matching algorithms and commands are installed
MODULE_NOT_FOUND_DICT_FOR_CHAIN = set()

try:
    import matchain.alg.autocal.autocal
except ModuleNotFoundError:
    MODULE_NOT_FOUND_DICT_FOR_CHAIN.add('autocal')

try:
    import matchain.ext.node2vec.node2vec_main
except ModuleNotFoundError:
    MODULE_NOT_FOUND_DICT_FOR_CHAIN.add('node2vec')

try:
    import matchain.ext.embdi.embdi_main
except ModuleNotFoundError:
    MODULE_NOT_FOUND_DICT_FOR_CHAIN.add('embdi')

try:
    import matchain.ext.randomwalk.randomwalk
except ModuleNotFoundError:
    MODULE_NOT_FOUND_DICT_FOR_CHAIN.add('randomwalk')

try:
    import matchain.ext.weighted.weighted
except ModuleNotFoundError:
    MODULE_NOT_FOUND_DICT_FOR_CHAIN.add('weighted')

try:
    import matchain.ext.w2vpytorch.word2vec
except ModuleNotFoundError:
    MODULE_NOT_FOUND_DICT_FOR_CHAIN.add('w2vpytorch')


def check_commands(config: dict, chain: list) -> None:
    """Checks if all necessary libraries for the configured commands in the chain
    and the configured blocking methods are installed.

    :param config: the configuration dictionary
    :type config: dict
    :param chain: the chain of commands
    :type chain: list
    :raises ValueError: if a necessary library is not installed
    """
    for command in chain:
        if command in [
                'autocal', 'node2vec', 'embdi', 'randomwalk', 'weighted',
                'w2vpytorch'
        ] and command in MODULE_NOT_FOUND_DICT_FOR_CHAIN:
            message = '{command} is configured as part of the command chain but not found.'
            raise ValueError(message.format(command=command))
        elif command == 'blocking':
            matchain.blocking.check_blocking(config)


def run(config: dict) -> matchain.base.PinBoard:
    """
    Starts matching for a single dataset pair and
    iterates over the commands in the chain defined in the configuration.

    :param config: the complete configuration for matching a dataset pair
    :type config: dict
    :return: the results of all chain commands
    :rtype: matchain.base.PinBoard
    """

    startime = time.time()

    config = matchain.config.set_default_values(config)
    log_config_file = config['prepare'].get('log_config_file')
    log_file = config['prepare'].get('log_file')
    matchain.prepare.init_logging(log_config_file, log_file)
    pretty = matchain.util.pretty_format(config)
    logging.info('configuration=\n%s', pretty)

    chain = config['chain']
    name = config['dataset']['dataset_name']
    logging.info('starting matching for name=%s, chain=%s', name, chain)
    check_commands(config, chain)

    board = matchain.base.PinBoard()
    board.config = config

    for command in chain:
        execute_command(command, board)

    diff = time.time() - startime
    board.evaluation_metrics['total_time'] = diff
    logging.info('finished matching, total time=%s', diff)

    return board


def execute_command(command: str, board: matchain.base.PinBoard) -> None:
    """
    Executes a chain command and stores its results to the board.

    :param command: a chain command
    :type command: str
    :param board: the results of previously executed chain commands
    :type board: matchain.base.PinBoard
    :raises RuntimeError: unknown chain command
    """

    startime = time.time()
    logging.info('running command=%s', command)
    config = board.config

    if command == 'prepare':
        board.df_data, board.size_1, board.size_2 = matchain.prepare.run(
            config, board.df_1, board.df_2)

    elif command == 'mtoken':
        board.token_manager = matchain.mtoken.run(config, board.df_data,
                                                  board.size_1)

    elif command == 'blocking':
        token_index = None
        if board.token_manager:
            token_index = board.token_manager.token_index
        board.candidate_pairs, board.blocking_values, board.blocking_vectors \
            = matchain.blocking.run(config, board.df_data, board.size_1, token_index)

    elif command == 'similarity':
        token_index_unpruned = None
        if board.token_manager:
            token_index_unpruned = board.token_manager.token_index_unpruned
        board.df_sim, board.property_mapping = matchain.similarity.run(
            config, board.df_data, board.size_1, board.candidate_pairs,
            token_index_unpruned, board.blocking_values,
            board.blocking_vectors)

    elif command == 'autocal':
        board.predicted_matches, board.threshold, board.predictor = matchain.alg.autocal.autocal.run(
            config, board.size_1, board.df_sim, board.property_mapping)

    elif command == 'graph':
        matchain.graph.run(config, board.token_manager)

    elif command == 'node2vec':
        board.walk_file = matchain.ext.node2vec.node2vec_main.run(config)

    elif command == 'embdi':
        matchain.ext.embdi.embdi_main.run(config, board.df_data, board.size_1)

    elif command == 'randomwalk':
        board.walk_file = matchain.ext.randomwalk.randomwalk.run(
            config, board.df_data, board.size_1, board.token_manager)

    elif command == 'weighted':
        board.predicted_matches, board.threshold, board.predictor = \
            matchain.ext.weighted.weighted.run(config, board.size_1, board.df_sim)

    elif command == 'w2vpytorch':
        matchain.ext.w2vpytorch.word2vec.run(config, board.token_manager,
                                             board.walk_file)

    elif command == 'word2vec':
        word2vec_params = config['word2vec']
        file_embedding = config['output']['file_embedding']
        matchain.word2vec.run(board.walk_file, word2vec_params, file_embedding)

    elif command == 'predict':
        board.predicted_matches, board.threshold = matchain.predict.run(
            config, board.size_1, board.size_2)

    elif command == 'evaluate':
        if board.predicted_matches is None:
            # evaluate only blocking
            board.evaluation_metrics = matchain.evaluate.run(
                config, board.size_1, board.size_2, board.candidate_pairs)
        else:
            # evaluate blocking and matching
            board.evaluation_metrics = matchain.evaluate.run(
                config, board.size_1, board.size_2, board.candidate_pairs,
                board.predicted_matches, board.threshold, board.predictor,
                board.true_matches, board.true_nonmatches)

    else:
        raise RuntimeError(f'Unknown command={command}')

    diff = time.time() - startime
    logging.info('finished command=%s, time=%s', command, diff)

def run_config_file(config_file: str) -> List[Optional[matchain.base.PinBoard]]:
    """Runs matcha for a configuration file.

    :param config_file: the configuration file
    :type config_file: str
    :return: a list containing the results for each dataset pair or None if an exception was raised
    :rtype: List[Optional[matchain.base.PinBoard]]
    """
    config = matchain.config.read_yaml(config_file)
    if config.get('dataset'):
        single_config = matchain.config.resolve_config(config)
        split_configs = [single_config]
    else:
        split_configs = matchain.config.split_config(config)

    print('matching for:')
    for conf in split_configs:
        name = conf['dataset']['dataset_name']
        chain = conf['chain']
        print(f'{name:<10}{chain}')
    print()

    boards = []
    for conf in split_configs:
        try:
            board = run(conf)
            boards.append(board)
        except Exception as exc:  # pylint: disable=W0718
            # continue matching multiple dataset pairs
            # even if an exeption is raised during matching
            # the current dataset pair
            boards.append(None)
            print(exc)
            logging.exception(exc)

    return boards

def main() -> None:
    """
    Entry point to start matcha in a command line shell.
    The parameter --config allows to specify the configuration file.
    The configuration file may specify multiple dataset pairs for matching.
    Matching is performed for each dataset pair independently from each other.
    """

    print('current working directory=', os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=matchain.util.get_resource_name_commands())
    args = parser.parse_args()
    config_file = args.config
    print('config file=', config_file)
    run_config_file(config_file)


if __name__ == '__main__':
    main()
