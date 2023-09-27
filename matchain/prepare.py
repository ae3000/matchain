"""This module contains functions that are exectuted at the beginning of
the command chain, such as logging initialization, data loading,
setting the seed etc."""
import logging
import logging.config
import os
from typing import Optional, Tuple

import pandas as pd
import yaml

import matchain.util


def init_logging(log_config_file: str, log_file: str) -> None:
    """Initializes logging. Logs to console only
    if log_config_file or log_file is missing.

    :param log_config_file: yaml file for configuring logging
    :type log_config_file: str
    :param log_file: log file
    :type log_file: str
    """
    if (not log_config_file) or (not log_file):
        matchain.util.init_console_logging_only()
        return

    print('initializing logging with log config file=', log_config_file,
          ', log file=', log_file)
    with open(log_config_file, 'r', encoding='utf-8') as file:
        log_cfg: dict = yaml.safe_load(file.read())

    log_cfg['handlers']['file_handler']['filename'] = log_file
    dir_name = os.path.dirname(log_file)
    os.makedirs(dir_name, exist_ok=True)

    logging.config.dictConfig(log_cfg)


def load_data(file_data: str) -> pd.DataFrame:
    """Loads data from file.

    :param file_data: path to data file
    :type file_data: str
    :return: data frame
    :rtype: pd.DataFrame
    """
    return matchain.util.read_csv(file_data, offset=0, apply_format=False)


def concat_data(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Concats the data from dataset 1 and 2 and resets the index.

    :param df1: data frame 1
    :type df1: pd.DataFrame
    :param df2: data frame 2
    :type df2: pd.DataFrame
    :return: concatenated data frame
    :rtype: pd.DataFrame
    """
    df_data = pd.concat([df1, df2])
    df_data.reset_index(inplace=True)
    df_data.index.name = 'id'
    logging.info('size_1=%s, size_2=%s, concat df_data=%s', len(df1), len(df2),
                 len(df_data))
    return df_data


def run(config: dict,
        df1: Optional[pd.DataFrame] = None,
        df2: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, int, int]:
    """Entry point as part of the command chain
    for preparing the data etc. """
    seed = config['prepare']['seed']
    matchain.util.set_seed(seed)

    dir_name = config['prepare']['dir_experiments']
    os.makedirs(dir_name, exist_ok=True)

    if df1 is None:
        file_data = config['dataset']['data_1']
        df1 = load_data(file_data)
    if df2 is None:
        file_data = config['dataset']['data_2']
        df2 = load_data(file_data)

    df_data = concat_data(df1, df2)
    return df_data, len(df1), len(df2)
