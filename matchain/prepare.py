"""This module contains functions that are exectuted at the beginning of
the command chain, such as logging initialization, data loading,
setting the seed etc."""
import logging
import logging.config
import os
from typing import Optional, Tuple

import pandas as pd

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

    log_cfg: dict = matchain.util.read_yaml_from_file_or_resource(log_config_file)
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

def check_cuda_available(configured_device: str) -> None:
    """Checks if cuda is available and logs a warning if the configured
    device is not consistent with the availability of cuda.

    :param configured_device: the configured device
    :type configured_device: str
    """
    cuda_available = matchain.util.cuda_available()
    message = f'cuda available={cuda_available}, embedding_device={configured_device}'
    if ((cuda_available and configured_device != 'cuda')
        or (not cuda_available and configured_device == 'cuda')):
        logging.warning(message)
    else:
        logging.info(message)

def run(config: dict,
        df1: Optional[pd.DataFrame] = None,
        df2: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, int, int]:
    """Entry point as part of the command chain
    for preparing the data etc. """
    seed = config['prepare']['seed']
    matchain.util.set_seed(seed)

    device = config['similarity'].get('embedding_device')
    check_cuda_available(device)

    if df1 is None:
        file_data = config['dataset']['data_1']
        df1 = load_data(file_data)
    if df2 is None:
        file_data = config['dataset']['data_2']
        df2 = load_data(file_data)

    df_data = concat_data(df1, df2)
    return df_data, len(df1), len(df2)
