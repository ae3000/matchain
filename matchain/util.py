"""Utility functions for matchain."""
import importlib.resources
import logging
import logging.config
import os
import os.path
import pprint
import random
from typing import Any, List, Optional, Union, cast

import numpy as np
import pandas as pd
import torch
import yaml


class IndexFormatter():
    """Class to switch between integer values for identifying entities
    (positions in data files plus optional offset for the second dataset)
    and string values.
    """

    @staticmethod
    def format(index: int) -> str:
        return f'{index}__id'

    @staticmethod
    def as_int(index: str) -> Optional[int]:
        if isinstance(index, str):
            return int(index.split('__')[0])
        return index

    @staticmethod
    def is_index(token: str, size: int) -> bool:
        try:
            i = int(token)
            return i < size
        except ValueError:
            return token.endswith('__id')


def init_console_logging_only() -> None:
    log_format = '%(asctime)s %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)
    logging.warning('logging to console only')


def set_seed(seed: int) -> None:
    '''Set initial random seed to improve reproducibility on the same hardware / environment,
    see https://pytorch.org/docs/stable/notes/randomness.html .
    '''
    logging.info('setting seed=%s', seed)

    # see https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    # benchmark = False causes cuDNN to deterministically select an algorithm,
    # possibly at the cost of reduced performance.
    #torch.backends.cudnn.benchmark = False

    # additional settings
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Switch to deterministic algorithms (mode=True)
    # and if this is not possible completely throw a runtime error (warn_only=False)
    #torch.use_deterministic_algorithms(mode=True, warn_only=False)

def cuda_available() -> bool:
    return torch.cuda.is_available()

def pretty_format(config: dict, depth=None) -> str:
    ppr = pprint.PrettyPrinter(sort_dicts=True, depth=depth)
    return ppr.pformat(config)


def notnull(value: Any) -> bool:
    return pd.notnull(value)


def isnull(value: Any) -> bool:
    return pd.isnull(value)


def read_csv(file_name: str,
             offset: int = 0,
             apply_format: bool = True) -> pd.DataFrame:
    data = pd.read_csv(file_name)

    if 'id' in [str(c) for c in data.columns]:
        if apply_format:
            data['id'] = data['id'].apply(
                lambda x: IndexFormatter.format(x + offset))
        elif offset > 0:
            data['id'] = data['id'].apply(lambda x: x + offset)
        data.set_index(['id'], inplace=True)
        return data
    elif 'ltable_id' in [str(c) for c in data.columns]:
        data.rename(columns={
            'ltable_id': 'idx_1',
            'rtable_id': 'idx_2'
        },
                    inplace=True)

    if apply_format:
        data['idx_1'] = data['idx_1'].apply(IndexFormatter.format)
        data['idx_2'] = data['idx_2'].apply(
            lambda x: IndexFormatter.format(x + offset))
    elif offset > 0:
        data['idx_2'] = data['idx_2'].apply(lambda x: x + offset)
    data.set_index(['idx_1', 'idx_2'], inplace=True)
    return data


def read_and_concat_csv_table_ditto_format(
        test_file: str,
        offset: int = 0,
        apply_format: bool = True) -> pd.DataFrame:
    """Ditto is an entity matching system based on language models.
    It uses training, validation and test files in csv format
    which contain matching and non-matching entity pairs.
    This function reads the given test file and concatenates its pairs
    with those of the training and validation files.

    :param test_file: path to the test file
    :type test_file: str
    :param offset: added to the row position of the second dataset, defaults to 0
    :type offset: int, optional
    :param apply_format: If True, use string indices instead of integers, defaults to True
    :type apply_format: bool, optional
    :return: _description_
    :rtype: pd.DataFrame
    """

    if test_file.endswith('matches.csv'):
        df_matches = read_csv(test_file, offset, apply_format)
        return df_matches

    df_test = read_csv(test_file, offset, apply_format)
    train_file = test_file.replace('test.csv', 'train.csv')
    df_train = read_csv(train_file, offset, apply_format)
    val_file = test_file.replace('test.csv', 'valid.csv')
    df_val = read_csv(val_file, offset, apply_format)
    df_union = pd.concat([df_train, df_val, df_test])
    return df_union


def read_matches(source: str,
                 offset: int = 0,
                 apply_format: bool = True) -> pd.MultiIndex:

    if os.path.isdir(source):
        test_file = f'{source}/test.csv'
        df_tmp = read_and_concat_csv_table_ditto_format(
            test_file, offset, apply_format)
    else:
        df_tmp = read_csv(source, offset, apply_format)

    df_matches = df_tmp[df_tmp['label'] == 1]
    return cast(pd.MultiIndex, df_matches.index)


def rnd(number: float) -> float:
    return round(number, 5)


def sort_pairs(unsorted: Union[pd.DataFrame, pd.MultiIndex]) -> pd.MultiIndex:
    """Given (candidate or matching) pairs as multi index or as part of a data frame,
    this function sorts them by the first level index.

    :param unsorted: pairs to sort
    :type unsorted: Union[pd.DataFrame, pd.MultiIndex]
    :return: sorted pairs
    :rtype: pd.MultiIndex
    """
    if isinstance(unsorted, pd.MultiIndex):
        unsorted = pd.DataFrame(index=unsorted)
    unsorted.sort_index(level=0, inplace=True)
    return cast(pd.MultiIndex, unsorted.index)


def add_to_index(dframe: pd.DataFrame, offset_1: int, offset_2: int) -> None:
    """Adds to idx_1 and idx_2 of dframe the given offsets inplace.

    :param dframe: dataframe to modify
    :type dframe: pd.DataFrame
    :param offset_1: offset to add to idx_1
    :type offset_1: int
    :param offset_2: offset to add to idx_2
    :type offset_2: int
    """
    dframe.reset_index(inplace=True)
    if offset_1 != 0:
        dframe['idx_1'] = dframe['idx_1'] + offset_1
    if offset_2 != 0:
        dframe['idx_2'] = dframe['idx_2'] + offset_2
    dframe.set_index(['idx_1', 'idx_2'], inplace=True)


def advanced_indexing(
        arr: np.ndarray, index: Union[List[int], np.ndarray,
                                      pd.Series]) -> np.ndarray:
    """Applies numpy's fancy integer indexing to the given array,
    see https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

    :param arr: array
    :type arr: np.ndarray
    :param index: index to apply to the array
    :type index: Union[List[int], np.ndarray, pd.Series]
    :return: array with the given index applied
    :rtype: np.ndarray
    """
    if isinstance(index, pd.Series) or isinstance(index, np.ndarray):
        if index.dtype != np.int32:
            index = index.astype(np.int32)
    return arr[index]

def get_resource_package():
    return 'matchain.res'

def get_full_resource_name(resource_name: str) -> str:
    """Resource files are located in the matchain.res package.
    This method returns the full name of a resource of the form
    matchain.res.resource_name.

    :param resource_name: name of the resource
    :type resource_name: str
    :return: the full name: matchain.res.resource_name
    :rtype: str
    """
    return f'{get_resource_package()}.{resource_name}'

def get_resource_name_commands() -> str:
    return get_full_resource_name('mccommands_sdt_shg.yaml')

def get_resource_name_logging() -> str:
    return get_full_resource_name('logging_info.yaml')

def read_resource(resource_name: str) -> bytes:
    """Reads the given resource from the matchain.res package.
    The resource name can be given either as the full name
    matchain.res.resource_name or as resource_name only.

    :param resource_name: name of the resource
    :type resource_name: str
    :return: the stream of the resource
    :rtype: bytes
    """
    name = resource_name
    respack = get_resource_package()
    if resource_name.startswith(f'{respack}.'):
        # i.e. the full resource name was given
        name = resource_name[len(respack) + 1:]
    stream = importlib.resources.read_binary(respack, name)
    return stream

def read_yaml_from_file_or_resource(file_path: str) -> dict:
    if file_path.startswith(get_resource_package()):
        stream = read_resource(file_path)
        return yaml.safe_load(stream)

    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
