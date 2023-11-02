"""This module contains utility functions for reading and processing config files.
"""
import copy
import datetime
import logging
from typing import Any, Callable, List, Union

import matchain.util


def _apply_to_object(function: Callable[[Any], Any], obj: Any) -> Any:
    try:
        return function(obj)
    except (TypeError, AttributeError):
        return obj


def _apply(function: Callable[[Any], Any], objects: Any) -> Any:
    if isinstance(objects, str):
        return _apply_to_object(function, objects)
    try:
        iter(objects)
    except TypeError:
        # objects is not iterable
        return _apply_to_object(function, objects)
    if isinstance(objects, dict):
        return {key: _apply(function, obj) for key, obj in objects.items()}
    # objects are list-like
    return [_apply_to_object(function, obj) for obj in objects]


def _replace_variables(variables: dict,
                       objects: Any,
                       ignore_key_error: bool = False) -> dict:
    """Replace variable keys in all string-valued objects by corresponding
    variable values.

    :param variables: variable name-value pairs
    :type variables: dict
    :param objects: objects that maybe strings containing variable names
    :type objects: Any
    :param ignore_key_error: whether to ignore key errors, defaults to False
    :type ignore_key_error: bool, optional
    :return: objects with replaced variable names
    :rtype: dict
    """

    def fmap(string: str) -> str:
        if ignore_key_error:
            try:
                return string.format_map(variables)
            except KeyError:
                return string
        else:
            return string.format_map(variables)

    return _apply(fmap, objects)


def _collect_variables(variables: dict, dictionary: dict):
    """Collects all variables in the given dictionary and stores them in the
    given variables dictionary."""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            _collect_variables(variables, value)
        elif isinstance(value, (int, float)):
            variables[key] = value
        elif isinstance(value, str):
            if '{' not in value:
                variables[key] = value


def _get_current_time():
    """Returns the string representation of the current time."""
    now = datetime.datetime.now()
    return now.strftime("%y%m%d_%H%M%S")


def _resolve_variables(dictionary: dict, current_time: str) -> dict:
    """Replaces all variables in the given dictionary by their values.

    :param dictionary: The dictionary in which variables should be replaced.
    :type dictionary: dict
    :param current_time: The string representation of the current time.
    :type current_time: str
    :return: The dictionary with replaced variables.
    :rtype: dict
    """
    replaced = dictionary
    for _ in range(5):
        variables = {'current_time': current_time}
        _collect_variables(variables, replaced)
        replaced = _replace_variables(variables,
                                      dictionary,
                                      ignore_key_error=True)
    return replaced

def read_yaml(file_path: str) -> dict:
    """Reads a yaml file and returns the corresponding dictionary.

    :param file_path: path to the yaml file
    :type file_path: str
    :return: the dictionary corresponding to the yaml file
    :rtype: dict
    """
    result = matchain.util.read_yaml_from_file_or_resource(file_path)
    included_paths = result.get('include')
    if included_paths:
        for path in included_paths:
            included = matchain.util.read_yaml_from_file_or_resource(path)
            result.update(included)

    return result

def resolve_config(config: dict) -> dict:
    """The original configuration may contain string values which partly consists
    of variables in the form of {variable_name}. This methods collects the values
    of all variables and replaces {variable_name} iteratively by the corresponding
    variable values.

    example:
        tag: test_{current_time}
        dir_experiments: "./experiments/{tag}"

    The above example uses the special variable 'current_time' and replaces {current_time}
    by a string representation of the current time. The resulting values are similar to:

        tag: test_230101_091500
        dir_experiments: "./experiments/test_230101_091500"

    :param config: The original configuration
    :type config: dict
    :return: The configuration with resolved variables
    :rtype: dict
    """
    current_time = _get_current_time()
    return _resolve_variables(config, current_time)


def split_config(config: dict) -> List[dict]:
    """A configuration must contain the key 'select_datasets' which specifies
    a list of datasets to be used for matching. If the list contains more than one dataset,
    this method splits the configuration into configurations for each dataset,
    and returns a list of such configurations.

    :param config: The original configuration
    :type config: dict
    :raises RuntimeError: if 'select_datasets' is not specified or empty
    :raises RuntimeError: if the name of a dataset in 'select_datasets' is not defined
    :return: A list of configurations, one for each dataset
    :rtype: List[dict]
    """
    # split config into selected datasets and remaining config
    select_datasets = config.get('select_datasets')
    datasets = {}
    remaining = {}
    for key, value in config.items():
        # discard "command" items
        if key in ['select_datasets']:
            continue
        # check whether key defines a dataset
        if isinstance(value, dict):
            mytype = value.get('type')
            if mytype == 'dataset':
                if (not select_datasets) or (key in select_datasets):
                    datasets[key] = value
                continue
        remaining[key] = value

    if not datasets:
        raise RuntimeError('no datasets are selected')
    logging.info('selected datasets=%s', list(datasets.keys()))

    # combine each selected dataset with remaining
    configs = []
    if select_datasets:
        # keep the order of datasets as in select_datasets
        dataset_names = select_datasets
    else:
        # use the order of definition in config files
        dataset_names = datasets.keys()

    current_time = _get_current_time()

    for name in dataset_names:
        try:
            dataset = datasets[name]
        except KeyError as exc:
            raise RuntimeError(f'no dataset configured for {name=}') from exc
        dataset.pop('type')
        dataset['dataset_name'] = name
        new_config = copy.deepcopy(remaining)
        new_config['dataset'] = dataset
        new_config = _resolve_variables(new_config, current_time)
        configs.append(new_config)

    return configs


def set_default_values(config: Union[dict, str]):
    """This methods adds default values for parameter that are missing
    in the given configuration, e.g. if embeddings_device is not configured
    it sets the value 'cuda' if a GPU is available otherwise 'cpu'.

    :param config: The configuration
    :type config: Union[dict, str]
    :return: The configuration with added default values
    :rtype: _type_
    """
    #TODO-AE 231101 make default values completely consistent with matchain.res.mccommands.yaml
    # which e.g. uses sparse vectors instead of token-blocking.
    # Define all default values only in the yaml file
    if isinstance(config, str):
        config = read_yaml(config)
        config = resolve_config(config)

    if config.get('blocking'):
        if not config['blocking'].get('name'):
            config['blocking']['name'] = 'token'
        if not config['blocking'].get('njobs'):
            config['blocking']['njobs'] = 1

    if config.get('similarity'):
        device = config['similarity'].get('embedding_device')
        if not device:
            cuda_available = matchain.util.cuda_available()
            device = 'cuda' if cuda_available else 'cpu'
            config['similarity']['embedding_device'] = device

    if config.get('predict'):
        if not config['predict'].get('ntop'):
            config['predict']['ntop'] = 30

    if config.get('autocal'):
        if not config['autocal'].get('threshold'):
            config['autocal']['threshold'] = 'estimated'

    return config


def get_config(dataset_name: str, file_chains: str) -> dict:
    """Returns the configuration for the specified dataset as dictionary.
    If file_chains is not specified, the default file is used.

    :param dataset_name: The name of the dataset
    :type dataset_name: str
    :param file_chains: The path to the config file
    :type file_chains: str
    :return: The configuration for the specified dataset
    :rtype: dict
    """
    config = read_yaml(file_chains)
    config['select_datasets'] = [dataset_name]
    config = split_config(config)[0]
    config = set_default_values(config)
    return config
