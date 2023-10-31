"""API for matchain package. This module contains the MatChain class which is the
main entry point for the matchain package. The MatChain class provides an alternative
to the purely configuration file based execution of matching algorithms provided by
the matchain package. Each step (command) in the matching process is represented by a
method of the MatChain class. Each method allows to set the parameters of the command
which are then used instead of the values in the configuration file.
The MatChain class also allows to retrieve and manipulate intermediate results
of the matching process.
"""
import collections
from typing import Optional, Union, cast

import pandas as pd

import matchain.base
import matchain.chain
import matchain.config
import matchain.mtoken
import matchain.prepare
import matchain.similarity
import matchain.util


class MatChain:
    """This class provides methods to configure and execute the matching process.
    It also allows to dynamically adapt the configuration file based execution
    of the matching process.

    Most parameters of MatChain's methods are optional and set to None as default.
    If a parameter is set to None, the value of the parameter in the configuration file is used.
    If the parameter is set to any other value, the value in the configuration file is overwritten.
    """

    def __init__(self,
                 data_1: Optional[Union[str, pd.DataFrame]] = None,
                 data_2: Optional[Union[str, pd.DataFrame]] = None,
                 seed: Optional[int] = None,
                 config: Optional[dict] = None) -> None:
        """Initializes the class.

        :param data_1: The first dataset, defaults to None
        :type data_1: Optional[Union[str, pd.DataFrame]], optional
        :param data_2: The first dataset, defaults to None
        :type data_2: Optional[Union[str, pd.DataFrame]], optional
        :param seed: random seed, defaults to None
        :type seed: Optional[int], optional
        :param config: configuration dictionary, defaults to None
        :type config: Optional[dict], optional
        """

        matchain.util.init_console_logging_only()
        self.board = matchain.base.PinBoard()
        if not config:
            config = matchain.config.get_config('test')
            if config.get('chain'):
                config.pop('chain')
        self.board.config = config

        if isinstance(data_1, str):
            config['dataset']['data_1'] = data_1
        else:
            self.board.df_1 = data_1
        if isinstance(data_2, str):
            config['dataset']['data_2'] = data_2
        else:
            self.board.df_2 = data_2

        if seed:
            config['prepare']['seed'] = seed

        self.executed_commands = []

        self._execute('prepare')

    def _execute(self, commands: Union[str, list]) -> None:
        """Executes each command in the list of commands
        but only if the command has not been executed before.

        :param commands: a single command or a list of commands
        :type commands: Union[str, list]
        """
        if isinstance(commands, str):
            commands = [commands]
        for command in commands:
            if command in self.executed_commands:
                continue
            self.executed_commands.append(command)
            matchain.chain.execute_command(command, self.board)

    def _set_blocking_props(self, blocking_props: Optional[list]) -> None:
        """Sets the blocking properties.

        :param blocking_props: the blocking properties which must be a subset of the properties
            configured for the matching process,
            defaults to None
        :type blocking_props: Optional[list]
        :raises ValueError: if no blocking properties are defined
        """
        if blocking_props:
            self.board.config['dataset']['blocking_props'] = blocking_props
        else:
            bprops = self.board.config['dataset']['blocking_props']
            if not bprops:
                raise ValueError(
                    'No blocking properties found. Use either the parameter blocking_props or '
                    + 'the configuration file to set the blocking properties.')

    def _set_embedding_props(self,
                             embedding_batch_size: Optional[int] = None,
                             embedding_model: Optional[str] = None,
                             embedding_device: Optional[str] = None) -> None:
        """Set the properties used by sentence transformer for computing embeddings."""
        config = self.board.config['similarity']
        if embedding_batch_size:
            config['embedding_batch_size'] = embedding_batch_size
        if embedding_model:
            config['embedding_model'] = embedding_model
        if embedding_device:
            config['embedding_device'] = embedding_device

    def property(self,
                 property_name: str,
                 simfct: Optional[str] = None,
                 short_name: Optional[str] = None,
                 property_type: Optional[str] = None,
                 ndigits: Optional[int] = None) -> None:
        """Adds a property (column) which is relevant for the matching process.
        The property name must be a column name in the dataframes provided.

        :param property_name: the name of the property
        :type property_name: str
        :param simfct: the similarity function to be used for the property,
            see similarity.get_similarity_function_names() for allowed values,
            defaults to None
        :type simfct: Optional[str], optional
        :param short_name: a short name for the property, defaults to None
        :type short_name: Optional[str], optional
        :param property_type: the type of the property, defaults to None
        :type property_type: Optional[str], optional
        :param ndigits: the number of decimal places for properties of type float, defaults to None
        :type ndigits: Optional[int], optional
        :raises ValueError: unknown similarity function
        """

        config = self.board.config

        if simfct:
            allowed = matchain.similarity.get_similarity_function_names()
            if not simfct in allowed:
                raise ValueError(
                    f'unknown simfct={simfct}, only values={allowed} are allowed'
                )
            if simfct == 'tfidf':
                config['mtoken']['tfidf_index'] = True
            if not config['dataset'].get('props_sim'):
                config['dataset']['props_sim'] = collections.defaultdict(list)
            props_sim = config['dataset']['props_sim']
            props_sim[property_name].append(simfct)

        else:
            if not config['dataset'].get('props'):
                config['dataset']['props'] = {}
            if not short_name:
                short_name = property_name
            if property_type:
                description = [short_name, property_type, ndigits]
            else:
                description = short_name
            props = self.board.config['dataset']['props']
            props[property_name] = description

    def token(
            self,
            blocking_props: Optional[list] = None,
            maximum_token_frequency: Optional[int] = None,
            minimum_token_length: Optional[int] = None,
            readable: Optional[bool] = None,
            tfidf_index: Optional[bool] = None
    ) -> matchain.mtoken.TokenManager:
        """Creates the token index for the values of all blocking properties.
        The token index can be used for blocking or calculating similarity scores.

        :param blocking_props: the blocking properties which must be a subset of the properties,
            defaults to None
        :type blocking_props: Optional[list], optional
        :param maximum_token_frequency: tokens with a frequency higher than this value are ignored,
            defaults to None
        :type maximum_token_frequency: Optional[int], optional
        :param minimum_token_length: tokens with a length smaller than this value are ignored,
            defaults to None
        :type minimum_token_length: Optional[int], optional
        :param readable: whether tokens and instances are represented as human-readable strings
            or as integers, useful for debugging, defaults to None
        :type readable: Optional[bool], optional
        :param tfidf_index: if True, the token index is also created for those
            non-blocking properties using similarity function 'tdidf', defaults to None
        :type tfidf_index: Optional[bool], optional
        :return: the token manager containing the token index
        :rtype: matchain.mtoken.TokenManager
        """

        config = self.board.config['mtoken']
        self._set_blocking_props(blocking_props)

        if maximum_token_frequency:
            config['maximum_token_frequency'] = maximum_token_frequency
        if minimum_token_length:
            config['minimum_token_length'] = minimum_token_length
        if readable is not None:
            config['readable'] = readable
        if tfidf_index is not None:
            config['tfidf_index'] = tfidf_index

        self._execute('mtoken')
        return self.board.token_manager

    def blocking(self,
                 name: str = 'sparsedottopn',
                 blocking_props: Optional[list] = None,
                 vector_type: str = 'shingle_tfidf',
                 shingle_size: int = 3,
                 query_strategy: str = 'smaller',
                 njobs: int = 4,
                 ntop: int = 10,
                 blocking_threshold: float = 0.5,
                 embedding_batch_size: Optional[int] = None,
                 embedding_model: Optional[str] = None,
                 embedding_device: Optional[str] = None) -> pd.MultiIndex:
        """Creates a candidate set of matching pairs by blocking.
        Some parameters are only relevant for certain blocking methods and vector types.

        :param name: the name of the blocking method,
            either 'token', 'bruteforce', 'sklearn', 'sparsedottopn', 'nmslib' or 'faiss',
            defaults to 'token'
        :type name: str, optional
        :param blocking_props: the blocking properties, defaults to None
        :type blocking_props: Optional[list], optional
        :param vector_type: the type of the vector used for representing the blocking values,
            not relevant for token blocking,
            defaults to 'shingle_tfidf'
        :type vector_type: str, optional
        :param shingle_size: the size of the shingles used for vector type 'shingle_tfidf',
            defaults to 3
        :type shingle_size: int, optional
        :param query_strategy: the query strategy:
            'first' means use vectors from the first dataset as index vectors and the other ones
            as query vectors,
            'second' means use vectors from the second dataset as index vectors,
            'smaller' means use the smaller set of vectors as index vectors,
            'larger' means use the larger set of vectors as index vectors,
            , defaults to 'smaller'
        :type query_strategy: str, optional
        :param njobs: number of parallel jobs, only relevant for 'sklearn' and 'nmslib',
            defaults to 4
        :type njobs: int, optional
        :param ntop: number of top candidates to be returned, not relevant for token blocking,
            defaults to 10
        :type ntop: int, optional
        :param blocking_threshold: filter top candidates beyond this threshold, not relevant for
            token blocking, defaults to 0.5
        :type blocking_threshold: float, optional
        :param embedding_batch_size: the batch size used for computing embeddings,
            only relevant for embedding vector type, defaults to None
        :type embedding_batch_size: Optional[int], optional
        :param embedding_model: the name of the language model used by sentence transformer,
            see https://www.sbert.net/docs/pretrained_models.html for allowed values,
            only relevant for embedding vectory type, defaults to None
        :type embedding_model: Optional[str], optional
        :param embedding_device: 'cuda', 'cpu' or None, None means that 'cuda' is used if a GPU
            is available and 'cpu' otherwise, only relevant for embedding vector type,
            defaults to None
        :type embedding_device: Optional[str], optional
        :return: the candidate pairs
        :rtype: pd.MultiIndex
        """

        self._set_blocking_props(blocking_props)
        config = self.board.config['blocking']
        config['name'] = name
        config['vector_type'] = vector_type
        config['shingle_size'] = shingle_size
        config['query_strategy'] = query_strategy
        config['njobs'] = njobs
        config['ntop'] = ntop
        config['blocking_threshold'] = blocking_threshold
        self._set_embedding_props(embedding_batch_size, embedding_model,
                                  embedding_device)

        config_mtoken = self.board.config.get('mtoken')
        if name == 'token' or (config_mtoken and config_mtoken.get('tfidf_index')):
            self.token()

        self._execute('blocking')
        return cast(pd.MultiIndex, self.board.candidate_pairs)

    def similarity(self,
                   embedding_batch_size: Optional[int] = None,
                   embedding_model: Optional[str] = None,
                   embedding_device: Optional[str] = None,
                   tfidf_maxidf: Optional[int] = None) -> pd.DataFrame:
        """Computes the similarity scores for the candidate pairs created by blocking.

        :param embedding_batch_size: the batch size used for computing embeddings,
            only relevant for similarity function 'embedding', defaults to None
        :type embedding_batch_size: Optional[int], optional
        :param embedding_model: the name of the language model used by sentence transformer,
            see https://www.sbert.net/docs/pretrained_models.html for allowed values,
            only relevant for similarity function 'embedding', defaults to None
        :type embedding_model: Optional[str], optional
        :param embedding_device: 'cuda', 'cpu' or None, None means that 'cuda' is used if a GPU
            is available and 'cpu' otherwise, only relevant for similarity function 'embedding',
            defaults to None
        :type embedding_device: Optional[str], optional
        :param tfidf_maxidf: upper bound for the IDF value, only relevant for similarity function
            'tfidf', defaults to None
        :type tfidf_maxidf: Optional[int], optional
        :return: the similarity scores for all candidate pairs
        :rtype: pd.DataFrame
        """

        self._set_embedding_props(embedding_batch_size, embedding_model,
                                  embedding_device)
        if tfidf_maxidf:
            config = self.board.config['similarity']
            config['tfidf_maxidf'] = tfidf_maxidf

        if 'blocking' not in self.executed_commands:
            self.blocking(name='token')
        self._execute('similarity')
        return self.board.df_sim

    def evaluate(self,
                 matches: Optional[Union[str, pd.MultiIndex]] = None,
                 compute_max_f1: bool = False) -> dict:
        """Computes the evaluation metrics for the predicted matches.

        :param matches: a file or multi-index containing the true matches, defaults to None
        :type matches: Optional[Union[str, pd.MultiIndex]], optional
        :param compute_max_f1: if True the maximum F1-score is computed additionally,
            defaults to False
        :type compute_max_f1: bool, optional
        :return: the evaluation metrics
        :rtype: dict
        """
        if matches is not None:
            if isinstance(matches, str):
                self.board.config['dataset']['file_matches'] = matches
            else:
                self.board.true_matches = matches
        if compute_max_f1 is not None:
            self.board.config['evaluate']['compute_max_f1'] = compute_max_f1

        if self._predict_required():
            self.predict()

        self._execute('evaluate')
        return self.board.evaluation_metrics

    def _predict_required(self) -> bool:
        """Checks whether calling the predict command is required.

        :return: True if calling the predict command is required
        :rtype: bool
        """
        for command in ['autocal', 'weighted', 'predict']:
            if command in self.executed_commands:
                return False
        return True

    def predict(self, ntop: Optional[int] = None) -> pd.MultiIndex:
        """The predict command is required if the choosen matching algorithm
        computes vector representations for each entity as a whole. In this case, the predict
        command computes for each entity the most similar entities from the other dataset.

        :param ntop: number of most similar entities to be computed for each entity,
            used for executing the predict command, defaults to None
        :type ntop: Optional[int], optional
        :return: the predicted matches
        :rtype: pd.MultiIndex
        """
        if not self._predict_required():
            return self.board.predicted_matches

        config = self.board.config['predict']
        if ntop is not None:
            config['ntop'] = ntop

        if 'node2vec' in self.executed_commands:
            if not ('word2vec' in self.executed_commands
                    and 'w2vpytorch' in self.executed_commands):
                self.word2vec()

        self._execute('predict')
        return self.board.predicted_matches

    def autocal(self,
                delta: Optional[float] = None,
                threshold_method: Optional[str] = None) -> None:
        """Start matching with the autocal algorithm.

        :param delta: bin size, defaults to None
        :type delta: Optional[float], optional
        :param threshold_method: 'estimated' or 'majority', defaults to None
        :type threshold_method: Optional[str], optional
        """
        if delta:
            self.board.config['autocal']['delta'] = delta
        if threshold_method:
            self.board.config['autocal']['threshold'] = threshold_method

        self.similarity()
        self._execute('autocal')

    def weighted(self, threshold: Optional[float] = None) -> None:
        if threshold:
            self.board.config['weighted']['threshold'] = threshold

        self.similarity()
        self._execute('weighted')

    def embdi(self, matches: str) -> None:
        # parameter matches is required since the original embdi code
        # uses it for internal evaluation
        if matches is not None:
            if isinstance(matches, str):
                self.board.config['dataset']['file_matches'] = matches

        self.token()
        self._execute('embdi')

    def node2vec(self,
                 p: Optional[int] = None,
                 q: Optional[int] = None,
                 walk_length: Optional[int] = None,
                 walk_number: Optional[int] = None,
                 walk_file: Optional[str] = None) -> None:

        config = self.board.config['node2vec']
        if p is not None:
            config['p'] = p
        if q is not None:
            config['q'] = q
        if walk_length is not None:
            config['walk_length'] = walk_length
        if walk_number is not None:
            config['walk_number'] = walk_number
        if walk_file is not None:
            config['walk_file'] = walk_file

        self.graph()
        self._execute('node2vec')

    def graph(self, file_graph: Optional[str] = None) -> None:
        if file_graph:
            self.board.config['graph']['file_graph'] = file_graph

        self.token()
        self._execute('graph')

    def word2vec(self,
                 epochs: Optional[int] = None,
                 min_count: Optional[int] = None,
                 negative: Optional[int] = None,
                 sample: Optional[float] = None,
                 sg: Optional[int] = None,
                 vector_size: Optional[int] = None,
                 window: Optional[int] = None,
                 workers: Optional[int] = None) -> None:

        config = self.board.config['word2vec']
        if epochs is not None:
            config['epochs'] = epochs
        if min_count is not None:
            config['min_count'] = min_count
        if negative is not None:
            config['negative'] = negative
        if sample is not None:
            config['sample'] = sample
        if sg is not None:
            config['sg'] = sg
        if vector_size is not None:
            config['vector_size'] = vector_size
        if window is not None:
            config['window'] = window
        if workers is not None:
            config['workers'] = workers

        self._execute('word2vec')
