"""This module provides functionality for blocking. Blocking is performed to reduce the number of
candidate pairs to a manageable size before applying a matching algorithm. The idea is to
partition the datasets into blocks and to only compare records within the same block.
This module provides several blocking approaches:

- Token blocking: Strings are split into tokens and two records become a candidate pair
if their values share at least one token.

- Sparse vectors: String values are represented as (weighted) sparse vectors and are compared
based on the cosine or Jaccard similarity of the vectors.

- Dense vectors: String values are represented as dense vectors (embeddings) and are compared
based on the cosine similarity of the vectors.

In case of sparse and dense vectors, nearest neighbour search is used to find the closest vectors
efficiently. The corresponding records become candidate pairs. Exact nearest neighbour search is
implemented by brute force which is slow but works well for small datasets.
There are very efficient libraries for approximate nearest neighbour search either for sparse
or dense vectors. This module wraps some search algorithms of some of these libraries
(sklearn, faiss, nmslib).
"""

import itertools
import logging
import time
from typing import Callable, List, Optional, Set, Tuple, Union, cast

import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.feature_extraction.text
import sklearn.neighbors
from tqdm import tqdm

import matchain.similarity
import matchain.util

# check which libraries for nearest neighbour search are installed
MODULE_NOT_FOUND_DICT_FOR_BLOCKING = set()

try:
    import faiss
except ModuleNotFoundError:
    MODULE_NOT_FOUND_DICT_FOR_BLOCKING.add('faiss')

try:
    import nmslib
except ModuleNotFoundError:
    MODULE_NOT_FOUND_DICT_FOR_BLOCKING.add('nmslib')

try:
    import sparse_dot_topn
except ModuleNotFoundError:
    MODULE_NOT_FOUND_DICT_FOR_BLOCKING.add('sparsedottopn')


def check_blocking(config: dict) -> None:
    """Checks whether necessary libraries for the configured blocking methods
    are installed.

    :param config: The configuration dictionary
    :type config: dict
    :raises ValueError: If a blocking method is configured but the corresponding library
        is not installed.
    """
    blocking_name = config['blocking']['name']
    for name in ['faiss', 'nmslib', 'sparsedottopn']:
        if blocking_name == name and name in MODULE_NOT_FOUND_DICT_FOR_BLOCKING:
            message = '{annlib} is configured for blocking but not found. ' + \
                'Blocking by nearest neighbour search with {annlib} is not possible.'
            raise ValueError(message.format(annlib=name))


class TokenBlocking:
    """Token blocking is a simple and fast blocking method
    that is based on the assumption that two records
    are similar if they share at least one token. The given token index contains for
    each token the set of record ids that contain the token. The candidate pairs
    are the cross product of the sets of record ids for each token.
    """

    @staticmethod
    def get_candidate_pairs(df_index: pd.DataFrame) -> pd.MultiIndex:
        """Uses the token index to generate the candidate pairs according to token blocking.

        :param df_index: The token index.
        :type df_index: pd.DataFrame
        :return: Candidate pairs sorted by the row index of the first dataset.
        :rtype: pd.MultiIndex
        """
        candidates = set()
        index = df_index.to_dict(orient='index')
        for _, row in tqdm(index.items()):
            links_1 = row['links_1']
            links_2 = row['links_2']
            if isinstance(links_1, set) and isinstance(links_2, set):
                new_pairs = itertools.product(links_1, links_2)
                candidates.update(new_pairs)

        cand_index = pd.MultiIndex.from_tuples(candidates,
                                               names=['idx_1', 'idx_2'])
        cand_index = matchain.util.sort_pairs(cand_index)
        return cand_index


def _read_candidate_pairs(file: str, offset: int) -> pd.MultiIndex:
    """Reads the candidate pairs from the given file and returns them as a set
    sorted according to the row index of the first dataset.

    :param file: The file to read the candidate pairs from.
    :type file: str
    :param offset: The offset added to the row index of the second dataset.
    :type offset: int
    :return: A set of sorted candidate pairs.
    :rtype: pd.MultiIndex
    """
    if file.endswith('test.csv'):
        df_all = matchain.util.read_and_concat_csv_table_ditto_format(
            file, offset, apply_format=False)
    else:
        df_all = matchain.util.read_csv(file, offset, apply_format=False)

    cand_index = matchain.util.sort_pairs(df_all)
    return cand_index


class ShingleWrapper:
    """Creates vectors of shingles with TF-IDF weights.
    """

    def __init__(self, shingle_size: int):
        """Init the ShingleWrapper

        :param shingle_size: size of the shingles
        :type shingle_size: int
        """
        self.shingle_size = shingle_size

    def create_shingles(self, s: str) -> Set[str]:
        """Creates the shingles for the given string.
        Example: The shingles of size 3 for string 'matching' are
        {'mat', 'atc', 'tch', 'chi', 'hin', 'ing'}.

        :param s: The string to create the shingles for.
        :type s: str
        :return: The set of shingles.
        :rtype: Set[str]
        """
        k = self.shingle_size
        return set(s[i:i + k] for i in range(len(s) - k + 1))

    def generate_vectors(self, values: pd.Series) -> scipy.sparse.csr_matrix:
        """Creates shingles from the values
        and computes the TF-IDF weights of the shingles.

        :param values: series containing the data
        :type values: pd.Series
        :return: sparse matrix with TF-IDF weights
        :rtype: scipy.sparse.csr_matrix
        """

        start_time = time.time()
        analyzer = self.create_shingles
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
            min_df=1, analyzer=analyzer)
        tf_idf_matrix = vectorizer.fit_transform(values)
        logging.debug('tf_idf_matrix=%s, time=%s', tf_idf_matrix.shape, time.time() - start_time)

        return cast(scipy.sparse.csr_matrix, tf_idf_matrix)


class NNWrapperBase():
    """Base class to wrap nearest neighbour search implemented by different libraries.
    """

    def nearest_neighbours(self, index_vectors: np.ndarray,
                           query_vectors: np.ndarray,
                           ntop: int) -> pd.DataFrame:
        """Find for each query vector the ntop nearest neighbours in the index vectors.

        :param index_vectors: The rows represent the vectors that are stored in the index.
        :type index_vectors: np.ndarray
        :param query_vectors: Each row represents a vector for which the nearest neighbours
            are searched.
        :type query_vectors: np.ndarray
        :param ntop: The number of nearest neighbours for each query vector to return.
        :type ntop: int
        :raises NotImplementedError: This method must be implemented by the subclasses.
        :return: A dataframe with columns
            'id' (row number of index vector),
            'query_id' (row number of query vector), and
            'score' (similarity score between index and query vector).
        :rtype: pd.DataFrame
        """
        raise NotImplementedError()

    @staticmethod
    def _create_dataframe_for_search_result(
            search_index: np.ndarray,
            search_score: np.ndarray) -> pd.DataFrame:
        """The n-th row in the given search index and search score represents the search result
        for the n-th query vector. The method combines both arrays and creates a dataframe
        with columns 'id', 'query_id' and 'score'.

        :param search_index: The entries in each row are the indices of the ntop nearest neighbours.
        :type search_index: np.ndarray
        :param search_score: The entries in each row are the similarity scores of the ntop
            nearest neighbours.
        :type search_score: np.ndarray
        :return: The combined dataframe.
        :rtype: pd.DataFrame
        """
        n_query_vectors, ntop = search_index.shape
        vec_i = np.ravel(search_index)
        vec_s = np.ravel(search_score)
        vec_query_ids = np.empty(len(vec_i), dtype=int)
        for j in range(n_query_vectors):
            vec_query_ids[j * ntop:(j + 1) * ntop] = j
        dfr = pd.DataFrame({
            'id': vec_i,
            'query_id': vec_query_ids,
            'score': vec_s
        })
        # skip entries with id == -1 (if less than ntop nearest neighbours were found)
        dfr = dfr[dfr['id'] >= 0]
        return dfr


class NNWrapperSparseDotTopn(NNWrapperBase):
    """A wrapper for library sparse_dot_topn to find approximate nearest neighbours.
    """

    def __init__(self, threshold: float):
        """ntop results are skipped if the cosine similarity is smaller than the given threshold.

        :param threshold: The threshold for the cosine similarity.
        :type threshold: float
        """
        self.threshold = threshold

    @staticmethod
    def _awesome_cossim_top(A: scipy.sparse.csr_matrix,
                            B: scipy.sparse.csr_matrix,
                            ntop: int,
                            lower_bound: float = 0) -> scipy.sparse.csr_matrix:
        """This method is used to compute the cosine similarity between certain pairs of row
        vectors of the sparse matrix A efficiently. For each row vector, it only considers at
        most ntop other row vectors that have a cosine similarity greater than the given lower
        bound. The returned sparse matrix is quadratic and its number of rows and columns equals
        the number of rows of A. Each row of the matrix contains the ntop highest cosine
        similarities.

        The code was copied from
        https://bergvca.github.io/2017/10/14/super-fast-string-matching.html .
        See https://github.com/Bergvca/string_grouper for more information.

        :param A: The sparse matrix of index vectors.
        :type A: scipy.sparse.csr_matrix
        :param B: The sparse matrix of query vectors.
        :type B: scipy.sparse.csr_matrix
        :param ntop: The number of pairs with the highest cosine similarity to return.
        :type ntop: int
        :param lower_bound: The lower bound for the cosine similarity, defaults to 0.
        :type lower_bound: float, optional
        :return:  A sparse matrix containing the cosine similarity.
        :rtype: scipy.sparse.csr_matrix
        """
        # force A and B as a CSR matrix.
        # If they have already been CSR, there is no overhead
        A = A.tocsr()
        B = B.transpose().tocsr()
        M, _ = A.shape
        _, N = B.shape

        idx_dtype = np.int32

        nnz_max = M * ntop

        indptr = np.zeros(M + 1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)

        sparse_dot_topn.sparse_dot_topn.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype), A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype), B.data, ntop, lower_bound,
            indptr, indices, data)

        return scipy.sparse.csr_matrix((data, indices, indptr), shape=(M, N))

    def nearest_neighbours(self, index_vectors: np.ndarray,
                           query_vectors: np.ndarray,
                           ntop: int) -> pd.DataFrame:
        """see NNWrapperBase.nearest_neighbours
        """
        all_pairs = NNWrapperSparseDotTopn._awesome_cossim_top(
            index_vectors, query_vectors, ntop, self.threshold)

        nonzeros = all_pairs.nonzero()
        scores = all_pairs[nonzeros]
        #1-dim matrix as ndarray
        scores = scores.A1

        df_scores = pd.DataFrame({
            'id': nonzeros[0],
            'query_id': nonzeros[1],
            'score': scores
        })

        return df_scores


class NNWrapperNmslib(NNWrapperBase):
    """A wrapper for the nearest neighbour search library nmslib. Here, the library is used
    to find nearest neighbours based on the Jaccard similarity of sparse vectors (which arise
    from shingling and TF-IDF weighting). The code follows the examples in the notebooks
    search_sparse_cosine.ipynb and search_generic_sparse_jaccard.ipynb
    (see https://github.com/nmslib/nmslib/blob/master/python_bindings/notebooks )
    """

    def __init__(self, threshold: float, njobs: int = 1):
        """ntop results are skipped if the cosine similarity is smaller than the given threshold.

        :param threshold: The threshold for the Jaccard similarity.
        :type threshold: float
        :param njobs: number of jobs for search, defaults to 1
        :type njobs: int, optional
        """
        self.threshold = threshold
        self.similarity = 'cosine'
        self.n_threads = njobs
        self.index_time_params = {
            'M': 30,
            'indexThreadQty': self.n_threads,
            'efConstruction': 100,
            'post': 0
        }
        self.query_time_params = {'efSearch': 100}

    def nearest_neighbours(self, index_vectors: np.ndarray,
                           query_vectors: np.ndarray,
                           ntop: int) -> pd.DataFrame:
        """see NNWrapperBase.nearest_neighbours
        """
        index_vec: Union[np.ndarray, List[str]]
        if self.similarity == 'cosine':
            space = 'cosinesimil_sparse'
            data_type = nmslib.DataType.SPARSE_VECTOR
            index_vec = index_vectors
            query_vec = query_vectors
        else:
            space = 'jaccard_sparse'
            data_type = nmslib.DataType.OBJECT_AS_STRING
            # for Jaccard similarity, vectors must be passed as strings
            index_vec = NNWrapperNmslib._matr_to_str_array(index_vectors)
            query_vec = NNWrapperNmslib._matr_to_str_array(query_vectors)

        index = nmslib.init(method='hnsw', space=space, data_type=data_type)
        index.addDataPointBatch(index_vec)
        index.createIndex(self.index_time_params)

        index.setQueryTimeParams(self.query_time_params)
        # knnQueryBatch returns a list of tuples (np.ndarray, np.ndarray).
        # The first array of each tuple contains the nearest neighbours' ids.
        # The second array of each tuple contains the nearest neighbours' distances.
        # The arrays are sorted in ascending order of distances.
        # The arrays are of size ntop.
        neighbours = index.knnQueryBatch(query_vec,
                                         k=ntop,
                                         num_threads=self.n_threads)
        logging.debug('neighbours (query answer)=%s', len(neighbours))

        # convert neighbours to an numpy array of shape (# query_vectors, 2, ntop)
        arr = np.array(neighbours)
        search_index = arr[:, 0, :].astype(int)
        search_score = arr[:, 1, :]

        df_scores = NNWrapperBase._create_dataframe_for_search_result(
            search_index, search_score)
        # nmslib was initialized with 'jaccard_sparse' and jaccard distance
        # is returned as score (instead of cosine similarity)
        # thus, filter scores by condition <= 1 - threshold
        if self.threshold > 0:
            df_scores = df_scores[df_scores['score'] <= 1 - self.threshold]

        logging.debug('df_scores (after applying threshold)=%s',
                      len(df_scores))

        return df_scores

    @staticmethod
    def _matr_to_str_array(
            sparse_matrix: scipy.sparse.csr_matrix) -> List[str]:
        """For each row in the sparse matrix, extract a list of IDs and
        concatenate them to a string with space as separator.
        Return a list of such strings.

        :param sparse_matrix: The sparse matrix with vector IDs.
        :type sparse_matrix: scipy.sparse.csr_matrix
        :return: The list of concatenated strings.
        :rtype: List[str]
        """
        # copied from search_generic_sparse_jaccard.ipynb

        res = []
        indptr = sparse_matrix.indptr
        indices = sparse_matrix.indices
        for row in range(sparse_matrix.shape[0]):
            arr = [k for k in indices[indptr[row]:indptr[row + 1]]]
            if arr:
                arr.sort()
                res.append(' '.join([str(k) for k in arr]))
            else:
                # TODO-AE 230710 nmslib with Jaccard similarity and empty shingle vectors
                # if the original value is none or empty, the resulting
                # shingle vector is the zero vector. This is just a hack
                # to avoid errors. However, Jaccard does not work well
                # for blocking even if there are no zero shingle vectors.
                # Thus, resolving this hack has no high priority.
                res.append('10000000')
        return res


class NNWrapperSklearn(NNWrapperBase):
    """A wrapper for library sklearn to find nearest neighbours.
    """

    def __init__(self,
                 threshold: float,
                 algorithm: str = 'brute',
                 metric: str = 'cosine',
                 njobs: int = 1):
        """The library sklearn provides several algorithms and distances / similarity scores
        to find nearest neighbours. The algorithm 'brute' is the simplest one and returns
        exact nearest neighbours. It works well with the cosine similarity both for sparse
        and dense input but raises a ValueError for the Jaccard similarity and sparse input.
        The algorithms 'kd_tree' and 'ball_tree' return approximate nearest neighbours.
        Metrics 'l1' or 'l2' work for sparse and dense vectors with all algorithms but lead
        to many false negatives and low F1-scores and thus are not recommended.

        :param threshold: The threshold for distance / similarity scores.
        :type threshold: float
        :param algorithm: 'brute', 'kd_tree' or 'ball_tree', defaults to 'brute'
        :type algorithm: str, optional
        :param metric: 'cosine', 'l1', 'l2', defaults to 'cosine'
        :type metric: str, optional
        :param njobs: number of parallel jobs, defaults to 1
        :type njobs: int, optional
        """
        self.threshold = threshold
        self.algorithm = algorithm
        self.metric = metric
        self.njobs = njobs

    def nearest_neighbours(self, index_vectors: np.ndarray,
                           query_vectors: np.ndarray,
                           ntop: int) -> pd.DataFrame:
        """see NNWrapperBase.nearest_neighbours
        """
        nneighbors = sklearn.neighbors.NearestNeighbors(
            n_neighbors=ntop,
            radius=self.threshold,
            algorithm=self.algorithm,
            metric=self.metric,
            n_jobs=self.njobs)
        nneighbors.fit(index_vectors)

        # kneighbors returns a two lists of numpy arrays.
        # Both lists have the same length as the number of query vectors.
        # Each numpy array is of length ntop.
        # ValueError is raisen if ntop is larger than the number of index vectors.
        # search_score contains the cosine distances (not similarities)
        # search_index the indices of the index vectors (type int64)
        search_score, search_index = nneighbors.kneighbors(
            query_vectors, return_distance=True)

        df_scores = NNWrapperBase._create_dataframe_for_search_result(
            search_index, search_score)
        # use < 1 and < since search_score contains distances (not similarities)
        if self.threshold < 1:
            df_scores = df_scores[df_scores['score'] < self.threshold]

        logging.debug('df_scores (after applying threshold)=%s',
                      len(df_scores))

        return df_scores


class NNBruteForce(NNWrapperBase):
    """A straight forward brute force implementation for exact nearest neighbour search.
    It provides the same functionality as NNWrapperSklearn with parameters
    algorithm='brute' and metric='cosine' but is slower.
    """

    def __init__(self, threshold: float):
        """ntop results are skipped if the cosine similarity is smaller than the given threshold.

        :param threshold: The threshold for the cosine similarity.
        :type threshold: float
        """
        self.threshold = threshold

    def nearest_neighbours(self, index_vectors: np.ndarray,
                           query_vectors: np.ndarray,
                           ntop: int) -> pd.DataFrame:
        """see NNWrapperBase.nearest_neighbours
        """

        start_time = time.time()
        scores = np.dot(query_vectors, index_vectors.T)
        logging.debug("dot product time=%s", time.time() - start_time)
        if isinstance(scores, scipy.sparse.spmatrix):
            scores = scores.toarray()

        asort = np.argsort(scores)
        asort = asort[:, -ntop:]
        # reverse the order (from largest to smallest dot product)
        search_index = asort.T[::-1].T

        search_score = []
        n_query_vectors = query_vectors.shape[0]
        for i in tqdm(range(n_query_vectors), desc='brute force'):
            search_row = search_index[i]
            scores_row = scores[i]
            sorted_scores_row = scores_row[search_row]
            search_score.append(sorted_scores_row)

        search_score = np.vstack(search_score)

        df_scores = NNWrapperBase._create_dataframe_for_search_result(
            search_index, search_score)
        if self.threshold > 0:
            df_scores = df_scores[df_scores['score'] > self.threshold]

        logging.debug('df_scores (after applying threshold)=%s',
                      len(df_scores))

        return df_scores


class NNWrapperFaiss(NNWrapperBase):
    """A wrapper for library faiss to search approximate nearest neighbours
    with respect to cosine similarity.
    """

    def __init__(self, threshold: float):
        """ntop results are skipped if the cosine similarity is smaller than the given threshold.

        :param threshold: The threshold for the cosine similarity.
        :type threshold: float
        """
        self.threshold = threshold
        self.cosine_sim = True

    def nearest_neighbours(self, index_vectors: np.ndarray,
                           query_vectors: np.ndarray,
                           ntop: int) -> pd.DataFrame:
        """see NNWrapperBase.nearest_neighbours
        """
        # init faiss index with index vectors
        vector_dim = index_vectors.shape[1]
        if self.cosine_sim:
            # IndexFlatIP uses the inner product (dot product) which
            # is equivalent to cosine similarity if all vectors are normalized to unit length.
            # Thus, all index and query vectors must be normalized.
            faissindex = faiss.IndexFlatIP(vector_dim)
        else:
            faissindex = faiss.IndexFlatL2(vector_dim)
        faissindex.add(index_vectors)
        logging.debug('faiss index total=%s', faissindex.ntotal)

        # faissindex.search returns two numpy arrays of shape (# query_vectors, ntop).
        # search_score contains the distances or similarity values.
        # search_index refers to the index of the nearest neighbour.
        # If less than ntop nearest neighbours were found, the remaining entries
        # in scores_index are -1
        search_score, search_index = faissindex.search(query_vectors, ntop)

        logging.debug('faiss search index=%s, search score=%s',
                      search_index.shape, search_score.shape)

        df_scores = NNWrapperBase._create_dataframe_for_search_result(
            search_index, search_score)

        if self.threshold > 0:
            if self.cosine_sim:
                df_scores = df_scores[df_scores['score'] > self.threshold]
            else:
                raise NotImplementedError()

        logging.debug('df_scores (after applying threshold)=%s',
                      len(df_scores))

        return df_scores


class NearestNeighbourBlocking():
    """This class provides functionality to find candidate pairs by nearest neighbour search.
    It allows to use different vector representations, similarity measures and nearest
    neighbour search algorithms.
    """

    @staticmethod
    def _create_value_index_array(
            df_data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Iterates over all cells in the given dataframe and creates a series of unique
        cell values. The series is used to create a new dataframe with the same shape
        as the original dataframe. The new dataframe contains the position (index) of
        unique cell values in the series instead of the cell values in the original dataframe.

        :param df_data: data to be indexed
        :type df_data: pd.DataFrame
        :return: the unique cell values and the dataframe with the indices of the values
        :rtype: Tuple[pd.Series, pd.DataFrame]
        """
        value2int = {}
        rows_int = []
        for tup in df_data.itertuples(index=False, name=None):
            row_int = []
            for value in tup:
                if matchain.util.notnull(value):
                    i = value2int.get(value)
                    if i is None:
                        i = len(value2int)
                        value2int[value] = i
                    row_int.append(i)
                else:
                    row_int.append(None)
            rows_int.append(row_int)

        index_array = np.array(rows_int)
        df_index_array = pd.DataFrame(data=index_array)
        values = pd.Series(value2int.keys())
        return values, df_index_array

    @staticmethod
    def _get_candidates_by_property(
            vectors: np.ndarray, value_index_column: pd.Series, size_1: int,
            nearest_neighbours: Callable, ntop: int,
            query_strategy: str) -> Set[Tuple[int, int]]:
        """The rows of the given vectors are the vector representations of the original cell
        values. The integers in value_index_column are the positions of vectors corresponding
        to the original cell values. The method splits the integers into two groups according
        to the given size_1. Depending on the given query_strategy, either the first group is
        used as index vectors and the second group as query vectors or vice versa. The nearest
        neighbours are searched and the resulting candidate pairs are returned.

        :param vectors: The vector representations of the original cell values.
        :type vectors: np.ndarray
        :param value_index_column: The positions of the vectors corresponding to the original
            cell values.
        :type value_index_column: pd.Series
        :param size_1: The size of the first group of cell values.
        :type size_1: int
        :param nearest_neighbours: The nearest neighbour search algorithm.
        :type nearest_neighbours: Callable
        :param ntop: The number of nearest neighbours to return.
        :type ntop: int
        :param query_strategy: The query strategy. Either 'first', 'second', 'smaller' or 'larger'.
        :type query_strategy: str
        :raises ValueError: If the query strategy is unknown.
        :return: The set of candidate pairs.
        :rtype: Set[Tuple[int, int]]
        """

        index_1 = value_index_column[:size_1]
        index_1 = index_1[index_1.notnull()]

        index_2 = value_index_column[size_1:]
        index_2 = index_2[index_2.notnull()]

        reverted_order = False
        if query_strategy == 'first':
            pass  # nothing to do here
        elif query_strategy == 'second':
            reverted_order = True
        elif query_strategy == 'smaller':
            if len(index_1) > len(index_2):
                reverted_order = True
        elif query_strategy == 'larger':
            if len(index_1) < len(index_2):
                reverted_order = True
        else:
            raise ValueError(f'unknown query_strategy={query_strategy}')

        if reverted_order:
            index_1, index_2 = index_2, index_1

        nn_index_vectors = matchain.util.advanced_indexing(vectors, index_1)
        nn_query_vectors = matchain.util.advanced_indexing(vectors, index_2)

        logging.debug('starting nearest neighbours search')
        df_scores = nearest_neighbours(nn_index_vectors, nn_query_vectors,
                                       ntop)
        logging.debug('finished nearest neighbours search')

        ser_id = df_scores['id']
        ser_qid = df_scores['query_id']
        ind_id = index_1.iloc[ser_id].index
        ind_qid = index_2.iloc[ser_qid].index

        if reverted_order:
            df_tmp = pd.DataFrame({'idx_1': ind_qid, 'idx_2': ind_id})
        else:
            df_tmp = pd.DataFrame({'idx_1': ind_id, 'idx_2': ind_qid})

        df_tmp.set_index(['idx_1', 'idx_2'], inplace=True)
        candidates = set(df_tmp.index.to_list())

        return candidates

    @staticmethod
    def get_candidate_pairs(
        df_data: pd.DataFrame,
        size_1: int,
        blocking_props: List[str],
        generate_vectors: Callable,
        nearest_neighbours: Callable,
        ntop: int,
        query_strategy: str = 'smaller'
    ) -> Tuple[pd.MultiIndex, pd.Series, np.ndarray]:
        """For each given blocking property, the method creates vectors of all property values
        in the first and second dataset. Depending on the given query_strategy, the vectors of
        the first dataset are used as index vectors and the vectors of the second dataset as
        query vectors or vice versa. The nearest neighbours are searched and the resulting
        candidate pairs are combined for all blocking properties. The union of all candidate
        pairs is returned as multi index - together with vectors and values for all blocking
        properties.

        :param df_data: The property values.
        :type df_data: pd.DataFrame
        :param size_1: The number of rows in the first dataset.
        :type size_1: int
        :param blocking_props: The blocking properties.
        :type blocking_props: List[str]
        :param generate_vectors: The method to create vectors from the property values.
        :type generate_vectors: Callable
        :param nearest_neighbours: The nearest neighbour search algorithm.
        :type nearest_neighbours: Callable
        :param ntop: The number of nearest neighbours to return.
        :type ntop: int
        :param query_strategy: The query strategy:
            'first' means use vectors from the first dataset as index vectors and the other ones
            as query vectors,
            'second' means use vectors from the second dataset as index vectors,
            'smaller' means use the smaller set of vectors as index vectors,
            'larger' means use the larger set of vectors as index vectors,
            , defaults to 'smaller'
        :type query_strategy: str, optional
        :return: The candidate pairs, all property values and corresponding vectors.
        :rtype: Tuple[pd.MultiIndex, pd.Series, np.ndarray]
        """

        df_values = df_data[blocking_props]
        values, df_index_array = NearestNeighbourBlocking._create_value_index_array(
            df_values)
        start_time = time.time()
        vectors = generate_vectors(values)
        diff = time.time() - start_time
        logging.info(
            'generated vectors=%s, time=%s, df_values=%s, df_index_array=%s, values=%s',
            vectors.shape, diff, df_values.shape, df_index_array.shape,
            len(values))

        start_time = time.time()
        all_candidates = set()
        total_time_nn_search = 0
        for j, prop in enumerate(blocking_props):
            value_index_column = df_index_array[j]
            candidates = NearestNeighbourBlocking._get_candidates_by_property(
                vectors, value_index_column, size_1, nearest_neighbours, ntop,
                query_strategy)
            all_candidates = all_candidates.union(candidates)
            total_time_nn_search = time.time() - start_time
            logging.info(
                'blocking prop=%s, new candidates=%s, all candidates=%s, total time nn search=%s',
                prop, len(candidates), len(all_candidates),
                total_time_nn_search)

        candidates_as_multi_index = pd.MultiIndex.from_tuples(
            tuples=all_candidates, names=['idx_1', 'idx_2'])
        candidates_as_multi_index = matchain.util.sort_pairs(
            candidates_as_multi_index)
        return candidates_as_multi_index, values, vectors


def run(
    config: dict, df_data: Optional[pd.DataFrame], size_1: int,
    token_index_pruned: Optional[pd.DataFrame]
) -> Tuple[pd.MultiIndex, Optional[pd.Series], Optional[np.ndarray]]:
    """Entry point as part of the command chain
    for blocking and computing the set of candidate pairs.
    """
    config_blocking = config['blocking']
    blocking_name = config_blocking['name']
    #try:
    #    blocking_name = config_blocking['name']
    #except KeyError:
    #    blocking_name = 'token'

    # string values and corresponding (embedding) vectors that might be reused
    # when computing the similarity scores of the candidate pairs
    values = None
    vectors = None

    if blocking_name == 'token' and token_index_pruned is not None:
        # exact token blocking
        candidate_pairs = TokenBlocking.get_candidate_pairs(token_index_pruned)

    else:
        # blocking by approximate nearest neighbour search for shingle or embedding vectors

        if df_data is None:
            raise ValueError('df_data must not be None')

        vector_type = config_blocking['vector_type']
        if vector_type == 'embedding':
            model = config['similarity']['embedding_model']
            device = config['similarity']['embedding_device']
            batch_size = config['similarity']['embedding_batch_size']
            wrapper = matchain.similarity.SentenceTransformerWrapper(
                model, device, batch_size)
            generate_vectors = wrapper.generate_vectors

        elif vector_type == 'shingle_tfidf':
            shingle_size = config_blocking['shingle_size']
            wrapper = ShingleWrapper(shingle_size)
            generate_vectors = wrapper.generate_vectors

        else:
            raise ValueError(f'unknown vector_type={vector_type}')

        blocking_threshold = config_blocking['blocking_threshold']
        njobs = config_blocking['njobs']
        if blocking_name == 'bruteforce':
            nn_inst = NNBruteForce(blocking_threshold)
        elif blocking_name == 'sklearn':
            nn_inst = NNWrapperSklearn(blocking_threshold, njobs=njobs)
        elif blocking_name == 'sparsedottopn':
            nn_inst = NNWrapperSparseDotTopn(blocking_threshold)
        elif blocking_name == 'nmslib':
            nn_inst = NNWrapperNmslib(blocking_threshold, njobs=njobs)
        elif blocking_name == 'faiss':
            nn_inst = NNWrapperFaiss(blocking_threshold)
        else:
            raise ValueError(f'unknown blocking_name={blocking_name}')

        nearest_neighbours = nn_inst.nearest_neighbours
        blocking_props = config['dataset']['blocking_props']
        ntop = config_blocking['ntop']
        query_strategy = config_blocking['query_strategy']
        candidate_pairs, values, vectors = NearestNeighbourBlocking.get_candidate_pairs(
            df_data,
            size_1,
            blocking_props,
            generate_vectors,
            nearest_neighbours,
            ntop=ntop,
            query_strategy=query_strategy)

        if vector_type != 'embedding':
            values = None
            vectors = None

    if config.get('autocal'):
        candidate_file = config['autocal'].get('file_candidates')
        if candidate_file:
            candidate_pairs_read = _read_candidate_pairs(
                candidate_file, size_1)
            n_orig = len(candidate_pairs)
            #candidate_pairs = candidate_pairs_read
            candidate_pairs = candidate_pairs.intersection(
                candidate_pairs_read)
            logging.info(
                'read candidate file=%s, read=%s, original=%s, resulting candidates=%s',
                candidate_file,
                len(candidate_pairs_read),
                n_orig,
                len(candidate_pairs),
            )

    logging.info('candidate pairs=%s', len(candidate_pairs))
    return candidate_pairs, values, vectors
