"""This module contains classes and functions which are used by several algorithms
or used for data exchange between the api module, the chain module and
implemented algorithms."""
from typing import List, Optional, cast

import matchain.mtoken
import matchain.util
import numpy as np
import pandas as pd


class PinBoard():
    """A pure data structure to store results of commands that the chain module has already
    executed. It provides results of previous steps to subsequent commands and the api module.
    """

    def __init__(self) -> None:
        """Configuration dictionary"""
        self.config: dict
        """Data of the first dataset"""
        self.df_1: Optional[pd.DataFrame] = None
        """Data of the second dataset"""
        self.df_2: Optional[pd.DataFrame] = None
        """Size of the first dataset"""
        self.size_1: int
        """Size of the second dataset"""
        self.size_2: int
        """Stacked data of both datasets. The index ranges from 0 to size_1 + size_2 - 1."""
        self.df_data: pd.DataFrame
        """Token manager"""
        self.token_manager: Optional[matchain.mtoken.TokenManager] = None
        """Set of candidate pairs represented as MultiIndex.
        The first level refers to the position in the first dataset,
        the second level refers to the position in the second dataset
        plus the size of the first dataset."""
        self.candidate_pairs: Optional[pd.MultiIndex] = None
        """Values for blocking properties that are used during blocking"""
        self.blocking_values: Optional[pd.Series] = None
        """Vectors corresponding to values (in the same order) that are used during blocking.
        The computation of (embedding) vectors is an expensive operation. The vectors can be
        reused when computing the similarity scores of candidate pairs."""
        self.blocking_vectors: Optional[np.ndarray] = None
        """The dataframe contains the similarity scores of the candidate pairs."""
        self.df_sim: pd.DataFrame
        """Defines which similarity functions are applied to which properties."""
        self.property_mapping: List[dict]
        """The file that stores the walks on graph nodes as used e.g. by node2vec."""
        self.walk_file: str
        """The matches predicted by the algorithm."""
        self.predicted_matches: Optional[pd.MultiIndex] = None
        """The threshold used by the algorithm (if any) to predict matches."""
        self.threshold: float
        """Optionally provided by an algorithm.
        Allows to computed additional metrics such as the maximum f1 score."""
        self.predictor: Optional[IterativePredictor] = None
        """The gold standard matches used for evaluation.
        Can only be set by the API module and is an alternative to the
        configured test file."""
        self.true_matches: Optional[pd.MultiIndex] = None  # api only
        """A set of non-matches used for evaluation.
        Can only be set by the API module and is an alternative to the
        configured test file."""
        self.true_nonmatches: Optional[pd.MultiIndex] = None  # api only
        """The result of the evaluation module"""
        self.evaluation_metrics: dict


class IterativePredictor():
    """This class provides additional functionality for evaluating an algorithm.
    In particular, it allows to predict matches for different thresholds that are decreased
    iteratively from 1 to 0. This is useful for computing the maximum f1 score which indicates
    the potential of an algorithm.
    Some of the implemented algorithms such as AutoCal return both the predicted matches and
    this optional predictor."""

    def __init__(self,
                 df_scores: pd.DataFrame,
                 score_column: str = 'score') -> None:
        self.df_scores = df_scores
        self.df_intersected = df_scores
        self.score_column = score_column

    def intersect(self, matches: pd.MultiIndex,
                  nonmatches: pd.MultiIndex) -> None:
        """Some datasets are published in combination with sets of true matching and nonmatching
        pairs. This allows to train and evaluate supervised methods in a standardized way.
        Some unsupervised methods such as AutoCal create their own set of candidate pairs.
        This method intersects the set of candidate pairs with the union set
        of matches and nonmatches such that the predict method only returns predicted matches
        contained in the union set. This ensures a fair comparison between supervised and
        unsupervised methods. Please note that true matches outside the original set of candidate
        pairs are still counted correctly as false negatives.

        :param matches: the set of true matches
        :type matches: pd.MultiIndex
        :param nonmatches: the set of true nonmatches
        :type nonmatches: pd.MultiIndex
        """
        if nonmatches is not None:
            all_pairs = matches.union(nonmatches)
            intersection = self.df_scores.index.intersection(all_pairs)
            self.df_intersected = self.df_scores.loc[intersection]

    def get_thresholds(self, n_thresholds: int) -> List[float]:
        """Returns a list of equidistant thresholds from 1 to 0.

        :param n_thresholds: The number of thresholds to be returned.
        :type n_thresholds: int
        :return: A list of equidistant thresholds
        :rtype: List[float]
        """
        thresholds = [
            matchain.util.rnd(t)
            for t in np.linspace(1, 0, num=n_thresholds, endpoint=True)
        ]
        return thresholds

    def predict(self, threshold: float) -> pd.MultiIndex:
        """Returns candidate pairs with a score greater or equal
        to the threshold as predicted matches.

        :param threshold: The threshold
        :type threshold: float
        :return: the predicted matches
        :rtype: pd.MultiIndex
        """
        mask = self.df_intersected[self.score_column] >= threshold
        return cast(pd.MultiIndex, self.df_intersected[mask].index)

    def threshold_ntop(self, ntop: int) -> float:
        """Returns the threshold that is required to predict the candidate pairs
        with the ntop highest scores as matches.

        :param ntop: The number of candidate pairs to be predicted as matches.
        :type ntop: int
        :return: The threshold
        :rtype: float
        """
        df_sorted = self.df_intersected.sort_values(by=self.score_column,
                                                    ascending=False)
        pos = min(ntop, len(df_sorted)) - 1
        threshold = df_sorted.iloc[pos][self.score_column]
        return threshold
