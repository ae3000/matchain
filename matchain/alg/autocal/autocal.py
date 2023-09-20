"""This module implements AutoCal, an unsupervised algorithm for matching two datasets.
It is based on the idea of calibrating similarity scores to obtain a better separation between
matches and non-matches. The algorithm is described in the following preprint:
https://como.ceb.cam.ac.uk/media/preprints/c4e-293-2023-01-26.pdf

AutoCal takes as input a similarity matrix and a list of properties (columns)
that are used for matching.
Each row of the similarity matrix corresponds to a candidate pair, each column corresponds
to a property.

The implementation of AutoCal in this module consists of the following main steps:

1. Calculate the maximum similarity scores.

This step is done for each dataset entity (instance, record) and for each property
by taking the maximum similarity score over all candidate pairs having this entity
as first or second entity.

2. Calibrate the similarity scores.

For this step, the range for similarity values, i.e. the interval between 0 and 1,
is divided into bins (subintervals) of the specified size delta.
For each bin, we count the number of candidate pairs having a maximum similarity
score falling into this bin. We also count the number of candidate pairs having a similarity
score falling into this bin. The ratio of these two counts is the calibrated similarity score.
Since each candidate pair consists of two entities, there are actually two maximum similarity
values. Thus, for each candidate pair and property, this step computes two calibrated
similarity scores .

3. Compute total scores for each candidate pair.

This step aggregates the calibrated similarity scores from step 2 to obtain a total score
for each candidate pair.

4. Estimate the best threshold.

This step estimates the best threshold for separating matches and non-matches.
AutoCal predicts all candidate pairs having a total score above the estimated threshold as
matches.
"""
import collections
import logging
import os
import os.path
from typing import Dict, List, Optional, Tuple

import matchain.base
import matchain.util
import numpy as np
import pandas as pd


class AutoCal():
    """This class implements the AutoCal algorithm. See the module docstring for details.
    """

    def __init__(self, df_sim: pd.DataFrame):
        """Initializes the AutoCal instance.

        :param df_sim: the similarity matrix, each row corresponds to a candidate pair
        :type df_sim: pd.DataFrame
        """
        self.df_sim = df_sim
        self.df_max_scores_1: pd.DataFrame
        self.df_max_scores_2: pd.DataFrame
        self.df_autocalibrated_scores_1: pd.DataFrame
        self.df_autocalibrated_scores_2: pd.DataFrame
        self.df_total_scores: pd.DataFrame
        self.estimated_threshold: float
        self.majority_matches = None

    def start(self, property_mapping: List[dict],
              delta: float) -> Tuple[pd.DataFrame, float]:
        """Performs the AutoCal algorithm.

        :param property_mapping: Describes which properties of the two input dataframes
            are used for matching.
        :type property_mapping: List[dict]
        :param delta: step size to divide the similarity interval between 0 and 1
            into bins
        :type delta: float
        :return: Total scores for all candidate pairs and the estimated threshold
        :rtype: Tuple[pd.DataFrame, float]
        """

        dataset_id = 1
        self.df_max_scores_1 = AutoCal._calculate_maximum_similarity(
            self.df_sim, dataset_id)
        df_scores = AutoCal._calculate_auto_calibrated_total_scores(
            self.df_sim, property_mapping, self.df_max_scores_1, delta)
        self.df_autocalibrated_scores_1 = AutoCal._identify_best_total_scores(
            df_scores, dataset_id)

        dataset_id = 2
        self.df_max_scores_2 = AutoCal._calculate_maximum_similarity(
            self.df_sim, dataset_id)
        df_scores = AutoCal._calculate_auto_calibrated_total_scores(
            self.df_sim, property_mapping, self.df_max_scores_2, delta)
        self.df_autocalibrated_scores_2 = AutoCal._identify_best_total_scores(
            df_scores, dataset_id)

        df_total_scores = AutoCal._combine(self.df_autocalibrated_scores_1,
                                           self.df_autocalibrated_scores_2)

        self.estimated_threshold, self.majority_matches = AutoCal._estimate_best_threshold(
            df_total_scores, delta)

        mask = ~df_total_scores['best']
        df_total_scores.loc[mask, 'score'] = 0.
        self.df_total_scores = df_total_scores
        return self.df_total_scores, self.estimated_threshold

    @staticmethod
    def _calculate_maximum_similarity(df_sim: pd.DataFrame,
                                      dataset_id: int) -> pd.DataFrame:
        """This method refers to step 1 as described in the module docstring.

        :param df_sim: the similarity matrix, each row corresponds to a candidate pair
        :type df_sim: pd.DataFrame
        :param dataset_id: 1 or 2, depending on whether the maximimum similarity
            scores are calculated for the entities of the first or second dataset
        :type dataset_id: int
        :return: maximum similarity scores
        :rtype: pd.DataFrame
        """
        level = dataset_id - 1
        groups = df_sim.groupby(level=level)
        df_max = groups.max(numeric_only=True)
        logging.info(
            'calculated maximum similarity, dataset_id=%s, entities=%s',
            dataset_id, len(df_max))
        logging.debug('maximum similarity statistics:\n%s', df_max.describe())
        return df_max

    @staticmethod
    def _calculate_auto_calibrated_total_scores(df_sim: pd.DataFrame,
                                                property_mapping: List[dict],
                                                df_max_scores: pd.DataFrame,
                                                delta: float) -> pd.DataFrame:
        """This method (together with method _combine) refers to steps 2 and 3
        as described in the module docstring.

        :param df_sim: the similarity matrix, each row corresponds to a candidate pair
        :type df_sim: pd.DataFrame
        :param property_mapping: Describes which properties of the two input dataframes
            are used for matching.
        :type property_mapping: List[dict]
        :param df_max_scores: maximum similarity scores
        :type df_max_scores: pd.DataFrame
        :param delta: step size to divide the similarity interval between 0 and 1
            into bins
        :type delta: float
        :return: total scores for all candidate pairs
        :rtype: pd.DataFrame
        """

        equivalent_prop_prop_counts = collections.Counter()
        for propmap in property_mapping:
            key = propmap['prop1'] + "_" + propmap['prop2']
            equivalent_prop_prop_counts[key] += 1

        ratios_for_all_props = {}
        for propmap in property_mapping:
            col = propmap['column_name']
            series_max = df_max_scores[col]
            series = df_sim[col]
            key = propmap['prop1'] + "_" + propmap['prop2']
            count = equivalent_prop_prop_counts[key]
            ratios_for_all_props[col] = AutoCal._calculate_prop_ratios(
                series_max, series, delta, count)

        prop_count = len(equivalent_prop_prop_counts)
        prop_scores, total_scores = AutoCal._calculate_auto_calibrated_scores(
            df_sim, ratios_for_all_props, prop_count, delta)

        df_scores = pd.DataFrame(data=prop_scores, index=df_sim.index)
        df_scores['score'] = total_scores
        logging.info('calculated auto calibrated scores')
        return df_scores

    @staticmethod
    def _calculate_prop_ratios(series_max: pd.Series, series: pd.Series,
                               delta: float,
                               equivalent_prop_prop_count: int) -> np.ndarray:
        """Calculates the ratio of candidate pairs having
        a maximum similarity score falling into a bin (subinterval)
        and a similarity score falling into the same bin.

        :param series_max: maximum similarity scores for a single property
        :type series_max: pd.Series
        :param series: similarity scores for a single property
        :type series: pd.Series
        :param delta: step size to divide the similarity interval between 0 and 1
            into bins
        :type delta: float
        :param equivalent_prop_prop_count: number of equivalent property pairs, e.g.
            if two different similarity functions are applied to the same property pair,
            then equivalent_prop_prop_count is 2
        :type equivalent_prop_prop_count: int
        :return: ratios for all bins for a single property
        :rtype: np.ndarray
        """

        ratios = []
        cma = []
        cna = []
        grid = np.arange(0, 1.00001, 2 * delta)
        for point in grid:
            mask = (series_max > point - delta) & (series_max <= point + delta)
            count_m = len(series_max[mask])
            mask = (series > point - delta) & (series <= point + delta)
            count_m_and_n = len(series[mask])
            if count_m_and_n == 0:
                ratio = 1.
            else:
                ratio = min(1., count_m / count_m_and_n)
            ratio = ratio / equivalent_prop_prop_count
            ratios.append(ratio)
            cma.append(count_m)
            cna.append(count_m_and_n - count_m)

        logging.debug('ratios=%s', ratios)
        logging.debug('count_m=%s', cma)
        logging.debug('count_n=%s', cna)

        return np.array(ratios)

    @staticmethod
    def _calculate_auto_calibrated_scores(
            df_sim: pd.DataFrame, ratios_for_all_props: Dict[str, np.ndarray],
            prop_count: int, delta: float) -> Tuple[np.ndarray, np.ndarray]:
        """This method calibrates all similarity scores efficiently: It replaces
        the similarity scores by the corresponding ratios pre-calculated
        for all matching properties.

        :param df_sim: the similarity matrix, each row corresponds to a candidate pair
        :type df_sim: pd.DataFrame
        :param ratios_for_all_props: ratios for all bins for all properties
        :type ratios_for_all_props: Dict[str, np.ndarray]
        :param prop_count: number of properties
        :type prop_count: int
        :param delta: step size to divide the similarity interval between 0 and 1
            into bins
        :type delta: float
        :return: calibrated property scores and total scores for all candidate pairs
        :rtype: Tuple[np.ndarray, np.ndarray]
        """

        columns = list(ratios_for_all_props.keys())
        # transpose the numpy array to obtain the shape (#columns, #candidate pairs)
        array_sim = df_sim[columns].to_numpy().T
        inv_delta = 1 / (2 * delta)
        # rint() rounds float to the "closest" int but return type is float
        # thus, astype() as final step is necessary
        array_indices = np.rint(inv_delta * array_sim)
        # set np.nan values to -1
        # alternatively, a masked array with specified fill value -1 could be used
        np.nan_to_num(array_indices, copy=False, nan=-1)
        array_indices = array_indices.astype(int)

        scores = []
        for j, col in enumerate(columns):
            ratios = ratios_for_all_props[col]
            # np.nan values in array_indices were set to -1.
            # for this reason, append 0.0 at the end of ratios
            # such that ratios[-1] returns score value 0.0
            ratios = np.append(ratios, [0.0])
            # apply advanced indexing on ratios with indices
            indices = array_indices[j]
            score = ratios[indices]
            scores.append(score)

        # score_array has the same shape as array_sim: (#columns, #candidate pairs)
        prop_scores = np.vstack(scores).T
        # after two transpositions (for array_sim and prop_scores),
        # axis=1 corresponds to the original row axis of df_sim
        total_scores = np.sum(prop_scores, axis=1) / prop_count
        return prop_scores, total_scores

    @staticmethod
    def _identify_best_total_scores(df_scores: pd.DataFrame,
                                    dataset_id: int) -> pd.DataFrame:
        df_tmp = df_scores[['score']]
        # get the maximum score for all candidate pairs with same level-index
        # the level-index correspondings either to idx_1 (level=0) or idx_2 (level=1)
        level = dataset_id - 1
        df_max = df_tmp.groupby(level=level).max(numeric_only=True)
        df_max.rename(columns={'score': 'maxscore'}, inplace=True)
        # join on level-index
        df_joined = df_tmp.join(df_max)
        # it can happen that multiple candidate pairs with same level-index
        # take the maximum score
        # thus, select all all these candidate pairs
        mask = df_joined['score'] == df_joined['maxscore']
        index_max = df_joined[mask].index
        df_scores['best'] = False
        df_scores.loc[index_max, 'best'] = True
        logging.info('identified best total scores, dataset_id=%s', dataset_id)
        return df_scores

    @staticmethod
    def _combine(df_total_1: pd.DataFrame,
                 df_total_2: pd.DataFrame) -> pd.DataFrame:
        df_t1 = df_total_1[['score', 'best']].copy()
        df_t1.rename(columns={
            'score': 'score_1',
            'best': 'best_1'
        },
                     inplace=True)
        df_t2 = df_total_2[['score', 'best']].copy()
        df_t2.rename(columns={
            'score': 'score_2',
            'best': 'best_2'
        },
                     inplace=True)

        df_combined = pd.concat([df_t1, df_t2], axis=1)
        arr = df_combined.to_numpy().T
        # arr[0] and arr[2] correspond to columns score_1 and score_2, resp.
        combined_score = np.maximum(arr[0], arr[2])
        df_combined['score'] = combined_score.astype(float)
        combined_best = np.logical_or(arr[1], arr[3])
        df_combined['best'] = combined_best.astype(bool)
        n_best_total_scores = len(df_combined[df_combined['best']])
        logging.info('combined total scores=%s, best total scores=%s',
                     len(df_combined), n_best_total_scores)
        return df_combined

    @staticmethod
    def _estimate_best_threshold(df_total: pd.DataFrame,
                                 delta: float) -> Tuple[float, Optional[set]]:
        """This method refers to step 4 as described in the module docstring.

        :param df_total: total scores for all candidate pairs
        :type df_total: pd.DataFrame
        :param delta: step size to divide the similarity interval between 0 and 1
            into bins
        :type delta: float
        :return: estimated best threshold and an alternatively predicted experimental
            set of matches
        :rtype: Tuple[float, Optional[set]]
        """

        logging.info('estimating best threshold')

        mask = ~df_total['best']
        df_nonbest_scores = df_total[mask]['score']
        mask = df_total['best']
        df_best_scores = df_total[mask]['score'].sort_values(ascending=False)

        grid = [round(x, 5) for x in np.arange(0, 1.0001, delta)]

        majority_matches = set()
        # sometimes there is only a poor statistics for thresholds from x_list close to zero
        # thus, a threshold is accepted only if three subsequent ratios are above 1.
        best_pos = -1
        count_above_one = 0
        delta = delta / 2
        ratios = []
        for i, point in enumerate(grid):
            mask = (df_best_scores > point - delta) & (df_best_scores <=
                                                       point + delta)
            candidate_matches = df_best_scores[mask]
            count_best = len(candidate_matches)
            mask = (df_nonbest_scores > point - delta) & (df_nonbest_scores <=
                                                          point + delta)
            candidate_nonmatches = df_nonbest_scores[mask]
            count_nonbest = len(candidate_nonmatches)
            if count_best > 0 and count_best >= count_nonbest:
                new_matches = set(candidate_matches.index)
                majority_matches = majority_matches.union(new_matches)
            if count_nonbest == 0:
                ratio = count_best
            else:
                ratio = count_best / count_nonbest
            ratios.append((ratio, count_best, count_nonbest))

            if count_above_one < 3:
                if ratio < 1.:
                    count_above_one = 0
                    best_pos = -1
                else:
                    count_above_one += 1
                    if best_pos == -1:
                        best_pos = i

        index_threshold = min(len(grid) - 1, best_pos)
        best_threshold = grid[index_threshold]

        logging.debug('grid=%s', grid)
        logging.debug('majority matches=%s', len(majority_matches))
        logging.debug('ratios=%s',
                      [(round(r[0], 2), r[1], r[2]) for r in ratios])
        logging.info('estimated best threshold=%s, best pos=%s',
                     best_threshold, best_pos)

        return best_threshold, majority_matches

    def dump(self, dir_dump: str):
        """Stores intermediate and final results of AutoCal as CSV files
        to the specified directory.

        :param dir_dump: Dump directory
        :type dir_dump: str
        """
        logging.info('dumping results to %s', dir_dump)
        os.makedirs(dir_dump, exist_ok=True)
        self.df_sim.to_csv(f'{dir_dump}/sim.csv')
        self.df_max_scores_1.to_csv(f'{dir_dump}/max_scores_1.csv')
        self.df_max_scores_2.to_csv(f'{dir_dump}/max_scores_2.csv')
        self.df_total_scores.to_csv(f'{dir_dump}/total_scores.csv')
        self.df_sim.to_csv(f'{dir_dump}/scores.csv')
        self.df_autocalibrated_scores_1.to_csv(
            f'{dir_dump}/autocal_scores_1.csv')
        self.df_autocalibrated_scores_2.to_csv(
            f'{dir_dump}/autocal_scores_2.csv')


def run(
    config: dict, size_1: int, df_sim: pd.DataFrame,
    property_mapping: List[dict]
) -> Tuple[pd.MultiIndex, Optional[float],
           Optional[matchain.base.IterativePredictor]]:
    """Entry point for AutoCal as part of the command chain
    """

    delta = config['autocal']['delta']
    threshold_method = config['autocal']['threshold']
    matcher = AutoCal(df_sim)
    matcher.start(property_mapping, delta)

    dump_dir = config['autocal'].get('dir_dump')
    if dump_dir:
        matcher.dump(dump_dir)

    estimated_threshold = matcher.estimated_threshold
    df_scores = matcher.df_total_scores

    # use original idx_2 record positions (i.e. minus size_1)
    # for predicted matches
    matchain.util.add_to_index(df_scores, offset_1=0, offset_2=-size_1)
    predictor = matchain.base.IterativePredictor(df_scores)
    predicted_matches = predictor.predict(estimated_threshold)

    if threshold_method == 'majority' and matcher.majority_matches:
        predicted_majority_matches = [
            (idx1, idx2 - size_1) for idx1, idx2 in matcher.majority_matches
        ]
        predicted_majority_matches = pd.MultiIndex.from_tuples(
            predicted_majority_matches)
        return predicted_majority_matches, None, None
    assert threshold_method == 'estimated'
    return predicted_matches, estimated_threshold, predictor
