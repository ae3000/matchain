"""This module computes metrics such as precision, recall and F1 score
for predicted matching pairs if the ground truth is known."""
import logging
from typing import Optional, Tuple, cast

import matchain.util
import pandas as pd
from matchain.base import IterativePredictor


def compute_f1(tpos: int, fpos: int, fneg: int) -> Tuple[float, float, float]:
    """Computes the f1 score.

    :param tpos: number of true positives (true matches)
    :type tpos: int
    :param fpos: number of false positives (false matches)
    :type fpos: int
    :param fneg: number of false negatives (false non-matches)
    :type fneg: int
    :return: tuple with precision, recall and f1 score
    :rtype: Tuple[float, float, float]
    """
    precision = 0.
    recall = 0.
    f1score = 0.
    if tpos > 0:
        precision = tpos / (tpos + fpos)
        recall = tpos / (tpos + fneg)
        f1score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1score


def compute_metrics(predicted_matches: pd.MultiIndex,
                    matches: pd.MultiIndex,
                    threshold: Optional[float] = None) -> dict:
    """Computes precision, recall and f1 score for predicted matching pairs
    and returns the result as dictionary. If threshold is not None, threshold
    is added to the result.

    :param predicted_matches: predicted matching pairs
    :type predicted_matches: pd.MultiIndex
    :param matches: true matching pairs
    :type matches: pd.MultiIndex
    :param threshold: threshold which was originally used to compute predicted_matches
    :type threshold: Optional[float], optional
    :return: dictionary with precision, recall, f1 score,
        number of true positives, false positives and false negatives, and optional threshold
    :rtype: dict
    """
    true_matches = matches.intersection(predicted_matches)
    false_matches = predicted_matches.difference(matches)
    false_nonmatches = matches.difference(predicted_matches)
    tpos = len(true_matches)
    fpos = len(false_matches)
    fneg = len(false_nonmatches)

    precision, recall, f1score = compute_f1(tpos, fpos, fneg)
    result = {
        'f1': matchain.util.rnd(f1score),
        'p': matchain.util.rnd(precision),
        'r': matchain.util.rnd(recall),
        'tpos': tpos,
        'fpos': fpos,
        'fneg': fneg
    }
    if threshold is not None:
        res = {'t': threshold}
        res.update(result)
        result = res

    return result


def find_maximum_f1(predictor: IterativePredictor,
                    matches: pd.MultiIndex,
                    n_thresholds=201) -> dict:
    """Predicts matches for equidistant thresholds between 0 and 1 and
    returns the metrics for the threshold with the highest f1 score.

    :param predictor: allows to predict matching pairs for a given threshold
    :type predictor: IterativePredictor
    :param matches: true matching pairs
    :type matches: pd.MultiIndex
    :param n_thresholds: the number of equidistant thresholds between 0 and 1
    :type n_thresholds: int, optional
    :return: dictionary with precision, recall, f1 score,
        number of true positives, false positives and false negatives, and threshold
    :rtype: _type_
    """
    max_f1 = -1.
    max_metrics = {}
    thresholds = predictor.get_thresholds(n_thresholds)
    for threshold in thresholds:
        predicted_matches = predictor.predict(threshold)
        metrics = compute_metrics(predicted_matches, matches, threshold)
        f1 = metrics['f1']
        if f1 >= max_f1:
            max_f1 = f1
            max_metrics = metrics

    return max_metrics


def compute_match_frequencies_for_first_level(size: int,
                                              matches: pd.MultiIndex) -> dict:
    """Computes how often first level index values occur in matches.

    :param size: size of the dataset referring to the first level index
    :type size: int
    :param matches: true matching pairs
    :type matches: pd.MultiIndex
    :return: dictionary with frequencies
    :rtype: dict
    """
    dframe = pd.DataFrame(index=matches)
    dframe.reset_index(names=['level_1', 'level_2'], inplace=True)
    n_nonmatches = size - len(dframe['level_1'].unique())
    frequencies = {0: n_nonmatches}
    # compute the number of second level index values in matches for each first level index value
    ser = dframe['level_1'].value_counts()
    # compute the frequencies of the computer numbers
    df_frequencies = ser.value_counts()
    frequencies.update(df_frequencies.to_dict())
    return frequencies


def compute_match_frequencies(size_1: int, size_2: int,
                              matches: pd.MultiIndex) -> dict:
    """Computes how often an entity in dataset 1 (2) is matched with k=0,1,2 etc.
    entities in dataset 2 (1). If there is at most one matched entity, the result will only
    contain key-value pairs for k=0 and k=1.

    :param size_1: size of dataset 1
    :type size_1: int
    :param size_2: size of dataset 2
    :type size_2: int
    :param matches: set of all true matching pairs
    :type matches: pd.MultiIndex
    :return: dictionary with frequencies
    :rtype: dict
    """
    result = {}
    result[
        'match_frequencies_1_to_2'] = compute_match_frequencies_for_first_level(
            size_1, matches)
    result[
        'match_frequencies_2_to_1'] = compute_match_frequencies_for_first_level(
            size_2, matches.swaplevel())
    return result


def compute_estimated_and_maximum_f1(predicted_matches: pd.MultiIndex,
                                     matches: pd.MultiIndex,
                                     nonmatches: Optional[pd.MultiIndex],
                                     predictor: Optional[IterativePredictor],
                                     threshold: Optional[float],
                                     compute_max_f1: bool = False,
                                     ntop: int = 0) -> dict:
    """The method computes precious, recall and f1 score for the given predicted matching pairs.
    If compute_max_f1 == True and predictor is given, the method also computes the metrics
    with the highest f1 score. All results are returned as dictionary.

    :param predicted_matches: predicted matching pairs
    :type predicted_matches: pd.MultiIndex
    :param matches: true matching pairs
    :type matches: pd.MultiIndex
    :param nonmatches: true non-matching pairs
    :type nonmatches: Optional[pd.MultiIndex]
    :param predictor: allows to predict matching pairs for a given threshold
    :type predictor: Optional[IterativePredictor]
    :param threshold: threshold that was used to predict the matching pairs,
        not need for computation, only added to the result
    :type threshold: Optional[float]
    :param compute_max_f1: _description_, defaults to False
    :type compute_max_f1: bool, optional
    :param ntop: if ntop > 0, the method also computes the metrics for the threshold that
        corresponds to the ntop highest scores
    :type ntop: int, optional
    :return: dictionary with metrics
    :rtype: dict
    """
    result = {}

    if predictor:
        if nonmatches is not None:
            predictor.intersect(matches, nonmatches)
        if ntop:
            t_ntop = round(predictor.threshold_ntop(ntop), 3)
            result['ntop'] = compute_metrics(predicted_matches, matches,
                                             t_ntop)
        if compute_max_f1:
            result['max'] = find_maximum_f1(predictor, matches)

    if nonmatches is not None:
        all_pairs = matches.union(nonmatches)
        predicted_matches = predicted_matches.intersection(all_pairs)
    result['estimated'] = compute_metrics(predicted_matches, matches,
                                          threshold)

    return result


def evaluate(test_file: str,
             predicted_matches: Optional[pd.MultiIndex],
             predictor: Optional[IterativePredictor],
             threshold: Optional[float],
             compute_max_f1: bool = False,
             leak_n_matches: bool = False,
             size_1: int = -1,
             size_2: int = -1,
             candidate_pairs: Optional[pd.MultiIndex] = None) -> dict:
    """The method computes precious, recall and f1 score for the given predicted matching pairs.
    The true matches are read from the given test file.
    Optionally, the method computes further metrics and puts them into the result dictionary.

    :param test_file: The file that contains the true matching pairs.
    :type test_file: str
    :param predicted_matches: predicted matching pairs
    :type predicted_matches: Optional[pd.MultiIndex]
    :param predictor: allows to predict matching pairs for a given threshold
    :type predictor: Optional[IterativePredictor]
    :param threshold: threshold that was used to predict the matching pairs,
        not need for computation, only added to the result
    :type threshold: Optional[float]
    :param compute_max_f1: If True, the method also computes the metrics with the highest f1 score
    :type compute_max_f1: bool, optional
    :param leak_n_matches: If True, the method also computes the metrics for the threshold that
        corresponds to the ntop highest scores where ntop is the number of true matching pairs.
    :type leak_n_matches: bool, optional
    :param size_1: size of the first data set
    :type size_1: int, optional
    :param size_2: size of the second data set
    :type size_2: int, optional
    :param candidate_pairs: If given, the method computes the overlap between candidate pairs
        and true matching and non-matching pairs.
    :type candidate_pairs: Optional[pd.MultiIndex], optional
    :return: dictionary with all results
    :rtype: dict
    """
    result = {}

    ntop = 0
    if candidate_pairs is not None:
        # idx_2 in a candidate pair (idx_1, idx_2) was increased by size_1
        # compared to the original row number of the second data set.
        # For this reason, we read all data with an offset of size_1
        # for the second data set.
        df = matchain.util.read_and_concat_csv_table_ditto_format(
            test_file, offset=size_1, apply_format=False)
        mask = df['label'] == 1
        matches = cast(pd.MultiIndex, df[mask].index)
        # matches that are not considered as candidate pairs become false negatives
        # when calculating precision and recall
        diff_matches = matches.difference(candidate_pairs)
        mask = df['label'] == 0
        nonmatches = cast(pd.MultiIndex, df[mask].index)
        nonmatches_outside = nonmatches.difference(candidate_pairs)

        n_candidate_matches = len(matches) - len(diff_matches)
        result['blocking'] = {
            'matches': len(matches),
            'nonmatches': len(nonmatches),
            'diff_matches': len(diff_matches),
            'diff_nonmatches': len(nonmatches_outside),
            'candidate_matches': n_candidate_matches,
            'candidate_nonmatches': len(candidate_pairs) - n_candidate_matches
        }

        if leak_n_matches:
            ntop = n_candidate_matches

    # zero offset here because idx_2 in predicted matches was
    # already decreased by - size_1
    df = matchain.util.read_csv(test_file, offset=0, apply_format=False)
    mask = df['label'] == 1
    matches = cast(pd.MultiIndex, df[mask].index)
    nonmatches = cast(pd.MultiIndex, df[~mask].index)
    if predicted_matches is not None:
        result['test_set'] = compute_estimated_and_maximum_f1(
            predicted_matches, matches, nonmatches, predictor, threshold,
            compute_max_f1, ntop)

    df = matchain.util.read_and_concat_csv_table_ditto_format(
        test_file, offset=0, apply_format=False)
    mask = df['label'] == 1
    matches = cast(pd.MultiIndex, df[mask].index)
    nonmatches = cast(pd.MultiIndex, df[~mask].index)
    if predicted_matches is not None:
        result['union_set'] = compute_estimated_and_maximum_f1(
            predicted_matches, matches, nonmatches, predictor, threshold,
            compute_max_f1, ntop)

    frequencies = compute_match_frequencies(size_1, size_2, matches)
    result.update(frequencies)

    return result


def run(config: dict,
        size_1: int,
        size_2: int,
        candidate_pairs: Optional[pd.MultiIndex] = None,
        predicted_matches: Optional[pd.MultiIndex] = None,
        threshold: Optional[float] = None,
        predictor: Optional[IterativePredictor] = None,
        true_matches: Optional[pd.MultiIndex] = None,
        true_nonmatches: Optional[pd.MultiIndex] = None) -> dict:
    """Entry point as part of the command chain
    to evaluate the set of matches that have been predicted by an algorithm.
    """
    compute_max_f1 = config['evaluate'].get('compute_max_f1')
    if compute_max_f1 is None:
        compute_max_f1 = False
    leak_n_matches = config['evaluate'].get('leak_n_matches')
    if leak_n_matches is None:
        leak_n_matches = False

    if true_matches is None:
        file_matches = config['dataset'].get('file_matches')
        if not file_matches:
            raise RuntimeError(
                'evaluation is not possible since neither true matches ' +
                'nor match file are given')

        if file_matches.endswith('matches.csv'):
            test_file = file_matches
        else:
            test_file = f'{file_matches}/test.csv'
        result = evaluate(test_file, predicted_matches, predictor, threshold,
                          compute_max_f1, leak_n_matches, size_1, size_2,
                          candidate_pairs)

    elif predicted_matches is not None:
        result = compute_estimated_and_maximum_f1(predicted_matches,
                                                  true_matches,
                                                  true_nonmatches, predictor,
                                                  threshold, compute_max_f1)
    else:
        raise RuntimeError(
            'evaluation is not possible since predicted matches are not given')

    logging.info('metrics=\n%s', result)

    return result
