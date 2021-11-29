# classification_algorithm.py
# Kordel France
########################################################################################################################
# This file establishes helper functions used for the facilitation of different classification algorithms.
########################################################################################################################


import pandas as pd
import Lab4.helpers
from Lab4.config import DEBUG_MODE
from Lab4.Metric import Metric


def train_majority_predictor_algorithm(dataframe: pd.DataFrame, col: int):
    """
    Prepares and implements the majority predictor classification algorithm over a specified variable
    :param dataframe: pd.DataFrame - the test dataset to use for learning
    :param col: int - the specified variable in the dataset to learn
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print(f'\nperforming majority output algorithm')
        print(f'\tdataframe:\n\t{dataframe.head(5)}')

    # initialize an empty dictionary and working dataset
    mp_dict = {}
    data: pd.DataFrame = dataframe.sort_values(by=[col], ascending=True)
    # initialize return values
    majority: int = 0
    majority_key: str = ''
    # set up values for f1 score
    true_pos: int = 0
    false_pos: int = 0
    false_neg: int = 0

    # find the majority value within the dictionary
    # compare each value within the dictionary to the highest occurred value
    for index_i in range(0, len(data)):
        cur_key = data.iloc[index_i, col]
        if cur_key in mp_dict.keys():
            mp_dict[cur_key] += 1
            if mp_dict[cur_key] > majority:
                majority_key = cur_key
                majority = mp_dict[cur_key]
                true_pos += 1
            else:
                false_pos += 1
        else:
            mp_dict[cur_key] = 1
            false_neg += 1
        Lab1.helpers.COST_COUNTER += 1

    # compute f1 score
    numerator: float = true_pos
    denominator: float = true_pos + (0.5*(false_pos + false_neg))
    f1_score: float = numerator / denominator

    metric = Metric(explanatory_col=col,
                    response_col=0,
                    is_regression=False,
                    regresion_option=-1,
                    equation='majority predictor',
                    coeff0=0.0,
                    coeff1=0.0,
                    correlation_train=0.0,
                    mse_train=0.0,
                    class_score_train=majority,
                    class_value_train=majority_key,
                    f1_score_train=f1_score,
                    correlation_test=0,
                    mse_test=0,
                    class_score_test=0,
                    class_value_test='',
                    f1_score_test=0,
                    correlation_val=0,
                    mse_val=0,
                    class_score_val=0,
                    class_value_val='',
                    f1_score_val=0)

    if DEBUG_MODE:
        print(f'\n\t{mp_dict}')
        print(f'\n\tmajority value: {majority_key} with {majority} occurrences')

    # return the majority value with its number of occurences
    return metric


def evaluate_majority_predictor_algorithm(dataframe: pd.DataFrame, col: int, metric: Metric):
    """
    Prepares and implements the majority predictor classification algorithm over a specified variable
    :param dataframe: pd.DataFrame - the test dataset to use for learning
    :param col: int - the specified variable in the dataset to learn
    :param metric: Metric - an object containing statistical metrics and performance info
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print(f'\nperforming majority output algorithm')
        print(f'\tdataframe:\n\t{dataframe.head(5)}')

    # initialize an empty dictionary and working dataset
    mp_dict = {}
    data: pd.DataFrame = dataframe.sort_values(by=[col], ascending=True)
    # initialize return values
    majority: int = 0
    majority_key: str = ''
    # set up values for f1 score
    true_pos: int = 0
    false_pos: int = 0
    false_neg: int = 0

    # find the majority value within the dictionary
    # compare each value within the dictionary to the highest occurred value
    for index_i in range(0, len(data)):
        cur_key = data.iloc[index_i, col]
        if cur_key in mp_dict.keys():
            mp_dict[cur_key] += 1
            if mp_dict[cur_key] > majority:
                majority_key = cur_key
                majority = mp_dict[cur_key]
                true_pos += 1
            else:
                false_pos += 1
        else:
            mp_dict[cur_key] = 1
            false_neg += 1
        Lab1.helpers.COST_COUNTER += 1

    # compute f1 score
    numerator: float = true_pos
    denominator: float = true_pos + (0.5 * (false_pos + false_neg))
    f1_score: float = numerator / denominator

    if DEBUG_MODE:
        print(f'\n\t{mp_dict}')
        print(f'\n\tmajority value: {majority_key} with {majority} occurrences')

    if metric.class_value_test == '':
        metric.class_score_test = majority
        metric.class_value_test = majority_key
        metric.f1_score_test = f1_score
    else:
        metric.class_score_val = majority
        metric.class_value_val = majority_key
        metric.f1_score_val = f1_score

    # return the majority value with its number of occurrences
    return metric



