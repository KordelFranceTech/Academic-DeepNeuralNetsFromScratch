# evaluation.py
# Kordel France
########################################################################################################################
# This file establishes the driver and helper functions used for computing evaluation metrics of data and algorithms.
########################################################################################################################


import pandas as pd
import Lab4.regression_algorithm as regression_algorithm
import Lab4.classification_algorithm as classification_algorithm
from Lab4.config import DEBUG_MODE
from Lab4.Metric import Metric


def get_evaluation_metrics(dataframe: pd.DataFrame, metric: Metric, needs_val_set: bool):
    """
    Coordinates the evaluation of testing and validation of a dataset if not using K-fold CV
    :param dataframe: pd.DataFrame - the test dataset to use for learning
    :param metric: Metric - an object containing statistical metrics and performance info
    :param needs_val_set: bool - indicates whether or not a validation set should be constructed
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print(f'\n\tgetting evaluation metric for metric: {metric.print_metric()}')
        print(f'true values head:\n\t{dataframe.head(5)}')

    final_metric = get_testing_evaluation(dataframe, metric)
    if needs_val_set:
        final_metric = get_validation_evaluation(dataframe, metric)

    return final_metric


def get_testing_evaluation(dataframe: pd.DataFrame, metric: Metric):
    """
    Facilitates the dataset over a specified algorithm over a train set
    :param dataframe: pd.DataFrame - the test dataset to use for learning
    :param metric: Metric - an object containing statistical metrics and performance info
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print(f'\nperforming majority output algorithm')
        print(f'\tdataframe:\n\t{dataframe.head(5)}')
        print(f'\tresponse column: {metric.response_col}')
        print(f'\texplanatory column: {metric.explanatory_col}')

    # intialize a working dataset
    data: pd.DataFrame = dataframe
    x_vals = []
    y_vals = []

    # build a list of x values and y values to feed the regression equations
    if metric.regression_option >= 0:
        for index in range(0, len(data)):
            x_i = float(data.iloc[index, metric.explanatory_col])
            y_i = float(data.iloc[index, metric.response_col])
            x_vals.append(x_i)
            y_vals.append(y_i)

        # we have our dataset prepared, now fit a regression equation to it
        if metric.regression_option == 0:
            metric_train = regression_algorithm.eval_linear_regression_equation(x_vals, y_vals, metric)
            return metric_train
        elif metric.regression_option == 1:
            metric_train = regression_algorithm.eval_inverse_regression_equation(x_vals, y_vals, metric)
            return metric_train
        elif metric.regression_option == 2:
            metric_train = regression_algorithm.eval_exponential_regression_equation(x_vals, y_vals, metric)
            return metric_train
        elif metric.regression_option == 3:
            metric_train = regression_algorithm.eval_euler_exponential_regression_equation(x_vals, y_vals, metric)
            return metric_train
        elif metric.regression_option == 4:
            metric_train = regression_algorithm.eval_power_regression_equation(x_vals, y_vals, metric)
            return metric_train
    else:
        metric_train = classification_algorithm.evaluate_majority_predictor_algorithm(dataframe, metric.explanatory_col, metric)
        return metric_train


def get_validation_evaluation(dataframe: pd.DataFrame, metric: Metric):
    """
    Facilitates the dataset over a specified algorithm over a validation set
    :param dataframe: pd.DataFrame - the test dataset to use for learning
    :param metric: Metric - an object containing statistical metrics and performance info
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print(f'\nperforming majority output algorithm')
        print(f'\tdataframe:\n\t{dataframe.head(5)}')

    # intialize a working dataset
    data: pd.DataFrame = dataframe
    x_vals = []
    y_vals = []

    # build a list of x values and y values to feed the regression equations
    if metric.regression_option >= 0:
        for index in range(0, len(data)):
            x_i = float(data.iloc[index, metric.explanatory_col])
            y_i = float(data.iloc[index, metric.response_col])
            x_vals.append(x_i)
            y_vals.append(y_i)


        # we have our dataset prepared, now fit a regression equation to it
        if metric.regression_option == 0:
            metric_val = regression_algorithm.eval_linear_regression_equation(x_vals, y_vals, metric)
            return metric_val
        elif metric.regression_option == 1:
            metric_val = regression_algorithm.eval_inverse_regression_equation(x_vals, y_vals, metric)
            return metric_val
        elif metric.regression_option == 2:
            metric_val = regression_algorithm.eval_exponential_regression_equation(x_vals, y_vals, metric)
            return metric_val
        elif metric.regression_option == 3:
            metric_val = regression_algorithm.eval_euler_exponential_regression_equation(x_vals, y_vals, metric)
            return metric_val
        elif metric.regression_option == 4:
            metric_val = regression_algorithm.eval_power_regression_equation(x_vals, y_vals, metric)
            return metric_val
    else:
        metric_val = classification_algorithm.evaluate_majority_predictor_algorithm(dataframe, metric.explanatory_col, metric)
        return metric_val

