# learning_algorithm.py
# Kordel France
########################################################################################################################
# This file establishes the driver and helper functions used for the facilitation of the learning algorithm.
########################################################################################################################


import pandas as pd
import Lab4.regression_algorithm as regression_algorithm
import Lab4.classification_algorithm as classification_algorithm
from Lab4.config import DEBUG_MODE
from Lab4.Metric import Metric


def select_learning_algorithm(train_data: pd.DataFrame,
                              is_classify: bool,
                              explanatory_col: int,
                              response_col: int,
                              regression_option: int):
    """
    Driver function for implementing the learning algorithm.
    :param train_data: pd.Dataframe - the training dataFrame
    :param is_classify: bool - a flag that indicates whether a classification or regression algorithm should be selected
    :param explanatory_col: int - the column indicating the explanatory variable, or x series
    :param response_col: int - the column indicating the response variable, or y series
    :param regression_option: int - the value indicating which regression algorithm to perform
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if is_classify:
        return run_majority_predictor_algorithm(train_data, explanatory_col)
    else:
        return run_regression_algorithm(dataframe=train_data,
                                        explanatory_col=explanatory_col,
                                        response_col=response_col,
                                        regression_option=regression_option)


def run_majority_predictor_algorithm(dataframe: pd.DataFrame, col: int):
    """
    Prepares and implements the majority predictor classification algorithm over a specified variable
    :param dataframe: pd.DataFrame - the test dataset to use for learning
    :param col: int - the specified variable in the dataset to learn
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print(f'\nperforming majority output algorithm')
        print(f'\tdataframe:\n\t{dataframe.head(5)}')

    metric = classification_algorithm.train_majority_predictor_algorithm(dataframe, col)

    # return the majority value with its number of occurences
    return metric


def run_regression_algorithm(dataframe: pd.DataFrame, explanatory_col: int, response_col: int, regression_option: int):
    """
    Prepares and implements the majority predictor classification algorithm over a specified variable
    :param dataframe: pd.DataFrame - the test dataset to use for learning
    :param explanatory_col: int - the column indicating the explanatory variable, or x series
    :param response_col: int - the column indicating the response variable, or y series
    :param regression_option: int - the value indicating which regression algorithm to perform
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print(f'\nsetting up regression algorithm')
        print(f'\tdataframe:\n\t{dataframe.head(5)}')

    # intialize a working dataset
    data: pd.DataFrame = dataframe
    x_vals = []
    y_vals = []

    # build a list of x values and y values to feed the regression equations
    for index in range(0, len(data)):
        x_i = float(data.iloc[index, explanatory_col])
        y_i = float(data.iloc[index, response_col])
        x_vals.append(x_i)
        y_vals.append(y_i)

    # we have our dataset prepared, now fit a regression equation to it
    if regression_option == 0:
        metric: Metric = regression_algorithm.fit_linear_regression_equation(x_vals, y_vals)
        metric.explanatory_col = explanatory_col
        metric.response_col = response_col
        return metric
    elif regression_option == 1:
        metric: Metric = regression_algorithm.fit_inverse_regression_equation(x_vals, y_vals)
        metric.explanatory_col = explanatory_col
        metric.response_col = response_col
        return metric
    elif regression_option == 2:
        metric: Metric = regression_algorithm.fit_power_regression_equation(x_vals, y_vals)
        metric.explanatory_col = explanatory_col
        metric.response_col = response_col
        return metric
    elif regression_option == 3:
        metric: Metric = regression_algorithm.fit_euler_exponential_regression_equation(x_vals, y_vals)
        metric.explanatory_col = explanatory_col
        metric.response_col = response_col
        return metric
    elif regression_option == 4:
        metric: Metric = regression_algorithm.fit_exponential_regression_equation(x_vals, y_vals)
        metric.explanatory_col = explanatory_col
        metric.response_col = response_col
        return metric
    else:
        metric: Metric = regression_algorithm.fit_linear_regression_equation(x_vals, y_vals)
        metric.explanatory_col = explanatory_col
        metric.response_col = response_col
        return metric
