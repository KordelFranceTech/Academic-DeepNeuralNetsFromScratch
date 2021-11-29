# logistic_regression.py
# Kordel France
########################################################################################################################
# This file establishes the protocol and driver function and class for a logistic regression algorithm.
########################################################################################################################


from Lab4.config import DEMO_MODE
import numpy as np
import time


def compute_sigmoid(values):
    """
    Computes the logistic (sigmoid) between values
    :param values: list - a list of values to compute sigmoid over
    """
    s = 2.71828 ** (-values)
    return 1.0 / (1.0 + s)
    # return 1.0 / (1.0 + np.exp(-values))


def compute_log_likelihood(weights, x_data, y_data):
    """
    Computes the log likelihood between explanatory and response vars and their weights
    :param weights: list - list of floating point values
    :param x_data: list - list of explanatory values
    :param y_data: list - list of response values
    """
    values = np.dot(x_data, weights)
    log_likelihood = np.sum(y_data * values - np.log(1 + np.exp(values)))
    return log_likelihood


def fit_logistic_regression(x_data: list, y_data: list, epochs: int, learning_rate: float, should_add_bias: bool=False):
    """
    Fits a logistic regression equation to a set of data with specific hyperparameters.
    :param x_data: list - list of explanatory values
    :param y_data: list - list of response values
    :param epochs: int - number of training iterations
    :param learning_rate: float - the rate at which function should replace old knowledge with new
    :param should_add_bias: bool - whether or not bias should be added to and rectified in the model
    """
    # prepare model for bias
    if should_add_bias:
        # intercept = np.ones((features.shape[0], 1))
        intercept = [[1]] * x_data.shape[0]
        x_data = np.hstack((intercept, x_data))

    # init weights - may optionally init to zero
    # weights = np.zeros(features.shape[1])
    weights = [0] * x_data.shape[1]

    # start training iterations
    for index_i in range(0, epochs):
        values = np.dot(x_data, weights)
        y_hats = compute_sigmoid(values=values)

        error = y_data - y_hats
        gradient = np.dot(x_data.T, error)
        weights += learning_rate * gradient

        if DEMO_MODE:
            print(f'\n\ny-data: {y_data}')
            print(f'\ny-hats: {y_hats}')
            print(f'\nerror: {error}')
            print(f'\ngradient: {gradient}')
            print(f'\nweights: {weights}')
            time.sleep(10)
            exit()

        if index_i % 1000 == 0:
            print(compute_log_likelihood(weights=weights, x_data=x_data, y_data=y_data))

    return weights, intercept


