# helper_function.py
# Kordel France
########################################################################################################################
# This file establishes helper functions used for various tasks in constructing deep learning algorithms.
########################################################################################################################


import numpy as np
import progressbar


def auto_generate_batches(X, y=None, batch_size=64):
    """
    Automatically generates batches from the training and testing data
    """
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]


def train_test_split(X, y, test_size=0.5):
    """
    This is an upgrade on my previous train/test split that is specific for neural
    network training. It contains some key dimensionality checks.
    """
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test




def accuracy_score(y_true, y_hat):
    """
    Compare y_true to y_hat and return the accuracy
    """
    accuracy = np.sum(y_true == y_hat, axis=0) / len(y_true)
    return accuracy


"""This is a progress bar indicator for neural network training similar to what Keras
and Tensorflow use. Here is the source I leveraged to make it:

Baweja, C. (2020, March 25). A complete guide to using progress bars in Python. Medium. Retrieved November 14, 2021, from https://towardsdatascience.com/a-complete-guide-to-using-progress-bars-in-python-aa7f4130cda8. 

"""
status_indicator = [
    'training: ', progressbar.Percentage(),
    ' ',
    progressbar.Bar(marker='.', left='{', right='}'),
    ' ',
    progressbar.ETA()
]


