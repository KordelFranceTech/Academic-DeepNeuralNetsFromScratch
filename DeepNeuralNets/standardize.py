# standardize.py
# Kordel France
########################################################################################################################
# This file establishes the driver and helper functions used for the standardizing of data.
########################################################################################################################


import pandas as pd
import Lab4.helpers
from Lab4.config import DEBUG_MODE


def standardize_data_via_z_score(dataframe: pd.DataFrame, cols: [int]):
    """
    Prepares and discretizes the data for designated columns, number of bins, and with eq bin freq or bin width
    :param dataframe: pd.DataFrame - the dataset to standardize
    :param cols: [int] - the specified columns in the dataset to standardize
    :return data: pd.DataFrame - the newly discretized dataset
    """
    if DEBUG_MODE:
        print(f'\n\tdata before standardization:\n\t{dataframe.head(5)}')

    # establish a working data variable
    data: pd.DataFrame = dataframe

    # iterate over each specified column and standardize its data
    for index_i in range(0, len(cols)):
        col = cols[index_i]
        x_sum = 0
        n = len(data)

        # compute the column sum to be used to compute the column mean
        for index_j in range(0, len(data)):
            x_i = float(data.iloc[index_j, col])
            x_sum += float(x_i)
            Lab4.helpers.COST_COUNTER += 1

        # compute the mean
        x_bar = x_sum / n
        # initialize the variance
        x_var = 0
        # compute the variance
        for index_j in range(0, len(data)):
            x_i = float(data.iloc[index_j, col])
            x_var += pow(x_i - x_bar, 2)
            Lab4.helpers.COST_COUNTER += 1

        # divide variance by degrees of freedom
        x_var /= (n - 1)
        # compute standard deviation
        x_sigma = pow(x_var, 0.5)
        # compute z-score and set it as the new value in the dataset
        for index_j in range(0, len(data)):
            x_i = float(data.iloc[index_j, col])
            z_score = (x_i - x_bar) / x_sigma
            data.iloc[index_j, col] = z_score
            Lab1.helpers.COST_COUNTER += 1

    if DEBUG_MODE:
        print(f'\n\tdata after standardization:\n\t{data.head(5)}')

    return data


def standardize_data_via_minimax(dataframe: pd.DataFrame, cols: [int]):
    if DEBUG_MODE:
        print(f'\n\tdata before standardization:\n\t{dataframe.head(5)}')

    # establish a working data variable
    data: pd.DataFrame = dataframe

    # iterate over each specified column and standardize its data
    for index_i in range(0, len(cols)):
        col = cols[index_i]
        x_min = 1000000000
        x_max = -1000000000

        # find the min and max value of the data
        for index_j in range(0, len(data)):
            x_i = float(data.iloc[index_j, col])
            if x_i < x_min:
                x_min = x_i
            if x_i > x_max:
                x_max = x_i
            Lab4.helpers.COST_COUNTER += 1

        # get the current value, normalize it, and set it as the new variable in the dataset
        for index_j in range(0, len(data)):
            x_i = float(data.iloc[index_j, col])
            x_norm = (x_i - x_min) / (x_max - x_min)
            data.iloc[index_j, col] = x_norm
            Lab4.helpers.COST_COUNTER += 1

    if DEBUG_MODE:
        print(f'\n\tdata after standardization:\n\t{data.head(5)}')

    # the newly standardized dataset
    return data


