# discretize.py
# Kordel France
########################################################################################################################
# This file establishes the driver and helper functions used for the discretization of data.
########################################################################################################################


import pandas as pd
import math
import Lab4.helpers
from Lab4.config import DEBUG_MODE


def discretize_data(dataframe: pd.DataFrame, num_bins: int, eq_freq: bool, eq_width: bool, cols: [int]):
    """
    Prepares and discretizes the data for designated columns, number of bins, and with eq bin freq or bin width
    :param dataframe: pd.DataFrame - the dataset to discretize
    :param num_bins: int - number of bins to iterate over and separately discretize
    :param eq_freq: bool - set this parameter to specify bins of equal frequency
    :param eq_width: bool - set this parameter to specify bins of equal width
    :param cols: [int] - the specified columns in the dataset to discretize
    :return data: pd.DataFrame - the newly discretized dataset
    """

    # establish a working data variable
    data: pd.DataFrame = dataframe

    if DEBUG_MODE:
        print(f'\ndiscretizing for dataframe:\n\t{dataframe.head(5)}')
        print(f'\tnumber of bins: {num_bins}\n\tequal width: {eq_width}\n\tequal freq: {eq_freq}')

    # iterate over each specified column and discretize its data
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
            Lab1.helpers.COST_COUNTER += 1

        # discretize bins of equal frequency
        if eq_freq:
            bin_width = (x_max - x_min) / num_bins
            bins = []
            bin_min_bound = x_min
            bin_max_bound = x_min + bin_width
            data = dataframe.sort_values(by=[col], ascending=True)

            # establish the bin span
            for index_j in range(0, num_bins):
                bin_span = [bin_min_bound, bin_max_bound]
                bins.append(bin_span)
                bin_min_bound = bin_max_bound
                bin_max_bound += bin_width
                Lab1.helpers.COST_COUNTER += 1

            if DEBUG_MODE:
                print(f'\nbins: {bins}')
                print(f'\tbin max bound: {bin_max_bound},\tbin min bound: {bin_min_bound}')

            # compute new value for each existing value in dataframe, then swap
            for index_j in range(0, len(data)):
                x_i = float(data.iloc[index_j, col])

                for index_k in range(0, num_bins):
                    cur_bin = bins[index_k]

                    if (x_i >= cur_bin[0]) and (x_i < cur_bin[1]):
                        data.iloc[index_j, col] = f'[{round(cur_bin[0], 3)}, {round(cur_bin[1], 3)})'
                        if DEBUG_MODE:
                            print(f'----\n\tpre-binned value: {x_i}\n\tpost-binned value: {index_k}')
                        break
                    else:
                        data.iloc[index_j, col] = f'[{cur_bin[0]}, inf)'
                    Lab1.helpers.COST_COUNTER += 1

        # discretize bins of equal width
        elif eq_width:
            n: int = len(data)
            bin_width = int(math.ceil(n / num_bins))
            bin_count = 0
            bins = []
            bin_min_bound = x_min
            bin_max_bound = data[col][bin_width]
            data = dataframe.sort_values(by=[col], ascending=True)

            # establish the bin span
            for index_j in range(1, num_bins + 1):
                bin_span = [bin_min_bound, bin_max_bound]
                bins.append(bin_span)
                bin_min_bound = bin_max_bound
                row: int = int(bin_width * index_j)
                if index_j >= num_bins - 1:
                    bin_max_bound = x_max
                else:
                    bin_max_bound = data[col][row]
                Lab1.helpers.COST_COUNTER += 1

            if DEBUG_MODE:
                print(f'\nbins: {bins}')
                print(f'\tbin width: {bin_width}')

            # compute new value for each existing value in dataframe, then swap
            for index_j in range(0, len(data)):
                x_i = float(data.iloc[index_j, col])

                for index_k in range(0, num_bins):
                    cur_bin = bins[index_k]

                    if (x_i >= cur_bin[0]) and (x_i < cur_bin[1]):
                        data.iloc[index_j, col] = f'[{round(cur_bin[0], 3)}, {round(cur_bin[1], 3)})'
                        if DEBUG_MODE:
                            print(f'----\n\tpre-binned value: {x_i}\n\tpost-binned value: {index_k}')
                        break
                    else:
                        data.iloc[index_j, col] = f'[{cur_bin[0]}, inf)'
                    Lab1.helpers.COST_COUNTER += 1


    if DEBUG_MODE:
        print(f'\n\tdata after discretization:\n\t{data}')

    # the newly discretized dataset
    return data



