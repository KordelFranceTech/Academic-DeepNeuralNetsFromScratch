# encoding.py
# Kordel France
########################################################################################################################
# This file establishes the driver and helper functions used for the encoding of data.
########################################################################################################################


import pandas as pd
import Lab4.helpers
from Lab4.config import DEBUG_MODE


def encode_ordinal_data(dataframe: pd.DataFrame, cols: [int]):
    """
    Prepares and encodes the data as ordinal for designated columns
    :param dataframe: pd.DataFrame - the dataset to encode over
    :param cols: [int] - the specified columns in the dataset to encode
    :return dataframe: pd.DataFrame - the newly encoded dataset
    """

    if DEBUG_MODE:
        print(f'\n\tdata before ordinal encoding:\n\t{dataframe.head(5)}')

    # initialize a dictionary to store the columns we want to encode
    col_vals = []
    col_dict = dict()

    # iterate over every specified column and encode its data
    for index_i in range(0, len(cols)):
        # iterate over each value in the dataframe and populate the working col_vals variable
        for index_j in range(0, len(dataframe)):
            col_val = dataframe.iloc[index_j, cols[index_i]]
            if DEBUG_MODE:
                print(f'previous ordinal column value: {col_val}')
            if col_val not in col_vals:
                col_vals.append(col_val)
            Lab4.helpers.COST_COUNTER += 1

        # encode categorical data as numerical data
        # so get the corresponding value from a helper dictionary, alpha_list
        for index_j in range(0, len(col_vals)):
            col_dict[col_vals[index_j]] = Lab4.helpers.alpha_list[index_j]
            Lab4.helpers.COST_COUNTER += 1

        # rebuild the dataframe with the encoded column
        for index_j in range(0, len(dataframe)):
            col_val = dataframe.iloc[index_j, cols[index_i]]
            dataframe.iloc[index_j, cols[index_i]] = col_dict[col_val]
            if DEBUG_MODE:
                print(f'\t new ordinal column value: {dataframe.iloc[index_j, cols[index_i]]}')
            Lab4.helpers.COST_COUNTER += 1

    if DEBUG_MODE:
        print(f'\n\tdata after ordinal encoding:\n\t{dataframe.head(5)}')

    # the newly encoded dataframe
    return dataframe


def encode_nominal_data(dataframe: pd.DataFrame, cols: [int]):
    """
    Prepares and encodes the data as nominal for designated columns
    :param dataframe: pd.DataFrame - the dataset to encode over
    :param cols: [int] - the specified columns in the dataset to encode
    :return dataframe: pd.DataFrame - the newly encoded dataset
    """
    if DEBUG_MODE:
        print(f'\n\tdata before nominal encoding:\n\t{dataframe.head(5)}')

    # initialize a dictionary to store the columns we want to encode
    col_vals = []
    col_dict = dict()

    # iterate over every specified column and encode its data
    for index_i in range(0, len(cols)):
        # iterate over each value in the dataframe and populate the working col_vals variable
        for index_j in range(0, len(dataframe)):
            col_val = dataframe.iloc[index_j, cols[index_i]]
            if DEBUG_MODE:
                print(f'previous nominal column value: {col_val}')
            if col_val not in col_vals:
                col_vals.append(col_val)
            Lab4.helpers.COST_COUNTER += 1

        # encode categorical data as numerical data
        # so get the corresponding value from a helper dictionary, alpha_list
        for index_j in range(0, len(col_vals)):
            try:
                col_dict[col_vals[index_j]] = Lab4.helpers.alpha_list[index_j]
                col_dict[col_vals[index_j]] = float(convert_value_to_binary(index_j))
            except IndexError:
                col_dict[col_vals[index_j]] = 1000
                col_dict[col_vals[index_j]] = float(convert_value_to_binary(index_j))
            Lab4.helpers.COST_COUNTER += 1

        # rebuild the dataframe with the encoded column
        for index_j in range(0, len(dataframe)):
            col_val = dataframe.iloc[index_j, cols[index_i]]
            dataframe.iloc[index_j, cols[index_i]] = col_dict[col_val]
            if DEBUG_MODE:
                print(f'\t new nominal column value: {dataframe.iloc[index_j, cols[index_i]]}')
            Lab4.helpers.COST_COUNTER += 1

    if DEBUG_MODE:
        print(f'\n\tdata after nominal encoding:\n\t{dataframe.head(5)}')

    # the newly encoded dataframe
    return dataframe


def convert_value_to_binary(conversion_value: int):
    """
    Convert a given integer to its binary value for the purposes of categorical data transform
    :param conversion_value: int - the value to convert into binary
    :return binary_value: str - the equivalently converted binary value
    """
    if DEBUG_MODE:
        print(f'pre-converted nominal value: {conversion_value}')
    binary_list: [str] = []
    binary_value: str = ''

    ## v1
    # start_value: int = conversion_value
    # while start_value != 0:
    #     value = start_value // 2
    #     binary_list.append(str(int(value)))
    #     start_value //= 2

    ## v2
    start_value = format(conversion_value, 'b')
    for value in start_value:
        binary_list.append(str(value))

    for value in range(0, 4 - len(binary_list)):
        binary_value += '0'

    for value in binary_list:
        binary_value += str(value)

    if DEBUG_MODE:
        print(f'post-converted nominal value: {binary_value}')

    Lab4.helpers.COST_COUNTER += 1
    # the newly converted binary value
    return binary_value


