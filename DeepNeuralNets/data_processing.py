# data_processing.py
# Kordel France
########################################################################################################################
# This file establishes the driver and helper functions used for the processing, importing, and locating of data.
########################################################################################################################


import pandas as pd
import csv
import os
import shutil
import time
from Lab4 import config


def find_input_file(filename):
    """
    Finds the input file and moves it to the correct directory, 'io_files.'
    :param filename: str - the input file to find
    :returns input_file_path: str - the incorrect path the input file was found at for feedback to user.
    """
    # initialize file paths
    dir_path = os.path.curdir
    input_file_path = ''
    found = False

    # iterate over all items in directory to find filename
    for item in dir_path:
        if item == filename:
            input_file_path = os.path.join(dir_path, item)
            found = True
    # file wasn't found in current directory, so scan others
    if not found:
        for sub_dir in dir_path:
            if not sub_dir.endswith('.idea') and not sub_dir.endswith('.md'):
                for item in os.listdir(os.path.join(sub_dir)):
                    if item == filename:
                        input_file_path = os.path.join(sub_dir, item)
                        shutil.copy(input_file_path, os.path.join('io_files', item))
                        # typically we want to remove the file from the old path
                        # however, we comment this out in the event the copy fails
                        # os.remove(os.path.join(sub_dir, item))

    # the found file path
    return input_file_path


def preprocess_file_data(input_text_file):
    """
    Takes an input file to read data from line by line, character by character, and fills in missing values.
    :param input_text_file: the text file to read data in from.
    written to the output file
    """
    output_string = ''
    output_string += 'Checking input file for errors...'

    # find the maximum columns for our header
    # open the file once
    max_col_count = 0
    with open(input_text_file, 'r') as f:
        reader = csv.reader(f)

        # check number of columns and find the maximum num columns
        for i, row in enumerate(reader):
            if len(row) > max_col_count:
                max_col_count = len(row)

    # open the file a second time and populate missing values
    input_file = open(str(input_text_file), 'r')
    file_values = []
    header_vals = []
    header_count = 0
    line_count = 0

    # read the entire file and fill in missing values according to their type (float, int, str)
    while 1:
        single_char = input_file.read(1)
        if not single_char:
            break
        elif str(single_char) == ' ':
            if line_count > 1 and isinstance(header_vals[header_count], str):
                file_values.append(str('MISSING'))
                file_values.append(',')
            elif line_count > 1 and isinstance(header_vals[header_count], int):
                file_values.append(str(0))
                file_values.append(',')
            elif line_count > 1 and isinstance(header_vals[header_count], float):
                file_values.append(str(0.0))
                file_values.append(',')
            continue
        elif single_char == '\n':
            line_count += 1
            file_values.append('\n')
            continue
        else:
            if line_count < 1:
                header_vals.append(single_char)
                file_values.append(single_char)
                file_values.append(',')
        header_count += 1

    # write the data to a new 'clean' file
    with open(input_text_file, 'w', newline='\n') as csvfile:
        writer = csv.DictWriter(csv, delimiter='\n', fieldnmes=header_vals)
        writer.writeheader()
        for row in file_values:
            writer.writerow(row)
    time.sleep(1.0)


def import_dataset(filename):
    """
    Imports the designated file in to the correct directory, 'io_files.'
    :param filename: str - the file to import
    :returns dataframe: pd.dataframe - the incorrect path the input file was found at for feedback to user.
    """
    # intialize directories
    dir_path = os.listdir(config.IO_DIRECTORY)
    file_path = os.path.join(config.IO_DIRECTORY, filename)

    # iterate over all files in current directory
    for item in dir_path:
        # file found, open it
        if item == filename:
            if config.DEBUG_MODE:
                print(f' file imported: {item}, {filename}')
            with open(file_path, 'r') as f:
                reader = csv.reader(f)

                # check number of columns and find the maximum num columns
                max_col_count = 0
                for i, row in enumerate(reader):
                    if len(row) > max_col_count:
                        max_col_count = len(row)

                # read the file data into a dataframe
                dataframe = pd.read_csv(file_path, header=None)[:100]
                if config.DEBUG_MODE:
                    print(f'\tmax columns: {max_col_count}')
                    print(f'\timported dataframe:\n\t{dataframe.head(5)}')

                # the pd.dataFrame for use in analysis
                return dataframe


def rename_data_files(subdir_name):
    """
    Renames all the files in the specified subdirectory to an extension that can be easily imported
    :subdir_name: string - the sub directory to rename all files in
    """
    # initialize directory of interest to rename contents in
    dir_path = os.listdir(subdir_name)

    # iterate over all files in the directories and change the name of each one
    for item in dir_path:
        if '.' in item:
            prename = str(item)
            postname = prename.split('.')[0]
            suffix = prename.split('.')[1]
            if suffix == 'data' or suffix == 'DATA':
                postname += '_DATA.csv'
                input_file_path = os.path.join(subdir_name, item)
                shutil.copy(input_file_path, os.path.join(subdir_name, postname))
                # typically we want to remove the file from the old path
                # however, we comment this out in the event the copy fails
                # os.remove(os.path.join(dir_path, item))
            elif suffix == 'names' or suffix == 'NAMES':
                postname += '_NAMES.txt'
                input_file_path = os.path.join(subdir_name, item)
                shutil.copy(input_file_path, os.path.join(subdir_name, postname))
                # typically we want to remove the file from the old path
                # however, we comment this out in the event the copy fails
                # os.remove(os.path.join(dir_path, item))


def preprocess_dataframe(dataframe: pd.DataFrame, columns: int):
    dataframe: pd.DataFrame = dataframe.iloc[:1, :]
    vars_to_encode: list = []
    for i in range(0, columns):
        var = dataframe.iloc[0, i]
        if 'int' in str(type(var)) or 'float' in str(type(var)):
            pass
        else:
            vars_to_encode.append(i)
    return vars_to_encode


def cast_column_to_float(dataframe: pd.DataFrame):
    dataframe: pd.DataFrame = dataframe.iloc[:, :]
    vars_to_encode: list = []
    count = 0
    for row in dataframe.iterrows():
        for j in range(0, len(dataframe.columns)):
            val = dataframe[j][count]
            if val == '?':
                dataframe[j][count] = 1
        count += 1

    # length = len(dataframe.columns)
    # rows = len(dataframe)
    # for i in range(0, rows):
    #     for j in range(0, length):
    #         val = dataframe[j][i]
    #         if val == '?':
    #             val = 1
    #             dataframe[j][i] = val

    return dataframe.reset_index(drop=True)

