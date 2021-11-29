# cross_validate.py
# Kordel France
########################################################################################################################
# This file establishes the driver and helper functions used for the facilitation of K-fold cross validation.
########################################################################################################################


import pandas as pd
import Lab4.evaluation as evaluation
import Lab4.helpers
from Lab4.config import DEBUG_MODE
from Lab4.learning_algorithm import select_learning_algorithm
from Lab4.Metric import Metric


def perform_k_fold_cross_validation(num_folds: int,
                                    dataframe: pd.DataFrame,
                                    is_classify: bool,
                                    explanatory_col: int,
                                    response_col: int,
                                    regression_option: int,
                                    needs_val_set: bool):
    """
    Driver function for k-fold cross validation.
    :param num_folds: int - the number of folds, or 'k'
    :param dataframe: pd.Dataframe - the dataframe to construct the separate dataset folds from
    :param is_classify: bool - a flag that indicates whether a classification or regression algorithm should be selected
    :param explanatory_col: int - the column indicating the explanatory variable, or x series
    :param response_col: int - the column indicating the response variable, or y series
    :param regression_option: int - the value indicating which regression algorithm to perform
    :param: needs_val_set: bool - an option to construct a third validation set on top of testing and training sets
    :return final_metric: Metric - an object containing statistical metrics and performance info
    """
    bin_list = construct_data_folds(num_folds, dataframe)
    metrics: [Metric] = []
    for index_i in range(0, num_folds):
        train_data, test_data, val_data = get_datasets_for_fold(index_i, bin_list, needs_val_set)
        metric_train = select_learning_algorithm(train_data=train_data,
                                                 is_classify=is_classify,
                                                 explanatory_col=explanatory_col,
                                                 response_col=response_col,
                                                 regression_option=regression_option)
        metric_test = evaluation.get_testing_evaluation(dataframe=test_data, metric=metric_train)
        if needs_val_set:
            metric_val = evaluation.get_validation_evaluation(dataframe=val_data, metric=metric_test)
            metrics.append(metric_val)
        else:
            metrics.append(metric_test)
        Lab2.helpers.COST_COUNTER += 1

    if regression_option == -1:
        init_metric: Metric = metrics[0]
        max_class_value: str = init_metric.class_value_test
        max_class_score: float = init_metric.class_score_test
        f1_score_bar: float = init_metric.f1_score_test
        for m in metrics:
            f1_score_bar += m.f1_score_test
            if m.class_score_test > max_class_score:
                max_class_score = m.class_score_test
                max_class_value = m.class_value_test
            Lab2.helpers.COST_COUNTER += 1

        f1_score_bar /= num_folds
        final_metric: Metric = metrics[0]
        final_metric.class_score_test = max_class_score
        final_metric.class_value_test = max_class_value
        final_metric.f1_score_test = f1_score_bar

        return final_metric

    else:
        init_metric: Metric = metrics[0]
        coeff0_bar: float = init_metric.coeff0
        coeff1_bar: float = init_metric.coeff1
        mse_bar: float = init_metric.mse_test
        r_bar: float = init_metric.correlation_test
        for m in metrics:
            coeff0_bar += m.coeff0
            coeff1_bar += m.coeff1
            mse_bar += m.mse_test
            r_bar += m.correlation_test
            Lab2.helpers.COST_COUNTER += 1

        coeff0_bar /= num_folds
        coeff1_bar /= num_folds
        mse_bar /= num_folds
        r_bar /= num_folds

        final_metric: Metric = metrics[0]
        final_metric.coeff0 = coeff0_bar
        final_metric.coeff1 = coeff1_bar
        final_metric.mse_test = mse_bar
        final_metric.correlation_test = r_bar

        return final_metric


def construct_data_folds(num_folds: int, dataframe: pd.DataFrame):
    """
    Parses the data, constructs the folds, and prepares the data to be separated into test, train, and val sets.
    :param num_folds: int - the number of folds, or 'k'
    :param dataframe: pd.Dataframe - the dataframe to construct the separate dataset folds from
    :return bin_list: list - the list of bins to stratify into respective datasets
    """
    # set the upper bound for our loops
    n = len(dataframe)
    if DEBUG_MODE:
        print(f'\n\tlength: {n}')
        print(f'\tperforming k-fold cross validation with {num_folds}')
        print(f'\tdataframe head: {dataframe.head(5)}')

    # a bit of error checking
    # we can't construct datasets for a number of folds less than our dataset size
    if n <= num_folds:
        print('ERROR - length of data is less than k folds')
        return [[]]
    # otherwise, continue with KFCV
    else:
        bin_size = int(n / num_folds)

        # cast the dataframe into a list for easier index use
        dataframe = dataframe.values.tolist()
        # initialize the list of bins that will be used arrange over the fold number, k
        bin_list = []
        # iterate over the number of folds to create the bin list
        for index_i in range(0, num_folds):
            cur_bin_list = []
            for index_j in range(0, bin_size):
                cur_bin_list.append(dataframe.pop())
                Lab4.helpers.COST_COUNTER += 1
            bin_list.append(cur_bin_list)

        # more error handling
        # if the bin size does not evenly divide into n, there will be a remainder of values
        # these values need a home, so evenly distribute them over bins until the dataframe is empty
        if len(dataframe) > 0:
            for index_i in range(0, len(dataframe)):
                bin_list[index_i].append(dataframe[index_i])
                Lab4.helpers.COST_COUNTER += 1

        if DEBUG_MODE:
            for bin in bin_list:
                print(f'\tbin:\n{bin}')

        # the list of binned data that will facilitate stratification of data into respective datasets
        return bin_list


def get_datasets_for_fold(fold: int, bin_list: list, needs_val_set: bool):
    """
    Given a dataset, this algorithm constructs one bin of test data, one bin of val data, and the
    remaining bins as training data.
    :param fold: int - the index of the bin to fold over
    :param bin_list: list - the list of bins to stratify into respective datasets
    :param needs_val_set: bool - an option to construct a third validation set on top of testing and training sets
    :return train_df: pd.DataFrame - the training dataset
    :return test_df: pd.DataFrame - the testing dataset
    :return val_df: pd.DataFrame - if applicable, the validation dataset
    """
    # initialize the datasets
    train_data: list = []
    test_data: list = []
    val_data: list = []
    # iterate over the folds and designate the bins of data to the appropriate datasets
    for index_i in range(0, len(bin_list)):
        row = bin_list[index_i]
        for index_j in range(0, len(row)):
            if index_i == fold:
                test_data.append(row[index_j])
            elif needs_val_set and len(val_data) == 0:
                val_data.append(row[index_j])
            else:
                train_data.append(row[index_j])
            Lab4.helpers.COST_COUNTER += 1

    # the datasets are lists, cast them as pd.DataFrames
    train_df: pd.DataFrame = pd.DataFrame(train_data)
    test_df: pd.DataFrame = pd.DataFrame(test_data)
    val_df: pd.DataFrame = pd.DataFrame(val_data)

    if DEBUG_MODE:
        print(f'train data:\n\t{train_df.head(3)}')
        print(f'test data:\n\t{test_df.head(3)}')
        print(f'val data:\n\t{val_df.head(3)}')

    # return the newly constructed datasets
    return train_df, test_df










