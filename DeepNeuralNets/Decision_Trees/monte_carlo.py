# monte_carlo.py
# Kordel France
########################################################################################################################
# This file establishes the specification for a monte carlo simulation of decision trees over multiple permutations
# of data.

import pandas as pd
import numpy as np
import math
import Lab3.helpers
from Lab4.Decision_Trees import Decision_Tree_ID3, Decision_Tree_CART
from Lab4.cross_validate import construct_data_folds, get_datasets_for_fold
from Lab4 import Tree_Metric as Metric
from Lab4 import data_processing
from Lab4 import encoding
from Lab4 import config


# definition of global parameters that monte carlo simulation will leverage
filenames: list = ['abalone(1)_DATA.csv',
                   'breast-cancer-wisconsin(1)_DATA.csv',
                   'car(1)_DATA.csv',
                   'forestfires(1)_DATA.csv',
                   'house-votes-84(1)_DATA.csv',
                   'machine(1)_DATA.csv']
# filenames: list = ['car(1)_DATA.csv',
#                    'forestfires(1)_DATA.csv',
#                    'house-votes-84(1)_DATA.csv']
type_list: list = [True, False]




def perform_k_fold_cross_validation(num_folds: int,
                                    dataframe: pd.DataFrame,
                                    is_classify: bool,
                                    explanatory_col: int,
                                    response_col: int,
                                    THRESHOLD_PRUNE: float,
                                    THRESHOLD_EARLY_STOP: float):
    """
    Driver function for k-fold cross validation.
    :param num_folds: int - the number of folds, or 'k'
    :param dataframe: pd.Dataframe - the dataframe to construct the separate dataset folds from
    :param is_classify: bool - a flag that indicates whether a classification or regression algorithm should be selected
    :param explanatory_col: int - the column indicating the explanatory variable, or x series
    :param response_col: int - the column indicating the response variable, or y series
    :param algorithm: int - a value indicating which KNN algorithm to use
    :param: k: int - the number of nearest neighbors to find
    :param: alpha: flaot - a tuning value used to train the algorithm
    :return accuracy_list: a list of floats indicating accuracies yielded on each KFCV fold of training (classification)
    :return mse_list: a list of floats indicating MSEs yielded on each KFCV fold of training (regression)
    """

    # initialize bins, construct data folds for KFCV from dataframe
    bin_list = construct_data_folds(num_folds, dataframe)

    # initialize metrics lists
    accuracy_list: list = []
    node_list: list = []
    mse_list: list = []
    gain_list: list = []
    tree_list: list = []

    # perform training and testing over each KFCV fold
    for index_i in range(0, num_folds):
        train_data, test_data = get_datasets_for_fold(index_i, bin_list, False)
        x_vals = dataframe.iloc[:, dataframe.columns != response_col].values
        y_vals = dataframe.iloc[:, response_col].values

        if config.DEBUG_MODE:
            print(f'dataframe from main: {dataframe}')
            print(f'x from main: {x_vals}')
            print(f'y from main: {y_vals}')
            print(f'lengths: x: {len(x_vals)}, y: {len(y_vals)}')
            partition_point: int = int(round(len(dataframe) * 0.8))
            x_data_train = x_vals[partition_point:, :]
            x_data_test = x_vals[:partition_point, :]
            y_data_train = y_vals[partition_point:, :]
            y_data_test = y_vals[:partition_point, :]

        # we have received the training and testing data
        # now stratify it according to x- and y- dataframes
        # x_data_train = train_data.iloc[:, train_data.columns != response_col].values
        # x_data_test = test_data.iloc[:, test_data.columns != response_col].values
        # y_data_train = train_data.iloc[:, response_col].values
        # y_data_test = test_data.iloc[:, response_col].values
        x_data_train = np.array(train_data.drop([response_col], 1))
        x_data_test = np.array(test_data.drop([response_col], 1))
        y_data_train = np.array(train_data[response_col])
        y_data_test = np.array(test_data[response_col])

        if config.DEBUG_MODE:
            print(f'dataframe:\n\t{dataframe}')
            print(f' x test data: {len(x_data_test)}\n\t:{x_data_test}')
            print(f' x train data: {len(x_data_train)}\n\t{x_data_train}')
            print(f' y test data: {len(y_data_test)}\n\t{y_data_test}')
            print(f' y train data: {len(y_data_train)}\n\t{y_data_train}')

        # if a classification algorithm is desired
        if is_classify:

            # implement ID3 algorithm
            all_labels = np.array(dataframe[response_col])
            Decision_Tree_ID3.reset_metrics()
            Decision_Tree_ID3.ALL_LABELS = all_labels
            decision_tree_id3 = Decision_Tree_ID3.train_ID3_decision_tree(x_data_train,
                                                                          y_data_train,
                                                                          True,
                                                                          THRESHOLD_PRUNE)
            tree = Decision_Tree_ID3.TREE_STRING
            tree_list.append(tree)
            accuracy: float = \
                float(Decision_Tree_ID3.calculate_metrics(x_data_test,
                                                          y_data_test,
                                                          decision_tree_id3))
            accuracy *= 100.0
            accuracy_list.append(accuracy)
            node_list.append(len(Decision_Tree_ID3.FINAL_NODES))
            if len(Decision_Tree_ID3.INFO_GAINS) > 0:
                gain_list.append(Decision_Tree_ID3.INFO_GAINS[0])
            # print('testing accuracy: ', accuracy)
            # print(f'number of nodes: {len(Decision_Tree_ID3.FINAL_NODES)}')

        # regression is desired
        else:

            # implement CART algorithm
            Decision_Tree_CART.reset_metrics()
            decision_tree_cart = Decision_Tree_CART.train_CART_decision_tree(x_data_train,
                                                                             y_data_train,
                                                                             True,
                                                                             THRESHOLD_EARLY_STOP)

            tree = Decision_Tree_CART.TREE_STRING
            tree_list.append(tree)
            mse: float = float(Decision_Tree_CART.calculate_metrics(x_data_test,
                                                                    y_data_test,
                                                                    decision_tree_cart))
            mse *= 100.0
            mse_list.append(mse)
            node_list.append(len(Decision_Tree_CART.FINAL_NODES))
            if len(Decision_Tree_CART.INFO_GAINS) > 0:
                gain_list.append(Decision_Tree_CART.INFO_GAINS[0])
            # print('testing accuracy: ', mse)
            # print(f'number of nodes: {len(Decision_Tree_CART.FINAL_NODES)}')


            Lab3.helpers.COST_COUNTER += 1

    # return classification metrics
    if len(accuracy_list) > 0:
        # print(f'\naverage accuracy for each fold:\n\t{accuracy_list}')
        return accuracy_list, mse_list, node_list, gain_list, tree_list
    # return regression metrics
    else:
        # print(f'\nmean squared errors for each fold:\n\t{mse_list}')
        return accuracy_list, mse_list, node_list, gain_list, tree_list


def find_optimal_metric(dataframe: pd.DataFrame,
                        num_folds: int,
                        is_classify: bool,
                        explanatory_col: int,
                        response_col: int,
                        algorithm: int,
                        filename: str):
    """
    Driver function for tuning alpha value.
    :param num_folds: int - the number of folds, or 'k'
    :param dataframe: pd.Dataframe - the dataframe to construct the separate dataset folds from
    :param is_classify: bool - a flag that indicates whether a classification or regression algorithm should be selected
    :param explanatory_col: int - the column indicating the explanatory variable, or x series
    :param response_col: int - the column indicating the response variable, or y series
    :param algorithm: int - a value indicating which KNN algorithm to use
    :param: k: int - the number of nearest neighbors to find
    :param: alpha: flaot - a tuning value used to train the algorithm
    :return accuracy_list: a list of floats indicating accuracies yielded on each KFCV fold of training (classification)
    :return mse_list: a list of floats indicating MSEs yielded on each KFCV fold of training (regression)
    """
    # initialize tuning metrics
    min_metric: float = math.inf
    max_metric: float = 0.0
    pruned_min_metric: float = math.inf
    pruned_max_metric: float = 0.0
    metric_list: list = []
    prune_list: list = []
    optimal_prune_metric: Metric

    # print header for aesthetics
    print('_________________________________________________________________')
    print('\n\nanalyzing all values of alpha:')
    print('_________________________________________________________________')

    # iterate over every alpha
    # perform KFCV over the data for each value of alpha
    # note the classification accuracies/regression mses, and find the optimal tuning value
    # dataframe = dataframe.iloc[0:200, :]

    # build initial unpruned decision tree
    THRESHOLD_PRUNE: float = math.inf  # 11
    THRESHOLD_EARLY_STOP: float = 0.0  # 0.1

    metric = Metric.Tree_Metric(name=filename,
                                is_classify=is_classify,
                                response_col=response_col,
                                pruning_factor=0.0,
                                accuracy_pre_prune=0.0,
                                accuracy_post_prune=0.0,
                                accuracy_list=[],
                                mse_pre_prune=0.0,
                                mse_post_prune=0.0,
                                mse_list=[],
                                nodes_pre_prune=0,
                                nodes_post_prune=0,
                                lowest_gains=[],
                                dataframe=dataframe,
                                tree='')

    accuracy_list, mse_list, node_list, gain_list, tree_list = perform_k_fold_cross_validation(num_folds=num_folds,
                                                                                               dataframe=dataframe,
                                                                                               is_classify=is_classify,
                                                                                               explanatory_col=explanatory_col,
                                                                                               response_col=response_col,
                                                                                               THRESHOLD_PRUNE=THRESHOLD_PRUNE,
                                                                                               THRESHOLD_EARLY_STOP=THRESHOLD_EARLY_STOP)

    average_acc: float = sum(accuracy_list)
    average_acc = average_acc / len(accuracy_list) if len(accuracy_list) > 0 else 0
    average_mse: float = sum(mse_list)
    average_mse = average_mse / len(mse_list) if len(mse_list) > 0 else 0
    metric.accuracy_pre_prune = average_acc
    metric.mse_pre_prune = average_mse
    metric.nodes_pre_prune = max(node_list) if len(node_list) > 0 else metric.nodes_pre_prune
    metric.lowest_gains = gain_list
    metric.accuracy_list = accuracy_list
    metric.mse_list = mse_list
    metric.tree = tree_list[0]
    metric_list.append(metric)


    # found a new local minimum MSE for regression
    if len(mse_list) > 0:
        min_metric = max(mse_list)
        prune_list.append(mse_list)

    # found a new local maximum accuracy for classification
    else:
        max_metric = max(accuracy_list)
        prune_list.append(accuracy_list)

    # return results indicating which alpha value returned the best results
    print('_________________________________________________________________')
    print('_________________________________________________________________')
    if is_classify:
        print(f'accuracy list for each fold in KFCV: {accuracy_list} %')
        print(f'the best results were returned for maximum accuracy of {round(max_metric, 0)} %')
    else:
        print(f'mse list for each fold in KFCV: {mse_list}')
        print(f'the best results were returned for minimum mse = {min_metric}')

    # nos start pruning tree and return results
    pruning_gain: float = min(gain_list) if len(gain_list) > 0 else 0.1
    pruning_list: list = []
    for index_i in range(0, 5):
        if not is_classify:
            pg: float = 0.9 + (float(index_i) / 10)
            pruning_list.append(pruning_gain / pg)
        else:
            pg: float = 0.9 + (float(index_i) / 100)
            pruning_list.append(pruning_gain * pg)

    for index_i in range(0, len(pruning_list)):
        pruned_metric = Metric.Tree_Metric(name=metric.name,
                                           is_classify=metric.is_classify,
                                           response_col=metric.response_col,
                                           pruning_factor=pruning_list[index_i],
                                           accuracy_pre_prune=max_metric,
                                           accuracy_post_prune=0.0,
                                           accuracy_list=[],
                                           mse_pre_prune=min_metric,
                                           mse_post_prune=0.0,
                                           mse_list=[],
                                           nodes_pre_prune=metric.nodes_pre_prune,
                                           nodes_post_prune=0,
                                           lowest_gains=[],
                                           dataframe=metric.dataframe,
                                           tree=metric.tree)

        pruned_accuracy_list, \
            pruned_mse_list, \
            pruned_node_list, \
            pruned_gain_list, \
            pruned_tree_list = \
            perform_k_fold_cross_validation(num_folds=num_folds,
                                            dataframe=dataframe,
                                            is_classify=is_classify,
                                            explanatory_col=explanatory_col,
                                            response_col=response_col,
                                            THRESHOLD_PRUNE=pruning_list[index_i],
                                            THRESHOLD_EARLY_STOP=pruning_list[index_i])

        average_pp_acc: float = sum(pruned_accuracy_list)
        average_pp_acc = average_pp_acc / len(pruned_accuracy_list) if len(pruned_accuracy_list) > 0 else 0
        average_pp_mse: float = sum(pruned_mse_list)
        average_pp_mse = average_pp_mse / len(pruned_mse_list) if len(pruned_mse_list) > 0 else 0
        pruned_metric.accuracy_post_prune = average_pp_acc
        pruned_metric.mse_post_prune = average_pp_mse
        pruned_metric.nodes_post_prune = min(pruned_node_list) if len(pruned_node_list) > 0 else pruned_metric.nodes_post_prune
        pruned_metric.lowest_gains = pruned_gain_list
        pruned_metric.accuracy_list = pruned_accuracy_list
        pruned_metric.mse_list = pruned_mse_list
        pruned_metric.tree = pruned_tree_list[0]
        metric_list.append(pruned_metric)

        # found a new local minimum MSE for regression
        if len(pruned_mse_list) > 0:
            pruned_min_metric -= pruned_min_metric
            pruned_min_metric += max(pruned_mse_list)
            optimal_prune_metric = pruned_metric
            prune_list.append(pruned_mse_list)
            # print(f'min mse: {pruned_min_metric}')
        # found a new local maximum accuracy for classification
        else:
            pruned_max_metric -= pruned_max_metric
            pruned_max_metric += max(pruned_accuracy_list)
            optimal_prune_metric = pruned_metric
            prune_list.append(pruned_accuracy_list)
            # print(f'max accuracy: {pruned_max_metric}')

    # return results indicating which alpha value returned the best results
    print('_________________________________________________________________')
    print('_________________________________________________________________')
    if is_classify:
        print(f'accuracy list for each fold in KFCV: {optimal_prune_metric.accuracy_list} %')
        print(f'the best results were returned for maximum accuracy of {max(optimal_prune_metric.accuracy_list)} %')
    else:
        print(f'mse list for each fold in KFCV: {optimal_prune_metric.mse_list}')
        print(f'the best results were returned for minimum mse = {max(optimal_prune_metric.mse_list)}')



    if is_classify:
        output_file = open(f'{config.IO_DIRECTORY}/output/classification_{filename[:-4]}_output_{response_col}.txt', "w")
        output_file.write(f'{filename}\n\n_________________________________________________________________')
        output_file.write(f'\nUn-Pruned Results:\n{metric.print_metric(should_print=False)}')
        output_file.write(f'\n\nUn-Pruned Tree:\n{metric.tree}')
        output_file.write(f'\n_________________________________________________________________\n')
        output_file.write(f'\nPruning results for best threshold:\n')
        output_file.write(optimal_prune_metric.print_metric(should_print=False))
        output_file.write(f'\nPruned Tree:\n{optimal_prune_metric.tree}')
        output_file.write(f'\n_________________________________________________________________\n')
        output_file.write(f'\nPruning results for other observed thresholds:\n')

        for index_i in range(1, len(metric_list)):
            m: Metric = metric_list[index_i]
            output_file.write(m.print_metric())

        output_file.write(f'\n\n{str(prune_list)}')
        output_file.close()

        return is_classify, optimal_prune_metric
    else:
        output_file = open(f'{config.IO_DIRECTORY}/output/regression_{filename[:-4]}_output_{response_col}.txt', "w")
        output_file.write(f'{filename}\n\n_________________________________________________________________')
        output_file.write(f'\nUn-Pruned Results:\n{metric.print_metric(should_print=False)}')
        output_file.write(f'\n\nUn-Pruned Tree:\n{metric.tree}')
        output_file.write(f'\n_________________________________________________________________\n')
        output_file.write(f'\nPruning results for best threshold:\n')
        output_file.write(optimal_prune_metric.print_metric(should_print=False))
        output_file.write(f'\nPruned Tree:\n{optimal_prune_metric.tree}')
        output_file.write(f'\n_________________________________________________________________\n')
        output_file.write(f'\nPruning results for other observed thresholds:\n')

        for index_i in range(1, len(metric_list)):
            m: Metric = metric_list[index_i]
            output_file.write(m.print_metric())

        output_file.write(f'\n\n{str(prune_list)}')
        output_file.close()

        return is_classify, optimal_prune_metric


def run_monte_carlo_simulation():
    """
    Performs a monte-carlo style series of trials over each dataset for different values of k, sigma, alpha and
    over different algorithms.
    Different response variables are also tested.
    This was designed for efficacy, not efficiency - runs at O(n^5) time complexity.
    """
    # initialize output file
    output_file = open('Lab3/io_files/output/output.txt', 'w')
    # output_file.close()
    output_file.write('output for program\n\n')
    output_file = open('Lab3/io_files/output/output.txt', 'a')

    _filenames: list = []
    _algorithms: list = []
    _ks: list = []
    _response_cols: list = []
    _metrics: list = []
    _alphas: list = []
    _metric_lists: list = []

    for filename in filenames:
        print(f'filename: {filename}')
        df = data_processing.import_dataset(filename)
        encoding_options: list = data_processing.preprocess_dataframe(df, df.shape[1])
        dataframe = encoding.encode_nominal_data(df, encoding_options)
        # dataframe = df
        # dataframe = dataframe.iloc[0:, :]
        dataframe = dataframe.iloc[0:50, :]
        max_columns: int = len(dataframe.columns)

        for index_k in range(2):

            for index_i in range(0, max_columns):

                # if index_i not in encoding_options:

                    a, b = find_optimal_metric(dataframe=dataframe,
                                               num_folds=5,
                                               is_classify=type_list[index_k],
                                               explanatory_col=0,
                                               response_col=index_i,
                                               algorithm=index_k,
                                               filename=filename)
                    if a:
                        output_string: str = f'\n_________________________________________________________________'
                        output_string += f'\nfilename: {filename[:-12]}'
                        output_string += f'\n\talgorithm: ID3'
                        output_string += f'\n\tresponse column: {dataframe.index[index_i]}'
                        output_string += f'\n\tclassification\tpre-prune accuracy: {b.accuracy_pre_prune}\tpost-prune accuracy: {b.accuracy_post_prune}'
                        output_string += f'\n\tmetric list: {b.accuracy_list}'
                        output_file.write(output_string)
                        metric_list = b.accuracy_list
                    else:
                        output_string: str = f'\n_________________________________________________________________'
                        output_string += f'\nfilename: {filename[:-12]}'
                        output_string += f'\n\talgorithm: CART'
                        output_string += f'\n\tresponse column: {dataframe.index[index_i]}'
                        output_string += f'\n\tregression\tpre-prune accuracy: {b.mse_pre_prune}\tpost-prune accuracy: {b.mse_post_prune}'
                        output_string += f'\n\tmetric list: {b.mse_list}'
                        output_file.write(output_string)
                        metric_list = b.mse_list
    output_file.close()


if __name__ == '__main__':
    run_monte_carlo_simulation()

