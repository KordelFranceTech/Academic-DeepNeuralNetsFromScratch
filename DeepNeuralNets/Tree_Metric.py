# Metric_X.py
# Kordel France
########################################################################################################################
# This file establishes a class for a metric used to track properties of a decision tree as it is built and pruned.
########################################################################################################################


import pandas as pd

class Tree_Metric():

    def __init__(self,
                 name: str,
                 is_classify: bool,
                 response_col: int,
                 pruning_factor: float,
                 accuracy_pre_prune: float,
                 accuracy_post_prune: float,
                 accuracy_list: list,
                 mse_pre_prune: float,
                 mse_post_prune: float,
                 mse_list: list,
                 nodes_pre_prune: int,
                 nodes_post_prune: int,
                 lowest_gains: list,
                 dataframe: pd.DataFrame,
                 tree: str):
        """
        Object class that defines properties used to monitor and track a tree as it is built and pruned
        :param name: str - the name of the dataset used
        :param is_classify: str - indicates whether the tree is a classification or regression tree
        :param response_col: int - the index of the response column
        :param pruning_factor: float - the pruning value (alpha) used to prune the tree
        :param accuracy_pre_prune: float - the classification accuracy (if applicable) before pruning
        :param accuracy_post_prune: float - the classification accuracy (if applicable) after pruning
        :param accuracy_list: list - the list of accuracies received from pre-prune KFCV
        :param mse_pre_prune: float - the regression mse (if applicable) before pruning
        :param mse_post_prune: float - the regression mse (if applicable) after pruning
        :param mse_list: list - the list of mses received from pre-prune KFCV
        :param nodes_pre_prune: int - the number of nodes in the tree before pruning
        :param nodes_post_prune: int - the number of nodes in the tree after pruning
        :param lowest_gains: list - the lowest gain received from each KFCV fold used to deduce pruning factor
        :param dataframe: pd.DataFrame - the dataframe used to construct the treee
        :param tree: str - the final pruned tree represented as a list
        """
        self.name = name
        self.is_classify = is_classify
        self.response_col = response_col
        self.pruning_factor = pruning_factor
        self.accuracy_pre_prune = accuracy_pre_prune
        self.accuracy_post_prune = accuracy_post_prune
        self.accuracy_list = accuracy_list
        self.mse_pre_prune = mse_pre_prune
        self.mse_post_prune = mse_post_prune
        self.mse_list = mse_list
        self.nodes_pre_prune = nodes_pre_prune
        self.nodes_post_prune = nodes_post_prune
        self.lowest_gains = lowest_gains
        self.dataframe = dataframe
        self.tree = tree


    def print_metric(self, should_print: bool = True):
        """
        Function that structures the parameters of the Metric and organizes it into a form easy to interpret in console
        :param should_print: bool - indicates whether or not the metric should be printed or just written to file
        :return output_string: str - structured string that represents all details about the decision tree as it evolved
        """
        output_string: str = '\n_________________________________________________________________'
        output_string += '\n_________________________________________________________________'
        output_string += f'\nname: {self.name}'
        if self.is_classify:
            output_string += f'\n\talgorithm: ID3 Decision Tree, Classification'
        else:
            output_string += f'\n\talgorithm: CART Decision Tree, Regression'
        output_string += f'\n\tresponse column: {self.response_col}'
        output_string += f'\n\tlowest gains: {self.lowest_gains}'
        output_string += f'\n\tpruning factor: {self.pruning_factor}'
        output_string += f'\n\t# of nodes before pruning: {self.nodes_pre_prune}'
        output_string += f'\n\t# of nodes after pruning: {self.nodes_post_prune}'
        if self.is_classify:
            output_string += f'\n\taccuracy pre-pruned: {self.accuracy_pre_prune}'
            output_string += f'\n\taccuracy post-pruned: {self.accuracy_post_prune}'
            output_string += f'\n\tKFCV accuracy list: {self.accuracy_list}'
        else:
            output_string += f'\n\tmse pre-pruned: {self.mse_pre_prune}'
            output_string += f'\n\tmse post-pruned: {self.mse_post_prune}'
            output_string += f'\n\tKFCV mse list: {self.mse_list}'
        # output_string += '\n_________________________________________________________________'
        # output_string += f'\n{self.tree}'
        if should_print:
            print(output_string)
        return output_string

