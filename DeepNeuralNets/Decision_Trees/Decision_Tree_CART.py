# Decision_Tree_CART.py
# Kordel France
########################################################################################################################
# This file establishes the specification and driver for a CART decision tree model.
########################################################################################################################

import numpy as np
import math
from Lab4 import config
from Lab4.Decision_Trees import Rule as Rule
from Lab4.Decision_Trees import Rule_Node as Rule_Node
from Lab4.Decision_Trees import Leaf_Node as Leaf_Node


# define global parameters used to track metrics of tree performance and pruning effectiveness
FINAL_NODES: list = []
INFO_GAINS: list = []
MIN_INFO_GAIN: float = math.inf
TREE_STRING: str = ''

"""
Class definition of the CART Decision Tree with all helper methods
"""
class Decision_Tree_CART:

    def category_counts(self, items):
        """
        Defines a function that counts and returns the number of categories present within the tree.
        :param items: list - the list of values to count and return the categories for.
        :return num_categories: int - the number of categories represented by the list of items.
        """
        num_categories = {}

        # count the number of unique categories
        for item in items:
            label = item[-1]
            if label not in num_categories:
                num_categories[label] = 0
            num_categories[label] += 1
        return num_categories


    def find_split(self, items, rule):
        """
        Finds the split point as a function of the MSE values based on true and false values.
        :param items: list - the list of objects to split.
        :param rule: Rule - the rule at the split point.
        :return items_true: list - the left branch as indicated by the split.
        :return items_false: list - the right branch as indicated by the split.
        """
        items_true: list = []
        items_false: list = []

        # define left and right sub-trees based on the partition point
        for item in items:
            if rule.match_rule(item):
                items_true.append(item)
            else:
                items_false.append(item)
        return items_true, items_false


    def calculate_purity(self, items):
        """
        Calculates the impurity at a specifc node given a specific decision path or subtree.
        :param items: list - the list of objects to calculate the impurity score over.
        :return purity: float - a value indicating the purity of the tree up to a specific point.
        """
        count = self.category_counts(items)
        purity: float = 1

        # calculate the cumulative purity through the summation of individual purities
        for label in count:
            p_purity: float = count[label] / float(len(items))
            p_purity **= 2
            purity -= p_purity
        return purity


    def calculate_information_gain(self, left_tree, right_tree, purity):
        """
        Calculates the information gain (or entropy) at a specifc node by computing purities of left and right subtrees.
        :param left_tree: list - a list of items that construct the left subtree
        :param right_tree: list - a list o items that construct the right subtree
        :return info_gain float - the entropy or info gain calculated from both trees
        """
        prob: float = float(len(left_tree)) / (len(left_tree) + len(right_tree))
        info_gain: float = purity - prob * self.calculate_purity(left_tree) - (1 - prob) * self.calculate_purity(right_tree)
        return info_gain


    def calculate_optimal_split(self, items):
        """
        Calculates the next optimal partition point (the next split) of the tree.
        :param items: list - the dataframe of items to calculate the next partition point over
        :return gain_optimal: float - the gain calculated for the new partition point.
        :return rule_optimal: Rule - the new rule node that gives the maximum info gain at the partition point.
        """
        gain_optimal: float = 0
        rule_optimal = None
        purity: float = self.calculate_purity(items)
        num_extracted_features: int = len(items[0]) - 1

        # we need to look at each extracted feature to define new rules
        for feature in range(num_extracted_features):
            values = set([item[feature] for item in items])

            # define new rules for feature values
            for value in values:
                rule: Rule = Rule.Rule(feature, value)
                items_results = self.find_split(items, rule)
                items_true: list = items_results[0]
                items_false: list = items_results[1]

                if len(items_true) == 0 or len(items_false) == 0:
                    continue

                # calculate the local gain for this specific rule
                gain_local = self.calculate_information_gain(items_true, items_false, purity)

                # if the local gain is a global maximum, use rule defined above as official rule for extracted feature
                if gain_local >= gain_optimal:
                    gain_optimal, rule_optimal = gain_local, rule

        # return the rule that gives the maximum info gain and the info gain for the extracted feature
        return gain_optimal, rule_optimal


    def construct_decision_tree(self, items, threshold: float):
        """
        Takes the raw dataframe and constructs a CART decision tree by recursively definint rules until data runs out.
        :param items: list - a list of data extracted from the dataframe to construct the dataframe with.
        :param threshold: float - if != 0.0, the threshold defines when early stopping shall occur in tree construction.
        :return Leaf_Node - if no more features can be extracted for a given subtree, a leaf node is returned.
        :return Rule_Node - if more features can be extracted from data, then another rule node is defined and returned.
        """
        # globl counters used for tracking tree growth through pruning
        global FINAL_NODES
        global INFO_GAINS
        FINAL_NODES.append(items)

        # calculate the info gain and next optimal rule for the dataset
        info_gain, rule = self.calculate_optimal_split(items)
        if info_gain > 0.0:
            INFO_GAINS.append(info_gain)

        # no more features can be extracted, define a leaf node
        if info_gain == 0 or info_gain < threshold:
            return Leaf_Node.Leaf_Node(items)

        # more feature extraction can be performed over the given dataset
        items_results = self.find_split(items, rule)
        items_true: list = items_results[0]
        items_false: list = items_results[1]

        # define the next left and right subtrees to calculate the rule node
        tree_true = self.construct_decision_tree(items_true, threshold)
        tree_false = self.construct_decision_tree(items_false, threshold)
        return Rule_Node.Rule_Node(rule, tree_true, tree_false)


    def fit_data(self, data, truth_labels, threshold: float):
        """
        Small helper function that restructures the data by concatenating columns as a horizontal sequence.
        This helps for being able to represent the tree as a large list without losing track of index values.
        :param data: list - the dataset to fit the decision tree to represented as a list.
        :param truth_labels - ground truth labels used to train the decision tree with.
        :param threshold - if != 0.0, this value defines the stopping value for tree pruning.
        """
        # restructure dataset as a horizontal sequence
        items = np.hstack(
            (data, truth_labels[:, None]))

        # start at the root and recursively define the rules until all leaf nodes are defined
        self.root = self.construct_decision_tree(items, threshold)


    def predict_next_node(self, data, start_node, state: int = 0):
        """
        Predicts the next node (the y-hat) in a tree given preceding data.
        :param data: list - the explanatory data used to make the prediction.
        :param start_node: Leaf_Node or Rule_Node - the node to start the prediction at.
        :param state: int - debugging heuristic that helps to determine leaf node or rule node
        :return y-hat: Any - the value predicted by the tree.
        """

        # check if the starting node is a leaf node and return the y-hat specified by the leaf node
        if isinstance(start_node, Leaf_Node.Leaf_Node):
            if state == 0:
                if config.DEMO_MODE:
                    print(f'y-hat prediction: {start_node.y_hats}')
                return start_node.y_hats
            else:
                mode_predictions = max(start_node.y_hats, key=start_node.y_hats.get)
                if config.DEMO_MODE:
                    print(f'y-hat prediction: {mode_predictions}')
                return mode_predictions

        # otherwise recursively call this function until a leaf node is reached and return predicted value
        if start_node.rule.match_rule(data):
            y_hat = self.predict_next_node(data, start_node.tree_true, state)
            if config.DEMO_MODE:
                print(f'rule: {start_node.rule}')
            return y_hat
        else:
            y_hat = self.predict_next_node(data, start_node.tree_false, state)
            return y_hat


    def predict_label(self, data, state: int = 0):
        """
        Stores the labels predicted by the validation set to check against ground truth labels.
        :param data: list - the explanatory data used to make the predictions.
        :param state: int - debugging heuristic that helps to determine leaf node or rule node
        :return predicted_nodes: list - the full list of y-hats predicted by the tree.
        """
        # recursively call this function until all leaf nodes are reached and a full list of predicted nodes is built
        if data.ndim == 1:
            return self.predict_next_node(data, self.root, state)
        else:
            predicted_nodes: list = []
            for item in data:
                predicted_node = self.predict_next_node(item, self.root, state)
                predicted_nodes.append(predicted_node)
            return predicted_nodes


    def calculate_confidence_score(self, data, truth_labels, state: int = 0):
        """
        Calculates a confidence value indicating how much we can trust the y-hat predictions made by the model.
        :param data: list - the explanatory data used to make the predictions.
        :param truth_labels - the ground truth labels to validate the predicted data against.
        :param state: int - debugging heuristic that helps to determine leaf node or rule node
        :return confidence_score: float - a floating point value between 0 and 1.
        """
        # all leaf nodes have been reached, sum all confidence scores for this sub-tree
        if state == 0:
            accuracy: float = 0
            y_hat_dict = self.predict_label(data)
            count = len(truth_labels)

            for index_i in range(count):
                test_total = sum(y_hat_dict[index_i].values()) * 1.0

                for label in y_hat_dict[index_i].keys():
                    if label == truth_labels[index_i]:
                        accuracy += y_hat_dict[index_i][label] / test_total
            confidence_score: float = float(accuracy / count)
            return confidence_score

        # otherwise calculate cumulative confidence score from the rule node up
        else:
            true_predictions: float = 0
            y_hat = self.predict_label(data, state = 1)
            truth_count = len(truth_labels)

            for index_i in range(truth_count):
                if y_hat[index_i] == truth_labels[index_i]:
                    true_predictions += 1
            confidence_score: float = float(true_predictions / truth_count)
            return confidence_score


    def display_decision_tree(self, start_node, gap: str = ' '):
        """
        Compiles the constructed decision tree into a single string so it is cleanly interpretable by the console.
        :param start_node: Leaf_Node or Rule_Node - the node to start the prediction at.
        :param gap: str - a gap of whitespace that helps space nodes out according to their position in the hierarchy.
        """
        global TREE_STRING
        if isinstance(start_node, Leaf_Node.Leaf_Node):
            line_print: str = f'{gap} y-hat estimate'
            # print(line_print, start_node.y_hats)
            TREE_STRING += f'\n{line_print} {start_node.y_hats}'
            return

        rule_line: str = f'{gap}{start_node.rule}'
        # print(rule_line)
        TREE_STRING += f'\n{rule_line}'
        true_line: str = f'{gap}\____True'
        # print(true_line)
        TREE_STRING += f'\n{true_line}'
        self.display_decision_tree(start_node.tree_true, gap + '  ')
        false_line: str = f'{gap}\____False'
        # print(false_line)
        TREE_STRING += f'\n{false_line}'
        self.display_decision_tree(start_node.tree_false, gap + '  ')


    def print_decision_tree_to_console(self, gap:str = ' '):
        """
        Compiles the constructed decision tree into a single string so it is cleanly interpretable by the console.
        :param gap: str - a gap of whitespace that helps space nodes out according to their position in the hierarchy.
        """
        self.display_decision_tree(self.root, gap)


def reset_metrics():
    """
    Resets all metrics used by the tree so they can be used for the next tree.
    """
    global FINAL_NODES
    global INFO_GAINS
    global MIN_INFO_GAIN
    global TREE_STRING
    FINAL_NODES = []
    INFO_GAINS = []
    MIN_INFO_GAIN = 0.0
    TREE_STRING = ''


def train_CART_decision_tree(training_data, training_labels, should_print: bool, threshold: float = 0.0):
    """
    This is the driver function for training the CART decision tree.
    Coordinates the data pipeline, starts the training and builds the tree string.
    :param training_data: pd.DataFrame - the data to train the decision tree on.
    :param training_labels: pd.DataFrame - the ground truth training labels for the training data.
    :param should_print: bool - debugging parameter indicating whether the tree should be printed after construction.
    :param threshold - if != 0.0, this value defines the stopping value for tree pruning.
    :return decision_tree_cart: Decision_Tree_CART - a fully constructed CART decision tree model.
    """
    decision_tree_cart = Decision_Tree_CART()
    decision_tree_cart.fit_data(training_data, training_labels, threshold)
    # INFO_GAINS.sort()
    if should_print:
        decision_tree_cart.print_decision_tree_to_console()
    return decision_tree_cart


def calculate_metrics(testing_data, testing_labels, decision_tree):
    """
    Calculates the final MSE after the tree is constructed, trained and evaluated.
    :param testing_data: pd.DataFrame - the data to test the decision tree on.
    :param testing_labels: pd.DataFrame - the ground truth testing labels for the training data.
    :param decision_tree: Decision_Tree_CART - a fully constructed CART decision tree model.
    :param mse: float - the mean squared error of the final model.
    """
    return decision_tree.calculate_confidence_score(testing_data, testing_labels)



"""
MARK:   -   FOR DEBUGGING ONLY
These methods called scipy functions to compare the accuracy of my custom tree to the scipy package tree.
It effectively performed quality control on my tree to ensure accuracy was not too far off and mitigate inductive bias.
"""

if config.DEMO_MODE:
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import pandas as pd
    from datetime import datetime

    def evaluate_decision_tree(testing_data, testing_labels, decision_tree):
        print(metrics.classification_report(testing_labels, decision_tree.predict_label(testing_data, state = 1)))
        print(metrics.confusion_matrix(testing_labels, decision_tree.predict_label(testing_data, state = 1)))


    def import_dataset(filepath: str, column: int):
        input_data: pd.DataFrame = pd.read_csv(filepath, header=None)
        input_data_0 = np.array(input_data.drop([column], 1))
        labels = np.array(input_data[column])
        training_data, testing_data, training_labels, testing_labels = train_test_split(input_data_0, labels, test_size=0.3, random_state=87)
        return training_data, training_labels, testing_data, testing_labels


    def construct_decision_tree_and_evaluate(filename: str, column: int, threshold: float):
        reset_metrics()
        training_data, training_labels, testing_data, testing_labels = import_dataset(filename, column)
        start_time = datetime.now()
        decision_tree_cart = train_CART_decision_tree(training_data, training_labels, True, threshold)
        print(f'training time: ', str(datetime.now() - start_time))
        print('testing accuracy: ', float(calculate_metrics(testing_data, testing_labels, decision_tree_cart)))
        print(f'min info gain: {min(INFO_GAINS)}')
        print(f'length of decision tree: {len(FINAL_NODES)}')
        evaluate_decision_tree(testing_data, testing_labels, decision_tree_cart)


    if __name__ == '__main__':
        construct_decision_tree_and_evaluate('io_files/machine(1)_DATA.csv', 9, threshold=0)
