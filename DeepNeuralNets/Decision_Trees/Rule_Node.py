# Rule_Node.py
# Kordel France
########################################################################################################################
# This file establishes a class for non-leaf node (a rule, a feature statement) in a decision tree.
########################################################################################################################


class Rule_Node:
    def __init__(self, rule, tree_true, tree_false):
        """
        Object class that represents a rule as a node within a decision tree.
        :param rule: Rule - the rule / feature statement that the node represents.
        :param tree_true: list - a list of values specified as a tree indicating only the values classified as true.
        :param tree_false: list - a list of values specified as a tree indicating only the values classified as false.
        """
        self.rule = rule
        self.tree_true = tree_true
        self.tree_false = tree_false
