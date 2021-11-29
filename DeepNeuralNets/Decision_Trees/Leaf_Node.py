# Leaf_Node.py
# Kordel France
########################################################################################################################
# This file establishes a class for a leaf node (a result) in a decision tree.
########################################################################################################################


class Leaf_Node:
    def __init__(self, items):
        """
        Object class that represents a leaf node in a decision tree.
        :param y_hats: list - a list of values predicted by the decision tree.
        """
        self.y_hats = self.category_counts(items)


    def category_counts(self, items):
        """
        Defines a function that counts and returns the number of categories present within the leaf node.
        :param items: list - the list of values to count and return the categories for.
        :return num_categories: int - the number of categories represented by the list of items.
        """
        num_categories = {}
        for item in items:
            label = item[-1]
            if label not in num_categories:
                num_categories[label] = 0
            num_categories[label] += 1
        return num_categories



