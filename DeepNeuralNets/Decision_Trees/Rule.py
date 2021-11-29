# Rule.py
# Kordel France
########################################################################################################################
########################################################################################################################
# This file establishes a class for a rule / feature statement in a decision tree.
########################################################################################################################



class Rule:
    def __init__(self, y_val, x_val):
        """
        Object class that represents a rule or extracted feature definition in a decision tree.
        :param x_val: Any - the explanatory value for the rule node.
        :param y_val: Any - the response value for the rule node.
        """
        self.x_val = x_val
        self.y_val = y_val


    def match_rule(self, example):
        """
        Function that determines whether this rule matches another specified rule.
        :param example: Rule - the rule to match this rule to.
        :return value: bool - indicates whether the two rules are equivalent.
        """
        value = example[self.y_val]
        return value >= self.x_val


    def __repr__(self):
        """
        Function that defines a custom string representation of the Rule that is easier to interpret in the console.
        :return rule_description: str - a string that describes the rule and its place within the tree structure..
        """
        return 'feature[%s] >= %s' % (self.y_val, self.x_val)

