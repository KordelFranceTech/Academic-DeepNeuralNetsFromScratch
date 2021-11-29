# Metric.py
# Kordel France
########################################################################################################################
# This file establishes a class for an object called Metric, which can be used to store metrics of different algorithms
# computed simultaneously in a pipeline.
########################################################################################################################


class Metric():
    def __init__(self,
                 explanatory_col: int,
                 response_col: int,
                 is_regression: bool,
                 regresion_option: int,
                 equation: str,
                 coeff0: float,
                 coeff1: float,
                 correlation_test: float,
                 mse_test: float,
                 class_score_test: float,
                 class_value_test: str,
                 f1_score_test: float,
                 correlation_train: float,
                 mse_train: float,
                 class_score_train: float,
                 class_value_train: str,
                 f1_score_train: float,
                 correlation_val: float,
                 mse_val: float,
                 class_score_val: float,
                 class_value_val: str,
                 f1_score_val: float):
        """
        Object class that represents a metric for a performed algorithm over a dataset.
        :param explanatory_col: int - the column indicating the explanatory variable, or x series
        :param response_col: int - the column indicating the response variable, or y series
        :param equation: str - a string representing the regression equation
        :param is_regression: bool - a boolean heuristic that lets us know if this Metric is for regression or classification
        :param regression_option: int - the value indicating which regression algorithm to perform
        :param coeff0: float - the first coefficient in the regression equation (in linear regression, this value is slope m)
        :param coeff1: float - the second coefficient in the regression equation (in linear regression, this value is intercept b)
        :param r_test: float - the correlation value between the equation and the data
        :param mse_test: float - the mean squared error of the test dataset
        :param class_score_test: float - the classification score returned in classification tasks
        :param class_value_test: str - the classification value returned by the majority predictor algorithm
        :param f1_score_test: float - the f1 score of the resolution
        :param r_train: float - the correlation value between the equation and the data
        :param mse_train: float - the mean squared error of the test dataset
        :param class_score_train: float - the classification score returned in classification tasks
        :param class_value_train: str - the classification value returned by the majority predictor algorithm
        :param f1_score_train: float - the f1 score of the resolution
        :param r_val: float - the correlation value between the equation and the data
        :param mse_val: float - the mean squared error of the test dataset
        :param class_score_val: float - the classification score returned in classification tasks
        :param class_value_val: str - the classification value returned by the majority predictor algorithm
        :param f1_score_val: float - the f1 score of the resolution
        """
        self.explanatory_col = explanatory_col
        self.response_col = response_col
        self.is_regression = is_regression
        self.regression_option = regresion_option
        self.equation = equation
        self.coeff0 = coeff0
        self.coeff1 = coeff1
        self.correlation_test = correlation_test
        self.mse_test = mse_test
        self.class_score_test = class_score_test
        self.class_value_test = class_value_test
        self.f1_score_test = f1_score_test
        self.correlation_train = correlation_train
        self.mse_train = mse_train
        self.class_score_train = class_score_train
        self.class_value_train = class_value_train
        self.f1_score_train = f1_score_train
        self.correlation_val = correlation_val
        self.mse_val = mse_val
        self.class_score_val = class_score_val
        self.class_value_val = class_value_val
        self.f1_score_val = f1_score_val


    def print_metric(self):
        """
        Prints or writes the performance metric of an algorithm in an aesthetic manner.
        :returns output_string: str - a duplicate of the output to the console displaying metrics
        """
        output_string: str = ''
        print('_________________________________________________________________')
        output_string += '_________________________________________________________________\n'

        if self.is_regression:
            if self.correlation_train != 0:
                print('TRAINING_________________________________________________')
                print(f'\tregression equation: {self.equation} achieved the following results over the training data:')
                print(f'\t\tcorrelation: {format(self.correlation_train, ".3f")} %')
                print(f'\t\tmean squared error: {format(self.mse_train, ".3f")}')

                output_string += ('TRAINING_________________________________________________\n')
                output_string += (f'\tregression equation: {self.equation} achieved the following results over the training data:\n')
                output_string += (f'\t\tcorrelation: {format(self.correlation_train, ".3f")} %\n')
                output_string += (f'\t\tmean squared error: {format(self.mse_train, ".3f")}\n')

            if self.correlation_test != 0:
                self.equation = f'y = {self.coeff0}x + {self.coeff1}'
                print(f'\nTESTING_________________________________________________')
                print(f'\tregression equation: {self.equation} achieved the following results over the testing data:')
                print(f'\t\tcorrelation: {format(self.correlation_test, ".3f")} %')
                print(f'\t\tmean squared error: {format(self.mse_test, ".3f")}')

                output_string += (f'\nTESTING_________________________________________________\n')
                output_string += (f'\tregression equation: {self.equation} achieved the following results over the testing data:\n')
                output_string += (f'\t\tcorrelation: {format(self.correlation_test, ".3f")} %\n')
                output_string += (f'\t\tmean squared error: {format(self.mse_test, ".3f")}\n')

            if self.correlation_val != 0:
                print('\nVALIDATION_________________________________________________')
                print(f'\tregression equation: {self.equation} achieved the following results over the validation data:')
                print(f'\t\tcorrelation: {format(self.correlation_val, ".3f")} %')
                print(f'\t\tmean squared error: {format(self.mse_val, ".3f")}')

                output_string += ('\nVALIDATION_________________________________________________\n')
                output_string += (f'\tregression equation: {self.equation} achieved the following results over the validation data:\n')
                output_string += (f'\t\tcorrelation: {format(self.correlation_val, ".3f")} %\n')
                output_string += (f'\t\tmean squared error: {format(self.mse_val, ".3f")}\n')
        else:

            if self.f1_score_train != 0:
                print(f'\nTRAINING_________________________________________________')
                print(f'\tclassification algorithm {self.equation} achieved the following results over the training data:')
                print(f'\t\tclassification score: {format(self.class_score_train, ".3f")} for value: {self.class_value_train}')
                print(f'\t\tf1_score: {format(self.f1_score_train, ".3f")}')

                output_string += (f'\nTRAINING_________________________________________________\n')
                output_string += (f'\tclassification algorithm {self.equation} achieved the following results over the training data:\n')
                output_string += (f'\t\tclassification score: {format(self.class_score_train, ".3f")} for value: {self.class_value_train}\n')
                output_string += (f'\t\tf1_score: {format(self.f1_score_train, ".3f")}\n')

            if self.f1_score_test != 0:
                print(f'\nTESTING_________________________________________________')
                print(f'\tclassification algorithm {self.equation} achieved the following results over the testing data:')
                print(f'\t\tclassification score: {format(self.class_score_test, ".3f")} for value: {self.class_value_test}')
                print(f'\t\tf1_score: {format(self.f1_score_test, ".3f")}')

                output_string += (f'\nTESTING_________________________________________________\n')
                output_string += (f'\tclassification algorithm {self.equation} achieved the following results over the testing data:\n')
                output_string += (f'\t\tclassification score: {format(self.class_score_test, ".3f")} for value: {self.class_value_test}\n')
                output_string += (f'\t\tf1_score: {format(self.f1_score_test, ".3f")}\n')

            if self.f1_score_val != 0:
                print(f'\nVALIDATION_________________________________________________')
                print(f'\tclassification algorithm {self.equation} achieved the following results over the validation data:')
                print(f'\t\tclassification score: {format(self.class_score_val, ".3f")} for value: {self.class_value_val}')
                print(f'\t\tf1_score: {format(self.f1_score_val, ".3f")}')

                output_string += (f'\nVALIDATION_________________________________________________\n')
                output_string += (f'\tclassification algorithm {self.equation} achieved the following results over the validation data:\n')
                output_string += (f'\t\tclassification score: {format(self.class_score_val, ".3f")} for value: {self.class_value_val}\n')
                output_string += (f'\t\tf1_score: {format(self.f1_score_val, ".3f")}\n')

        return output_string

