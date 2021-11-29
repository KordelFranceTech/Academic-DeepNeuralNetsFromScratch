# condensed_knn_regressor.py
# Kordel France
########################################################################################################################
# This file establishes the specification and driver for an edited K-nearest neighbors regressor.
########################################################################################################################


import pandas as pd
import numpy as np
import math
from Lab4 import config


class Edited_KNN_Regressor():

    def __init__(self, k, alpha, explanatory_var, response_var):
        """
        Object class that represents an edited k nearest neighbors regressiion algorithm.
        :param k: int = the number of nearest neighbors to find
        :param alpha: float = the tuning value of the algorithm (learnable)
        :param explanatory_var = the explanatory variable
        :param response_var = the response variable
        """
        self.k = k
        self.alpha = alpha
        self.explanatory_var = explanatory_var
        self.response_var = response_var


    def fit_data(self, x_data_train, y_data_train):
        """
        Initializes the training and testing data and establishes shape
        :param x_data_train: list = the training data for the explanatory variable
        :param y_data_train: list = the trraining data for the response variable
        """
        self.x_data_train = x_data_train
        self.y_data_train = y_data_train
        self.n_train, self.num_features = x_data_train.shape


    def predict_data(self, x_data_test, y_data_test):
        """
        Initializes and cals the nearest neighbors algorithm to compute the y_hat estimates
        :param x_data_test: list = the testing data for the explanatory variable
        :param y_data_test: list = the testing data for the response variable
        :returns y_hats: list = the estimates of the response variable copmuted by the knn
        """
        # confirm data shape
        self.x_data_test = x_data_test
        self.n_test, self.num_features = self.x_data_test.shape

        # initialize an empty y_hats estimate list
        y_hats = np.zeros(self.n_test)

        # get edited nearest neighbors
        for index_i in range(0, self.n_test):
            x_i = self.x_data_test[index_i]

            # get the nearest neighbors with misclassifications "edited" out
            nearest_neighbors = self.compute_edited_nearest_neighbors(x_data_test, y_data_test)

            # apply RBF kernel - see paper for justification
            nn_mean = 0
            nn_var = 0
            for index_j in range(0, len(nearest_neighbors)):
                nn_mean += nearest_neighbors[index_j]
            nn_mean = nn_mean / len(nearest_neighbors)
            y_hats[index_i] = nn_mean

            for index_j in range(0, len(nearest_neighbors)):
                x_temp = nearest_neighbors[index_j] - nn_mean
                x_temp **= 2
                nn_var += x_temp

            nn_var /= len(nearest_neighbors)
            nn_sd = nn_var ** 0.5
            x_s: float = 0
            if len(x_i) > 1:
                x_s += float(x_i[self.explanatory_var])
            else:
                x_s += float(x_i)
            kernel = (x_s - nn_mean) / nn_sd
            kernel **= 2.0
            theta = self.alpha * (math.e ** -kernel)
            # y_hats[index_i] = theta[0]
            y_hats[index_i] = theta

        return y_hats[:self.k]


    def compute_edited_nearest_neighbors(self, x_data_test, y_data_test):
        """
        Computes the nearest neighbors of the dataset
        :param x_data_test: list = the testing data for the explanatory variable
        :param y_data_test: list = the testing data for the response variable
        :returns y_hats: list = the estimates of the response variable copmuted by the knn
        """

        # establish empty y_hats estimate list
        y_hats: list = []

        # begin finding nearest neighbors
        for index_i in range(0, self.n_test):
            x_i = self.x_data_test[index_i]
            nearest_neighbors = self.compute_nearest_neighbors(x_i)

            # compute mode of neighbors
            mode_dict = {}
            majority: int = 0
            for index_j in range(0, self.k):
                # cur_key = nearest_neighbors[index_j][0]
                cur_key = nearest_neighbors[index_j]
                if cur_key in mode_dict.keys():
                    mode_dict[cur_key] += 1
                    if mode_dict[cur_key] > majority:
                        majority = mode_dict[cur_key]
                        # y_hats[index_i] = majority_key
                else:
                    mode_dict[cur_key] = 1

            mode_dict = dict(sorted(mode_dict.items(), key=lambda item: item[1]))
            majority_class = list(mode_dict.keys())[0]

            if config.DEBUG_MODE:
                print(f'\n\nmajority dict: {mode_dict}')
                print(f'majority class: {majority_class}')
                print(f'# in majority: {majority}')

            # edit out any misclassifications and build a new set of "edited" neighbors
            for index_j in range(0, len(nearest_neighbors)):
                if nearest_neighbors[index_j] == majority_class:
                    y_hats.append(nearest_neighbors[index_j])
                else:
                    pass

        if config.DEBUG_MODE:
            print(f'\ty hats: {y_hats}')
            print(f'\t# in y hats: {len(y_hats)}')

        return y_hats



    def compute_nearest_neighbors(self, x_i):
        """
        Compute the nearest neighbors (non-edited version) - simple euclidean distances
        :param x_i: list = the list of training examples
        :returns y_train_sorted: list = list of response variables sorted by euclidean distance
        """
        distances = np.zeros(self.n_train)
        for index_i in range(0, self.n_train):
            distance = self.compute_euclidean_distance(x_i, self.x_data_train[index_i])
            distances[index_i] = distance

        distances_sorted = distances.argsort()
        y_train_sorted = self.y_data_train[distances_sorted]
        return y_train_sorted[:self.k]


    def compute_euclidean_distance(self, x_i, x_data_train):
        """
        Compute euclidean distances between two points
        :param x_i: list = the list of training examples (point a)
        :param x_data_train = list of training examples (point (b)
        :returns distance: list = the euclidean distance between the two points for each data pair
        """
        distance = 0
        for index_i in range(0, len(x_data_train)):
            x_0 = float(x_i[index_i]) - float(x_data_train[index_i])
            x_0 **= 2
            distance += x_0
        distance **= 0.5
        return distance


def build_edited_knn_regressor():
    """
    For testing only - builds a k nearest neighbors classifier over a test dataset
    """
    from sklearn.model_selection import train_test_split
    df = pd.read_csv("/Users/kordelfrance/Documents/School/JHU/Machine Learning/FranceLab/FranceLab2/Lab2/io_files/salary_data.csv")

    x_vals = df.iloc[:, :-1].values
    y_vals = df.iloc[:, -1:].values

    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_vals, y_vals, test_size=1/3, random_state=0)
    knn_model = Edited_KNN_Regressor(k=3, alpha=10, explanatory_var=1, response_var=1)
    knn_model.fit_data(x_data_train, y_data_train)

    y_hats = knn_model.predict_data(x_data_test, y_data_test)
    print(f'predicted values by our model: {np.round(y_hats[:3], 2)}')
    print(f'real values: {y_data_test[:3]}')


# build_edited_knn_regressor()
