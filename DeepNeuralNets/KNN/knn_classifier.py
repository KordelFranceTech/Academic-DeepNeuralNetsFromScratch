# knn_classifier.py
# Kordel France
########################################################################################################################
# This file establishes the specification and driver for a K-nearest neighbors classifier.
########################################################################################################################


import pandas as pd
import numpy as np



class KNN_Classifier():

    def __init__(self, k, alpha, explanatory_var, response_var):
        """
        Object class that represents a k nearest neighbors classification algorithm.
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


    def predict_data(self, x_data_test):
        """
        Initializes and cals the nearest neighbors algorithm to compute the y_hat estimates
        :param x_data_test: list = the testing data for the explanatory variable
        :param y_data_test: list = the testing data for the response variable
        :returns y_hats: list = the estimates of the response variable copmuted by the knn
        """
        # confirm data shape
        self.x_data_test = x_data_test
        self.n_test, self.num_features = x_data_test.shape

        # establish empty y_hats estimate list
        y_hats = np.zeros(self.n_test)

        # begin finding nearest neighbors
        for index_i in range(0, self.n_test):
            x_i = self.x_data_test[index_i]
            nearest_neighbors = self.compute_nearest_neighbors(x_i)

            # start plurality vote of nearest neighbors
            mode_dict = {}
            majority: int = 0
            for index_j in range(0, self.k):
                cur_key = nearest_neighbors[index_j]
                if cur_key in mode_dict.keys():
                    mode_dict[cur_key] += 1
                    if mode_dict[cur_key] > majority:
                        majority_key = cur_key
                        majority = mode_dict[cur_key]
                        y_hats[index_i] = majority_key
                else:
                    mode_dict[cur_key] = 1

        return y_hats


    def compute_nearest_neighbors(self, x_i):
        """
        Compute the nearest neighbors - simple euclidean distances
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


def build_knn_classifier():
    """
    For testing only - builds a k nearest neighbors classifier over a test dataset
    """
    from sklearn.model_selection import train_test_split
    df = pd.read_csv("/Users/kordelfrance/Documents/School/JHU/Machine Learning/FranceLab/FranceLab2/Lab2/io_files/diabetes.csv")
    x_vals = df.iloc[:, :-1].values
    y_vals = df.iloc[:, -1:].values

    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_vals, y_vals, test_size=1/3, random_state=0)
    knn_classifier = KNN_Classifier(k=3, alpha=10, explanatory_var=1, response_var=1)
    knn_classifier.fit_data(x_data_train, y_data_train)

    y_hats = knn_classifier.predict_data(x_data_test)
    y_hats_correct = 0
    count = 0

    for index_i in range(np.size(y_hats)):
        print('--')
        print(y_data_test[index_i])
        print(y_hats[index_i])
        if y_data_test[index_i] == y_hats[index_i]:
            y_hats_correct += 1
        count += 1

    print(f'accuracy on test set by knn: {(y_hats_correct / count) * 100}')


# build_knn_classifier()



