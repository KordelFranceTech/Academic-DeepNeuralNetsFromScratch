# loss_function.py
# Kordel France
########################################################################################################################
# This file establishes the driver and helper functions for different loss functions in a neural network / MLP.
########################################################################################################################


import numpy as np
from Lab4.Neural_Network.helper_function import accuracy_score
from Lab4.config import DEMO_MODE

class Loss():
    """This is an object that defines the protocol for loss functions.
    Using inheritance, either cross entropy or mean square error loss are constructed from this protocol.
    """
    def compute_loss(self, y_obs, y_hat):
        return NotImplementedError()


    def compute_gradient(self, y_obs, y_hat):
        raise NotImplementedError()


    def compute_accuracy(self, y_obs, y_hat):
        return 0


class CrossEntropyLoss(Loss):
    """
    Defines the Cross Entropy loss calculation as a loss function.
    Used in classification.
    """
    def __init__(self): pass

    def compute_loss(self, y, p):
        # use this to avoid division by zero or simply add a very small epsilon
        # for example add E = 0.000001 to every value within p
        p = np.clip(p, 1e-15, 1 - 1e-15)
        p = p.astype(float)
        # print(y.shape)
        # print(p.shape)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def compte_accuracy(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def compute_gradient(self, y, p):
        # use this to avoid division by zero or simply add a very small epsilon
        # for example add E = 0.000001 to every value within p
        p = np.clip(p, 1e-15, 1 - 1e-15)
        dx = - (y / p) + (1 - y) / (1 - p)
        if DEMO_MODE:
            print(f'probability: {p[0:2]}')
            print(f'y-values: {y[0:2]}')
            print(f'calculated gradient: {dx[0:2]}')
        return dx


class SquareLoss(Loss):
    """
    Defines the square loss calculation as a loss function.
    Used in regression.
    """
    def __init__(self):
        pass

    def compute_loss(self, y_obs, y_hat):
        l = 0.5 * np.power((y_obs - y_hat), 2)
        return l

    def compute_gradient(self, y_obs, y_hat):
        gradient = -(y_obs - y_hat)
        return gradient



# class CrossEntropyLoss(Loss):
#
#     def __init__(self):
#         pass
#
#
#     def compute_loss(self, y_obs, prob):
#         print('%%%%%%%')
#         print(y_obs[0])
#         print(prob[0])
#         # z = np.zeros((prob.shape[0], 1), dtype='int')
#         # np.append(prob, z, axis=1)
#         # prob = np.hstack((prob, z))
#         # y_obs = np.delete(y_obs, [9,1], axis=1)
#         # prob = np.append(prob, np.zeros([len(prob), 1]), 1)
#         prob = np.append([[1 for _ in range(0, len(prob))]], prob.T, 0).T
#
#         print(f'compute_loss, rows y: {y_obs.shape[0]}')
#         # np.resize(y_obs.shape[0], prob.shape[1])
#         prob = np.clip(prob, 1e-15, 1 - 1e-15)
#         p_loss = - y_obs * np.log(prob) - (1 - y_obs) * np.log(1 - prob)
#         # p_loss = - np.mat(y_obs) * np.mat(np.log(prob)) - (1 - y_obs) * np.log(1 - prob)
#         return p_loss
#
#
#     def compute_accuracy(self, y_obs, prob):
#         y_trues = np.argmax(y_obs, axis=1)
#         y_hats = np.argmax(prob, axis=1)
#         accuracy = np.sum(y_trues == y_hats, axis=0) / len(y_trues)
#         return accuracy
#
#
#     def compute_gradient(self, y_obs, prob):
#         prob = np.clip(prob, 1e-15, 1 - 1e-15)
#         dx = - (y_obs / prob) + (1 - y_obs) / (1 - prob)
#         return dx