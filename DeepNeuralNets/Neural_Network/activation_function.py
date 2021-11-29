# activation_function.py
# Kordel France
########################################################################################################################
# This file establishes the driver and helper functions for different activation functions in a neural network / MLP.
########################################################################################################################


import numpy as np
import math



class Sigmoid():
    """
    Class for computation of the Sigmoid (logistic) activator
    """
    def __call__(self, x):
        x = x.astype(float)
        return 1 / (1 + np.exp(-x))

    def compute_gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class SoftMax():
    """
    Class for computation of the SoftMax activator
    """

    def __call__(self, x):
        e_x = np.e ** (x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def compute_gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)

class HyperbolicTangent():
    """
    Class for computation of the Hyperbolic Tangent activator
    """
    def __call__(self, x):
        return 2 / (1 + np.exp(-2*x)) - 1

    def compute_gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)

class ReLU():
    """
    Class for computation of the rectified linear unit activator
    """
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def compute_gradient(self, x):
        return np.where(x >= 0, 1, 0)

class LeakyReLU():
    """
    Class for computation of the a modified rectified linear unit activator
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def compute_gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)



# class ReLU():
#     """
#     Class for computation of the rectified linear unit activator
#     """
#     def __call__(self, x_obs):
#         return np.where(x_obs >= 0, x_obs, 0)
#
#     def compute_gradient(self, x_obs):
#         return np.where(x_obs >= 0, 1, 0)
#
#
# class SoftMax():
#     """
#     Class for computation of the SoftMax activator
#     """
#
#     def __call__(self, x_obs):
#         a = np.exp(x_obs - np.max(x_obs, axis=1, keepdims=True))
#         s = a / np.sum(a, axis=1, keepdims=True)
#         return s
#
#     def compute_gradient(self, x_obs):
#         probability = self.__call__(x_obs)
#         dx = probability * (1 - probability)
#         return dx
#
#
# class Sigmoid():
#     """
#     Class for computation of the Sigmoid (logistic) activator
#     """
#
#     def __call__(self, x_obs):
#         s = 1 / (1 + np.exp(-x_obs))
#         return  s
#
#     def compute_gradient(self, x_obs):
#         dx = self.__call__(x_obs) * (1 - self.__call__(x_obs))
#         return dx
#
#
# class HyperbolicTangent():
#     """
#     Class for computation of the Hyperbolic Tangent activator
#     """
#     def __call__(self, x_obs):
#         t = 2 / (1 + np.exp(-2 * x_obs)) - 1
#         return t
#
#     def compute_gradient(self, x_obs):
#         dx = 1 - np.power(self.__call__(x_obs), 2)
#         return dx
#
#
# class LeakyReLU():
#     """
#     Class for computation of the a modified rectified linear unit activator
#     """
#     def __init__(self, alpha=0.2):
#         self.alpha = alpha
#
#     def __call__(self, x_obs):
#         return np.where(x_obs >= 0, x_obs, self.alpha * x_obs)
#
#     def compute_gradient(self, x_obs):
#         return np.where(x_obs >= 0, x_obs, self.alpha * x_obs)
#



