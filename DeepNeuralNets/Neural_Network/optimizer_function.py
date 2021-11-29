# optimizer_function.py
# Kordel France
########################################################################################################################
# This file establishes the driver and helper functions for different optimizer algorithms in a neural network / MLP.
########################################################################################################################


import numpy as np


class Stochastic_Gradient_Descent():
    """
    This defines the protocl for stochastic gradient decent.
    I expect this to work better than Adam in the future, but right now, it seems to have difficulty with
    categorical data represented as different matrices.
    It was not used in the assignment.
    """

    def __init__(self, learning_rate:float=0.01, momentum:float=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weights_update = None

    def update_weights(self, weights, gradient):
        if self.weights_update is None:
            self.weights_update = np.zeros(np.shape(weights))
        self.weights_update = self.momentum * self.weights_update + (1 - self.momentum) * gradient
        dw = weights - self.learning_rate * self.weights_update
        return dw


class Adam():
    """
    This defines the protocol for the Adam optimizer.
    I tuned the learning rate but did not experiment with different values of beta1 and beta2.
    The paper on Adam has some interesting insight on how to tune these values I would like to
    investigate in the future.
    """

    def __init__(self, learning_rate:float=0.001, beta1:float=0.9, beta2:float=0.999):
        self.learning_rate = learning_rate
        self.epsilon = 1e-8
        self.moment_vector = None
        self.inf_norm = None
        self.beta1 = beta1
        self.beta2 = beta2

    def update_weights(self, weights, gradient):
        if self.moment_vector is None:
            self.moment_vector = np.zeros(np.shape(gradient))
            self.inf_norm = np.zeros(np.shape(gradient))

        self.moment_vector = self.beta1 * self.moment_vector + (1 - self.beta1) * gradient
        self.inf_norm = self.beta2 * self.inf_norm + (1 - self.beta2) * np.power(gradient, 2)

        moment_vector_hat = self.moment_vector / (1 - self.beta1)
        inf_norm_hat = self.inf_norm / (1 - self.beta2)
        self.weights_update = self.learning_rate * moment_vector_hat / ((inf_norm_hat ** 0.5) + self.epsilon)
        return weights - self.weights_update


