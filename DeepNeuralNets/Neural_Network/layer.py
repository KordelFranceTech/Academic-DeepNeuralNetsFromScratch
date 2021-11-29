# layer.py
# Kordel France
########################################################################################################################
# This file establishes classes for different layers that may be used in the construction of neural networks or MLPs.
########################################################################################################################


import math
import numpy as np
import copy
import time
from Lab4.Neural_Network.activation_function import Sigmoid, ReLU, LeakyReLU, SoftMax, HyperbolicTangent
from Lab4.config import DEBUG_MODE, DEMO_MODE


class Layer(object):
    """
    This is a protocol that constructs a layer of neurons, defines their activation functions,
    and contains helper methods to coordinate forward passes for training and backward passes
    for backprop.
    """

    def define_input_shape(self, shape):
        """
        sets the input dimensionality
        """
        self.input_shape = shape

    def layer_title(self):
        """
        Not critical but helpful for debugging.
        Helpful when you want a summary of layers in model and want to print to log.
        """
        if self.__class__.__name__ == 'ActivationLayer':
            return 'Activation'
        elif 'Layer' in self.__class__.__name__:
            arr = self.__class__.__name__.split('Layer')
            return f'{arr[0]} Layer'
        return self.__class__.__name__

    def parameters(self):
        """
        parameter count that can be trained
        """
        return 0

    def forward_pass(self, x_data, training):
        """
        passes gradient / weights forward to next layer
        """
        raise NotImplementedError()

    def backward_pass(self, gradient):
        """
        This function is critical and especially useful for backprop.
        It coordinates the gradient calculation and cumulative weights to be used for the next layer of backprop updates.
        """
        raise NotImplementedError()

    def output_shape(self):
        """
        dimensionality of output of layer after forward pass - fed as input into next layer
        """
        raise NotImplementedError()


class DenseLayer(Layer):
    """
    A layer that represents a fully connected layer.
    All input nodes obtain connections to all output nodes.
    For this assignment, I used all densely connected layers.
    """
    def __init__(self, neurons, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.neurons = neurons
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        # init weights
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.neurons))
        self.w0 = np.zeros((1, self.neurons))
        # optimizers for weights
        self.W_opt  = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        """
        parameter count that can be trained
        """
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, x_data, training=True):
        """
        passes gradient / weights forward to next layer
        """
        self.layer_input = x_data
        if DEBUG_MODE:
            print("######")
            print(len(x_data))
            print(type(x_data).__name__)
            print(len(self.W))
            print(type(self.W).__name__)

        if DEMO_MODE:
            print(f'\n\nlayer input from forward pass: {self.layer_input}')
            # time.sleep(5)

        return x_data.dot(self.W) + self.w0

    def backward_pass(self, gradient):
        """
        This function is critical and especially useful for backprop.
        It coordinates the gradient calculation and cumulative weights to be used for the next layer of backprop updates.
        """

        # copy the weights
        W = self.W

        # if the layer is trainable, calculate the gradient and update the weights within the layer
        if self.trainable:
            grad_w = self.layer_input.T.dot(gradient)
            grad_w0 = np.sum(gradient, axis=0, keepdims=True)

            self.W = self.W_opt.update_weights(self.W, grad_w)
            self.w0 = self.w0_opt.update_weights(self.w0, grad_w0)

        # the gradient is now calculated based on weights from the forward pass.
        # no return it so it can be used on the next layer.
        gradient = gradient.dot(W.T)
        return gradient

    def output_shape(self):
        """
        dimensionality of output of layer after forward pass - fed as input into next layer
        """
        return (self.neurons, )



class DropoutLayer(Layer):
    def __init__(self, probability=0.2):
        """
        This is an experimental layer not part of the original assignment.
        I am just highly interested in dropout as a regularizer, so I wanted to experiment with it.
        """
        self.probability = probability
        self.mask = None
        self.input_shape = None
        self.neurons = None
        self.pass_through = True
        self.trainable = True

    def forward_pass(self, x_data, training=True):
        """
        passes gradient / weights forward to next layer
        """
        c = (1 - self.probability)
        if training:
            self.mask = np.random.uniform(size=x_data.shape) > self.probability
            c = self.mask
        return x_data * c

    def backward_pass(self, gradient):
        """
        This function is critical and especially useful for backprop.
        It coordinates the gradient calculation and cumulative weights to be used for the next layer of backprop updates.
        """
        return gradient * self.mask

    def output_shape(self):
        """
        dimensionality of output of layer after forward pass - fed as input into next layer
        """
        return self.input_shape


activation_functions: dict = {
    'relu': ReLU,
    'leaky_relu': LeakyReLU,
    'sigmoid': Sigmoid,
    'softmax': SoftMax,
    'hyperbolic_tangent': HyperbolicTangent
}

class ActivationLayer(Layer):
    def __init__(self, name):
        """
        Not realy a "layer". It was just helpful to segregate activation as its own layer.
        This finds the designated activation function and stores weights to help work with backprop.
        """
        self.activation_name = name
        self.activation_function = activation_functions[name]()
        self.trainable = True

    def layer_name(self):
        return 'Activation'

    def forward_pass(self, x_data, training=True):
        """
        passes gradient / weights forward to next layer
        """
        self.layer_input = x_data
        return self.activation_function(x_data)

    def backward_pass(self, gradient):
        """
        This function is critical and especially useful for backprop.
        It coordinates the gradient calculation and cumulative weights to be used for the next layer of backprop updates.
        """
        return gradient * self.activation_function.compute_gradient(self.layer_input)

    def output_shape(self):
        """
        dimensionality of output of layer after forward pass - fed as input into next layer
        """
        return self.input_shape



class BatchNormLayer(Layer):
    """
    EXPERIMENTAL ONLY
    """
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.trainable = True
        self.eps = 0.01
        self.running_mean = None
        self.running_var = None

    def initialize(self, optimizer):
        """
        parameter count that can be trained
        """
        # Initialize the parameters
        self.gamma  = np.ones(self.input_shape)
        self.beta = np.zeros(self.input_shape)
        # parameter optimizers
        self.gamma_opt  = copy.copy(optimizer)
        self.beta_opt = copy.copy(optimizer)

    def parameters(self):
        """
        parameter count that can be trained
        """
        return np.prod(self.gamma.shape) + np.prod(self.beta.shape)

    def forward_pass(self, x_data, training=True):
        """
        passes gradient / weights forward to next layer
        """

        # Initialize running mean and variance if first run
        if self.running_mean is None:
            self.running_mean = np.mean(x_data, axis=0)
            self.running_var = np.var(x_data, axis=0)

        if training and self.trainable:
            mean = np.mean(x_data, axis=0)
            var = np.var(x_data, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Statistics saved for backward pass
        self.X_centered = x_data - mean
        self.stddev_inv = 1 / np.sqrt(var + self.eps)

        X_norm = self.X_centered * self.stddev_inv
        output = self.gamma * X_norm + self.beta

        return output

    def backward_pass(self, gradient):
        """
        This function is critical and especially useful for backprop.
        It coordinates the gradient calculation and cumulative weights to be used for the next layer of backprop updates.
        """

        # Save parameters used during the forward pass
        gamma = self.gamma

        # If the layer is trainable the parameters are updated
        if self.trainable:
            X_norm = self.X_centered * self.stddev_inv
            grad_gamma = np.sum(gradient * X_norm, axis=0)
            grad_beta = np.sum(gradient, axis=0)

            self.gamma = self.gamma_opt.update_weights(self.gamma, grad_gamma)
            self.beta = self.beta_opt.update_weights(self.beta, grad_beta)

        batch_size = gradient.shape[0]

        # The gradient of the loss with respect to the layer inputs (use weights and statistics from forward pass)
        accum_grad = (1 / batch_size) * gamma * self.stddev_inv * (
            batch_size * gradient
            - np.sum(gradient, axis=0)
            - self.X_centered * self.stddev_inv**2 * np.sum(gradient * self.X_centered, axis=0)
            )
        return gradient

    def output_shape(self):
        """
        dimensionality of output of layer after forward pass - fed as input into next layer
        """
        return self.input_shape



































