# neural_net.py
# Kordel France
########################################################################################################################
# This file establishes the driver and helper functions used for the construction of a neural network.
########################################################################################################################


from Lab4.Neural_Network.helper_function import auto_generate_batches, status_indicator
from Lab4.config import DEMO_MODE
import numpy as np
import progressbar      # for clean interface as model trains  - improves aesthetics and keeps non-verbose log
import time


class NeuralNet():

    def __init__(self, optimizer_function, loss_function, val_data=None):
        """
        This class defines the protocol for a neural network.
        'Multilayer perceptron' or 'MLP' may have been a more appropriate class name in hindsight.
        This class is the core for the autoencoder and all other neural nets in the assignment.
        :param optimizer_function: Optimizer - the selection of optimizer to use, choice between SGD or Adam.
        :param loss_function: Loss - the selection of loss function to use, choice between Cross Entropy and MSE
        :param val_data: list - optional validation data to pass in to network to help prepare it for training.
        """
        # establish the neural network conditions
        self.optimizer_function = optimizer_function
        self.loss_function = loss_function()
        self.layers: list = []

        # helpful for debugging
        self.errors: dict = {
            'training': [],
            'validation': []
        }
        # this is to provide a nice clean interface during training
        self.progressbar = progressbar.ProgressBar(widgets=status_indicator)
        self.val_set = None
        if val_data:
            x_obs, y_obs = val_data
            self.val_set = {
                'X': x_obs,
                'y': y_obs
            }


    def freeze_layers(self, should_freeze):
        """
        Freezes or unfreezes layers, allowing weights of layers to be adjusted and trained
        """
        for layer in self.layers:
            layer.freeze = should_freeze


    def add_layer(self, layer):
        """
        Prepare the network for the layer and add it to the network
        """

        # check if first layer of network
        # if not first layer, then define input shape
        if self.layers:
            layer.define_input_shape(shape=self.layers[-1].output_shape())

        # initialize weights of layer if it has weights
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer_function)

        # add layer to other layers of neural net
        self.layers.append(layer)


    def test_single_batch(self, x_obs, y_obs):
        """
        Evaluate the neural net over a single batch
        """
        y_hats: list = self.perform_forward_pass(x_obs, is_training=False)
        # print(f'neural net->test_single_batch->\n\ty_obs: {y_obs[0]}\n\ty_hats: {y_hats[0]}')
        loss: np.ndarray = np.mean(self.loss_function.compute_loss(y_obs, y_hats))
        accuracy: float = self.loss_function.compute_accuracy(y_obs, y_hats)
        return loss, accuracy


    def train_single_batch(self, x_obs, y_obs):
        """
        Evaluate neural net over a single batch of samples to compute gradient
        """
        y_hats: list = self.perform_forward_pass(x_obs)
        # print(f'1 neural net->test_single_batch->\n\ty_obs: {y_obs[0]}\n\ty_hats: {y_hats[0]}')
        loss: np.ndarray = np.mean(self.loss_function.compute_loss(y_obs, y_hats))
        # print(f'2 neural net->test_single_batch->\n\ty_obs: {y_obs[0]}\n\ty_hats: {y_hats[0]}')
        accuracy: float = self.loss_function.compute_accuracy(y_obs, y_hats)

        # compute gradient of loss function respect to y_hats
        gradient: float = self.loss_function.compute_gradient(y_obs, y_hats)
        if DEMO_MODE:
            print(f'\n\ny observed: {y_obs[0:2]}')
            print(f'\n\ny predicted: {y_hats[0:2]}')
            print(f'\n\ngradient: {gradient[0:2]}')
            # time.sleep(5)
        # update network weights through backprop
        self.perform_backprop(loss_gradient=gradient)
        return loss, accuracy


    def fit_neural_net(self, x_obs, y_obs, epochs: int, batch_size: int):
        """
        Batch data, train model, and fit it to data
        """
        for index_i in self.progressbar(range(0, epochs)):
            error_batch: list = []
            for x_batch, y_batch in auto_generate_batches(x_obs, y_obs, batch_size):

                # print(f'x, shape: {x_batch.shape}\n\t')
                # print(x_batch)
                # print(f'\n\ny, shape: {y_batch.shape}\n\t')
                # print(y_batch)
                loss, accuracy = self.train_single_batch(x_batch, y_batch)
                error_batch.append(loss)

            self.errors['training'].append(np.mean(error_batch))

            if self.val_set is not None:
                val_loss, val_accuracy = self.test_single_batch(self.val_set['X'], self.val_set['y'])
                self.errors['validation'].append(val_loss)

        return self.errors['training'], self.errors['validation']


    def perform_forward_pass(self, x_obs, is_training: bool = True):
        """
        Compute the output (y-hats) for one full forward pass through the neural net
        """
        output = x_obs
        # print(f'x_obs :\n\t{x_obs.shape}')
        for layer in self.layers:
            output = layer.forward_pass(output, is_training)
        # print(f'output :\n\t{output.shape}')
        if DEMO_MODE:
            print(f'\n\nlayer output from forward pass: {output[0:2]}')
            # time.sleep(5)
        return output


    def perform_backprop(self, loss_gradient):
        """
        Perform the backprop algorithm by propagating the gradient backwards through the neural net.
        Update the weights accordingly.
        """
        for layer in reversed(self.layers):
            loss_gradient = layer.backward_pass(loss_gradient)


    def model_analytics(self, title: str = 'Model Analytics'):
        print(f'\n\n{title}')
        print('\tlayer type\t# parameters\toutput dimensionality')
        print('_____________________________________________________________')
        param_count: float = 0
        for layer in self.layers:
            layer_name = layer.layer_title()
            parameters = layer.parameters()
            output_shape = layer.output_shape()
            # table_data.append([layer_name, str(parameters), str(output_shape)])
            print(f'\t{layer_name}\t\t{parameters}\t\t{output_shape}')
            param_count += parameters
        print('_____________________________________________________________')
        print(f'total parameters: %d\n' % param_count)
        # time.sleep(3)


    def calculate_y_hat(self, x_obs):
        """
        Calculate the predicted values for y based on observed x
        """
        y_hats = self.perform_forward_pass(x_obs, is_training=False)
        return y_hats




