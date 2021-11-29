# autoencoder.py
# Kordel France
########################################################################################################################
# This file establishes the class for an autoencoder that uses a deep neural network with fully connected layers.
########################################################################################################################


from Lab4.Neural_Network.optimizer_function import Adam
from Lab4.Neural_Network.loss_function import CrossEntropyLoss, SquareLoss
from Lab4.Neural_Network.layer import DenseLayer, ActivationLayer
from Lab4.Neural_Network.neural_net import NeuralNet
from Lab4.config import DEBUG_MODE, DEMO_MODE
import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


class Autoencoder():

    def __init__(self, dataframe):
        """
        Object class that sets up and coordinates an autoencoder through the NeuralNet class.
        :param dataframe: pd.Datafame - the dataframe that will act as the input
        """

        # establish input dimensionality
        self.data_rows = 1
        self.data_cols = dataframe.shape[1]
        self.data_dim = self.data_rows * self.data_cols
        # self.latent_dim = 128

        # declare optimizer - Adam shown to work best through trace runs
        optimizer_function = Adam(learning_rate=0.0002, beta1=0.5)
        # select loss function depending on whether classification or regression task
        # requirement from assignment
        loss_function = CrossEntropyLoss

        # constructor for encoder
        self.encoder = self.construct_encoder(optimizer_function, loss_function)
        # constructor for decoder
        self.decoder = self.construct_decoder(optimizer_function, loss_function)

        # conslidate encoder and decoder together
        self.autoencoder = NeuralNet(optimizer_function=optimizer_function, loss_function=loss_function)
        self.autoencoder.layers.extend(self.encoder.layers)
        self.autoencoder.layers.extend(self.decoder.layers)

        # print summary of layers to screen for QC
        self.autoencoder.model_analytics(title='variational autoencoder (VAE)')


    def construct_encoder(self, optimizer_function, loss_function):
        """
        Constructor of the encoder - establishes a neural net with encoding layers.
        Best practice to have dimensionality (# neurons) decrease with each subsequent layer.
        :param optimizer_function: Optimizer - the algorithm that will act as the optimizer and perform learning
        :param loss_function: Loss - the algorithm that will calculate loss and determine y-yhat error
        """
        encoder = NeuralNet(optimizer_function=optimizer_function, loss_function=loss_function)
        encoder.add_layer(DenseLayer(10, input_shape=(self.data_cols,)))
        encoder.add_layer(ActivationLayer('leaky_relu'))
        encoder.add_layer(DenseLayer(self.data_dim))
        encoder.model_analytics('Encoder')
        return encoder


    def construct_decoder(self, optimizer_function, loss_function):
        """
        Constructor of the decoder - establishes a neural net with decoding layers.
        Best practice to have dimensionality (# neurons) increase with each subsequent layer.
        :param optimizer_function: Optimizer - the algorithm that will act as the optimizer and perform learning
        :param loss_function: Loss - the algorithm that will calculate loss and determine y-yhat error
        """
        decoder = NeuralNet(optimizer_function=optimizer_function, loss_function=loss_function)
        decoder.add_layer(DenseLayer(10, input_shape=(self.data_cols,)))
        decoder.add_layer(ActivationLayer('leaky_relu'))
        decoder.add_layer(DenseLayer(self.data_cols))
        decoder.add_layer(ActivationLayer('hyperbolic_tangent'))
        decoder.model_analytics('Decoder')
        return decoder


    def train_autoencoder(self, x_data, epochs, batch_size=128, time_step_save=50):
        """
        Starts and coordinates the training of the autoencoder.
        Try to keep this as minimal as possible and avoid doing any data transformations here.
        :param x_data: pd.Dataframe - the dataframe that will be input, encoded, decoded, and reconstructed
        :param epochs: int - the number of iterations to update knowledge transfer
        :param batch_size: int - the number of training samples we give the algorithm at one time
        """

        # establish working variables
        x_input = x_data
        x_input = (x_input.astype(np.float32) - 127.5) / 127.5
        loss_values = []

        # perform a single batch of training and compute the loss
        for index_i in range(0, epochs):
            epoch = index_i
            sub_index = np.random.randint(0, x_input.shape[0], batch_size)
            data_train = x_input[sub_index]

            loss, _ = self.autoencoder.train_single_batch(data_train, data_train)
            loss_values.append(loss)
            if epochs / (index_i + 1) < 1:
                print('%d - Encoder loss: %f' % (epoch, loss))
            else:
                print('%d - Decoder loss: %f' % (epoch, loss))

        if DEMO_MODE:
            print(f'\n\nprediction from autoencoder: {self.autoencoder.calculate_y_hat(x_input)[0:2]}')

        if DEBUG_MODE:
            training, = plt.plot(range(epochs), loss_values, label='training loss')
            plt.legend(handles=[training])
            plt.title('Autoencoder Loss')
            plt.ylabel('loss')
            plt.xlabel('epoch #')
            plt.show()

        # training done, return data
        return loss_values





