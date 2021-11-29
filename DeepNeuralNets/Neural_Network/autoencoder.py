# autoencoder.py
# Kordel France
########################################################################################################################
# This file establishes the class for an autoencoder that uses a deep neural network with fully connected layers.
########################################################################################################################


from Lab4.Neural_Network.optimizer_function import Adam
from Lab4.Neural_Network.loss_function import CrossEntropyLoss, SquareLoss
from Lab4.Neural_Network.layer import DenseLayer, DropoutLayer, ActivationLayer, BatchNormLayer
from Lab4.Neural_Network.neural_net import NeuralNet
import numpy as np


class Autoencoder():

    def __init__(self, dataframe, is_classification, val_data):
        """
        Object class that sets up and coordinates an autoencoder through the NeuralNet class.
        :param dataframe: pd.Datafame - the dataframe that will act as the input
        :param is_classification: bool - indicates whether task is classification or regression task
        :param val_data: list - optional validation data to send to the neural network when converting from AE to NN.
        """

        # establish input dimensionality
        self.data_rows = 1
        self.data_cols = dataframe.shape[1]
        self.data_dim = self.data_rows * self.data_cols

        # declare optimizer - Adam shown to work best through trace runs
        optimizer_function = Adam(learning_rate=0.0002, beta1=0.5)

        # select loss function depending on whether classification or regression task
        # requirement from assignment
        if is_classification:
            loss_function = CrossEntropyLoss
        else:
            loss_function = SquareLoss
        # loss_function = CrossEntropyLoss

        # constructor for encoder
        self.encoder = self.construct_encoder(optimizer_function, loss_function, is_classification)
        # constructor for decoder
        self.decoder = self.construct_decoder(optimizer_function, loss_function, is_classification)

        # conslidate encoder and decoder together
        self.autoencoder = NeuralNet(optimizer_function=optimizer_function, loss_function=loss_function, val_data=val_data)
        self.autoencoder.layers.extend(self.encoder.layers)
        self.autoencoder.layers.extend(self.decoder.layers)

        # print summary of layers to screen for QC
        self.autoencoder.model_analytics(title='variational autoencoder (VAE)')


    def construct_encoder(self, optimizer_function, loss_function, is_classification):
        """
        Constructor of the encoder - establishes a neural net with encoding layers.
        Best practice to have dimensionality (# neurons) decrease with each subsequent layer.
        :param optimizer_function: Optimizer - the algorithm that will act as the optimizer and perform learning
        :param loss_function: Loss - the algorithm that will calculate loss and determine y-yhat error
        :param is_classification: bool - indicates whether task is classification or regression task
        """
        encoder = NeuralNet(optimizer_function=optimizer_function, loss_function=loss_function)
        encoder.add_layer(DenseLayer(self.data_cols, input_shape=(self.data_cols,)))
        encoder.add_layer(ActivationLayer('sigmoid'))
        encoder.add_layer(DenseLayer(self.data_dim))
        encoder.model_analytics('Encoder')
        return encoder


    def construct_decoder(self, optimizer_function, loss_function, is_classification):
        """
        Constructor of the decoder - establishes a neural net with decoding layers.
        Best practice to have dimensionality (# neurons) increase with each subsequent layer.
        :param optimizer_function: Optimizer - the algorithm that will act as the optimizer and perform learning
        :param loss_function: Loss - the algorithm that will calculate loss and determine y-yhat error
        :param is_classification: bool - indicates whether task is classification or regression task
        """
        decoder = NeuralNet(optimizer_function=optimizer_function, loss_function=loss_function)
        decoder.add_layer(DenseLayer(self.data_cols, input_shape=(self.data_cols,)))

        # select loss function depending on whether classification or regression task
        # requirement from assignment
        if is_classification:
            decoder.add_layer(ActivationLayer('softmax'))
        else:
            decoder.add_layer(ActivationLayer('relu'))
        decoder.model_analytics('Decoder')
        return decoder


    def train_autoencoder(self, x_data, epochs, batch_size=20):
        """
        Starts and coordinates the training of the autoencoder.
        Try to keep this as minimal as possible and avoid doing any data transformations here.
        :param x_data: pd.Dataframe - the dataframe that will be input, encoded, decoded, and reconstructed
        :param epochs: int - the number of iterations to update knowledge transfer
        :param batch_size: int - the number of training samples we give the algorithm at one time
        """

        # establish working variables
        x_input = x_data
        loss_values = []

        # perform a single batch of training and compute the loss
        for index_i in range(0, epochs):
            epoch = index_i
            sub_index = np.random.randint(0, x_input.shape[0], batch_size)
            data_train = x_input[sub_index]

            loss, _ = self.autoencoder.train_single_batch(data_train, data_train)
            loss_values.append(loss)
            print('%d - Autoencoder loss: %f' % (epoch, loss))

        # training done, return data
        return loss_values


    def remove_output_layer(self):
        """
        Removes the output layers and prepares the autoencoder (now just an encoder) for attachment
        to a neural network.
        """
        self.autoencoder.layers.pop()
        self.autoencoder.layers.pop()


    def add_prediction_layer_for_neural_network(self):
        """
        Adds a prediction layer to the neural network once output layers are removed and hidden layer
        is added.
        """
        self.encoder.add_layer(DenseLayer(10, input_shape=(self.data_cols,)))
        self.encoder.add_layer(ActivationLayer('softmax'))
        self.autoencoder.layers.extend(self.encoder.layers)


