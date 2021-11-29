# __main__.py
# Kordel France
########################################################################################################################
# This file establishes the driver for a decision tree model.
########################################################################################################################



from Lab4 import data_processing
from Lab4 import encoding
from Lab4 import config
from Lab4.cross_validate import construct_data_folds, get_datasets_for_fold
from Lab4 import standardize
from Lab4.Linear_Network import linear_network
from Lab4.Logistic_Regression import logistic_regression
from Lab4.Neural_Network.neural_net import NeuralNet
from Lab4.Neural_Network.autoencoder_solo import Autoencoder as AutoencoderSolo
from Lab4.Neural_Network.autoencoder import Autoencoder
from Lab4.Neural_Network.optimizer_function import Adam
from Lab4.Neural_Network.loss_function import CrossEntropyLoss, SquareLoss
from Lab4.Neural_Network.layer import DenseLayer, ActivationLayer
from Lab4.Neural_Network.helper_function import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

def perform_k_fold_cross_validation(algorithm: int,
                                    num_folds: int,
                                    dataframe: pd.DataFrame,
                                    is_classify: bool,
                                    explanatory_col: int,
                                    response_col: int):
    """
    Driver function for k-fold cross validation over the algorithm of choice.
    :param num_folds: int - the number of folds, or 'k'
    :param dataframe: pd.Dataframe - the dataframe to construct the separate dataset folds from
    :param is_classify: bool - a flag that indicates whether a classification or regression algorithm should be selected
    :param explanatory_col: int - the column indicating the explanatory variable, or x series (optional)
    :param response_col: int - the column indicating the response variable, or y series
    """

    # initialize bins, construct data folds for KFCV from dataframe
    bin_list = construct_data_folds(num_folds, dataframe)

    # perform training and testing over each KFCV fold
    for index_i in range(0, num_folds):
        train_data, test_data = get_datasets_for_fold(index_i, bin_list, False)
        x_vals = dataframe.iloc[:, dataframe.columns != response_col].values
        y_vals = dataframe.iloc[:, response_col].values

        if config.DEBUG_MODE:
            print(f'dataframe from main: {dataframe}')
            print(f'x from main: {x_vals}')
            print(f'y from main: {y_vals}')
            print(f'lengths: x: {len(x_vals)}, y: {len(y_vals)}')

        # we have received the training and testing data
        # now stratify it according to x- and y- dataframes
        # note that this is only used for the linear network
        x_data_train = np.array(train_data.drop([response_col], 1))
        x_data_test = np.array(test_data.drop([response_col], 1))
        x0_data_train = np.array(train_data[2])
        x0_data_test = np.array(test_data[2])
        x1_data_train = np.array(train_data[3])
        x1_data_test = np.array(test_data[3])
        y_data_train = np.array(train_data[response_col])
        y_data_test = np.array(test_data[response_col])
        y0_data_train = np.array(train_data[4])
        y0_data_test = np.array(test_data[4])
        y1_data_train = np.array(train_data[5])
        y1_data_test = np.array(test_data[5])

        if config.DEBUG_MODE:
            print(f'dataframe:\n\t{dataframe}')
            print(f' x test data: {len(x_data_test)}\n\t:{x_data_test}')
            print(f' x train data: {len(x_data_train)}\n\t{x_data_train}')
            print(f' y test data: {len(y_data_test)}\n\t{y_data_test}')
            print(f' y train data: {len(y_data_train)}\n\t{y_data_train}')

        ###############################################################################################################
        # build a linear network
        if algorithm == 0:

            #call the linear network driver
            m, b = linear_network.fit_linear_network(x0_data_train,
                                                     x0_data_test,
                                                     x1_data_train,
                                                     x1_data_test,
                                                     y0_data_train,
                                                     y0_data_test,
                                                     y1_data_train,
                                                     y1_data_test)
            print(m, b)
            print('\n\n______________________________________________')
            print(f'equation for linear network discriminator: y = {m} x + {b}')
            print('______________________________________________')

        ###############################################################################################################
        # build a logistic regressor
        elif algorithm == 1:

            # call the logistic regression driver
            weights, intercept = logistic_regression.fit_logistic_regression(x_vals, y_vals, epochs=50000, learning_rate=5e-5, should_add_bias=True)
            print(weights)
            print(intercept)

            # construct equation
            equation_str: str = ''
            y_hats = []
            for i in range(0, len(x_data_test)):
                for j in range(0, len(x_data_test[i])):
                    beta = 0
                    for k in range(0, len(weights)):
                        x_i = x_data_test[i]
                        weight = weights[k]
                        beta_i = weight * x_i[j]
                        beta += beta_i
                        equation_str += f' {weight} * x_{k} + '
                    y_hats.append(beta)
            print('\n\n______________________________________________')
            print(f'equation for logistic regressor: p/(1-p) = {equation_str}')
            print('______________________________________________')

        ###############################################################################################################
        # build an encoder only - not attached to a feed forwardneural network hidden layer
        elif algorithm == 2:

            x_vals_ae = dataframe
            for i in range(0, len(x_vals_ae.columns)):
                if i != response_col:
                    x_vals_ae.iloc[i].values[:] = 0

            x_vals_ae = np.array(x_vals)
            autoencoder = AutoencoderSolo(x_vals_ae)
            autoencoder.train_autoencoder(x_data=x_vals_ae, epochs=100, batch_size=64)

        ###############################################################################################################
        # build a feed forward neural network only - not combined with an encoder from an autoencoder
        elif algorithm == 3:

            # declare optimizer
            optimizer = Adam()

            # we need to cast the x- and y-vals as a new type for nn training
            # do it here
            x_vals_nn = dataframe
            for i in range(0, len(x_vals_nn.columns)):
                if i != response_col:
                    x_vals_nn.iloc[i].values[:] = 0

            y_vals_nn = dataframe
            for i in range(0, len(y_vals_nn.columns)):
                if i == response_col:
                    y_vals_nn.iloc[i].values[:] = 0

            # now declare these vals as np array
            # this ensures they are all of identical data type and math-operable
            x_data_train = np.array(x_vals_nn)
            y_data_train = np.array(y_vals_nn)

            # define # of samples and hidden nodes within hidden layer
            n_samples, n_features = x_data_train.shape
            n_hidden = n_features

            # define training, testing split
            # note that this is different from the bin list up above, particularly altered for nn training
            x_train, x_test, y_train, y_test = train_test_split(x_data_train, y_data_train, test_size=0.4)

            # select loss function based on task
            # requirement of assignment
            if is_classify:
                loss_function = CrossEntropyLoss
            else:
                loss_function = SquareLoss

            # init neural net
            neural_net = NeuralNet(optimizer_function=optimizer, loss_function=loss_function, val_data=(x_test, y_test))
            neural_net.add_layer(DenseLayer(n_hidden, input_shape=(n_features,)))
            neural_net.add_layer(ActivationLayer('sigmoid'))                        # not really a layer
            neural_net.add_layer(DenseLayer(n_hidden, input_shape=(n_features,)))
            neural_net.add_layer(ActivationLayer('sigmoid'))                        # not really a layer
            neural_net.add_layer(DenseLayer(n_hidden, input_shape=(n_features,)))
            neural_net.add_layer(ActivationLayer('sigmoid'))                        # not really a layer
            neural_net.add_layer(DenseLayer(n_hidden, input_shape=(n_features,)))

            # select loss function based on task
            # requirement of assignment
            if is_classify:
                neural_net.add_layer(ActivationLayer('softmax'))
            else:
                neural_net.add_layer(ActivationLayer('relu'))

            # print model summary to console
            neural_net.model_analytics(title='Neural Network')

            # neural net defined. now train it.
            train_loss, val_loss = neural_net.fit_neural_net(x_train, y_train, epochs=25, batch_size=10)
            n = len(train_loss)

            # plot the training and testing losses for a visual and for quality control check
            if config.DEMO_MODE:
                training, = plt.plot(range(n), train_loss, label='training error')
                validation, = plt.plot(range(n), val_loss, label='validation error')
                plt.legend(handles = [training, validation])
                plt.title('Training Error')
                plt.ylabel('error')
                plt.xlabel('epoch #')
                plt.show()

            # perform testing on the neural net to get our accuracy of results
            _, accuracy = neural_net.test_single_batch(x_test, y_test)
            # print(f'accuracy: {get_accuracy(accuracy)}')

            # get predictions from our neural net, y-hats
            y_hats = np.argmax(neural_net.calculate_y_hat(x_test), axis=1)

        ###############################################################################################################
        # build a neural network with the encoder from an autoencoder as the input and encoding layers
        elif algorithm == 4:

            # we need to cast the x- and y-vals as a new type for nn training
            # do it here
            x_vals_nn = dataframe
            for i in range(0, len(x_vals_nn.columns)):
                if i != response_col:
                    x_vals_nn.iloc[i].values[:] = 0

            y_vals_nn = dataframe
            for i in range(0, len(y_vals_nn.columns)):
                if i == response_col:
                    y_vals_nn.iloc[i].values[:] = 0

            # now declare these vals as np array
            # this ensures they are all of identical data type and math-operable
            x_data_train = np.array(x_vals_nn)
            y_data_train = np.array(y_vals_nn)
            x_vals_ae = dataframe

            for i in range(0, len(x_vals_ae.columns)):
                if i != response_col:
                    x_vals_ae.iloc[i].values[:] = 0

            x_vals_ae = np.array(x_vals)

            # define training, testing split
            # note that this is different from the bin list up above, particularly altered for nn training
            x_train, x_test, y_train, y_test = train_test_split(x_data_train, y_data_train, test_size=0.4)

            # define # of samples and hidden nodes within hidden layer
            n_hidden = x_vals_ae.shape[1]

            # init autoencoder
            autoencoder = Autoencoder(x_vals_ae, is_classify, (x_test, y_test))
            autoencoder.train_autoencoder(x_data=x_vals_ae, epochs=2000, batch_size=10)

            # autoencoder trained, now remove output layer so we just have the encoder
            autoencoder.remove_output_layer()

            # recast the encoder as the first two layers of a neural net - one input layer, one hidden layer
            neural_net = autoencoder.autoencoder

            # add one more hidden and output layer
            neural_net.add_layer(DenseLayer(n_hidden, input_shape=(x_vals_ae.shape[1],)))
            neural_net.add_layer(ActivationLayer('sigmoid'))                            # not really a layer
            neural_net.add_layer(DenseLayer(n_hidden, input_shape=(x_vals_ae.shape[1],)))
            neural_net.add_layer(ActivationLayer('sigmoid'))                            # not really a layer
            neural_net.add_layer(DenseLayer(n_hidden, input_shape=(x_vals_ae.shape[1],)))
            if is_classify:
                neural_net.add_layer(ActivationLayer('softmax'))                        # not really a layer
            else:
                neural_net.add_layer(ActivationLayer('relu'))                           # not really a layer

            # print model summary to console
            neural_net.model_analytics(title='Neural Network')

            # neural net defined, encoder trained
            # now train both of them together
            train_error, val_error = neural_net.fit_neural_net(x_test, y_test, epochs=25, batch_size=20)
            n = len(train_error)

            if config.DEMO_MODE:
            # plot the training and testing losses for a visual and for quality control check
                training, = plt.plot(range(n), train_error, label='training error')
                validation, = plt.plot(range(n), val_error, label='validation error')
                plt.legend(handles=[training, validation])
                plt.title('Autoencoder + Neural Network Error Plot - Abalone')
                plt.ylabel('error')
                plt.xlabel('epoch #')
                plt.show()

            # perform testing on the neural net to get our accuracy of results
            _, accuracy = neural_net.test_single_batch(x_train, y_train)
            # print(f'accuracy: {get_accuracy(accuracy)}')

            # get predictions from our neural net, y-hats
            y_hats = np.argmax(neural_net.calculate_y_hat(x_test), axis=1)



if __name__ == '__main__':

    # from Lab4.Neural_Network import main_demo
    # main_demo.main()
    # exit()


    # check path i/o directory and move it to the correct place if necessary
    data_processing.rename_data_files(config.IO_DIRECTORY)

    # AUTOENCODER - TRAINED NEURAL NETWORK
    ###############################################################################################
    ###############################################################################################

    filename: str = 'abalone(1)_DATA.csv'

    # read each value, check for missing / erroneous data, and import the data into a pandas dataframe
    dataframe: pd.DataFrame = data_processing.import_dataset(filename)
    # check for categorical columns that need encoded
    encoding_cols: list = data_processing.preprocess_dataframe(dataframe, dataframe.shape[1])
    # if any columns needed encoded, a new dataframe is returned with categorical values encoded nominally
    encoded_dataframe: pd.DataFrame = encoding.encode_nominal_data(dataframe, encoding_cols)

    dataframe = encoded_dataframe.drop([0,1,2], 1)
    # call the driver algorithm that sets up KFCV over a dataset for each decision tree algo
    perform_k_fold_cross_validation(algorithm=4,
                                    num_folds=2,
                                    dataframe=dataframe,
                                    is_classify=False,
                                    explanatory_col=4,
                                    response_col=0)
    ###############################################################################################


    # NEURAL NETWORK
    ###############################################################################################
    ###############################################################################################

    # filename: str = 'forestfires(1)_DATA.csv'
    #
    # # read each value, check for missing / erroneous data, and import the data into a pandas dataframe
    # dataframe: pd.DataFrame = data_processing.import_dataset(filename)
    # # check for categorical columns that need encoded
    # encoding_cols: list = data_processing.preprocess_dataframe(dataframe, dataframe.shape[1])
    # # if any columns needed encoded, a new dataframe is returned with categorical values encoded nominally
    # encoded_dataframe: pd.DataFrame = encoding.encode_nominal_data(dataframe, encoding_cols)
    #
    # # dataframe = encoded_dataframe.drop([0,1,2], 1)
    # # call the driver algorithm that sets up KFCV over a dataset for each decision tree algo
    # perform_k_fold_cross_validation(algorithm=3,
    #                                 num_folds=2,
    #                                 dataframe=encoded_dataframe,
    #                                 is_classify=False,
    #                                 explanatory_col=4,
    #                                 response_col=9)
    ###############################################################################################


    # AUTOENCODER
    ###############################################################################################
    ##############################################################################################
    # filename: str = 'breast-cancer-wisconsin(1)_DATA.csv'
    #
    # # read each value, check for missing / erroneous data, and import the data into a pandas dataframe
    # dataframe: pd.DataFrame = data_processing.import_dataset(filename)
    # # check for categorical columns that need encoded
    # encoding_cols: list = data_processing.preprocess_dataframe(dataframe, dataframe.shape[1])
    # # if any columns needed encoded, a new dataframe is returned with categorical values encoded nominally
    # encoded_dataframe: pd.DataFrame = encoding.encode_nominal_data(dataframe, encoding_cols)
    #
    # dataframe = dataframe.drop([0,1], 1)
    # # call the driver algorithm that sets up KFCV over a dataset for each decision tree algo
    # perform_k_fold_cross_validation(algorithm=2,
    #                                 num_folds=2,
    #                                 dataframe=dataframe,
    #                                 is_classify=True,
    #                                 explanatory_col=5,
    #                                 response_col=0)


    # LINEAR NETWORK
    ###############################################################################################
    ###############################################################################################

    # filename: str = 'machine(1)_DATA.csv'
    #
    # # read each value, check for missing / erroneous data, and import the data into a pandas dataframe
    # dataframe: pd.DataFrame = data_processing.import_dataset(filename)
    # # check for categorical columns that need encoded
    # encoding_cols: list = data_processing.preprocess_dataframe(dataframe, dataframe.shape[1])
    # # if any columns needed encoded, a new dataframe is returned with categorical values encoded nominally
    # encoded_dataframe: pd.DataFrame = encoding.encode_nominal_data(dataframe, encoding_cols)
    # dataframe = standardize.standardize_data_via_minimax(encoded_dataframe, encoded_dataframe.columns)
    #
    # # call the driver algorithm that sets up KFCV over a dataset for each decision tree algo
    # perform_k_fold_cross_validation(algorithm=0,
    #                                 num_folds=5,
    #                                 dataframe=dataframe,
    #                                 is_classify=True,
    #                                 explanatory_col=4,
    #                                 response_col=9)
    ###############################################################################################


    # LOGISTIC REGRESSION
    ###############################################################################################
    ###############################################################################################
    # filename: str = 'breast-cancer-wisconsin(2)_DATA.csv'
    #
    # # read each value, check for missing / erroneous data, and import the data into a pandas dataframe
    # dataframe: pd.DataFrame = data_processing.import_dataset(filename)
    # # check for categorical columns that need encoded
    # encoding_cols: list = data_processing.preprocess_dataframe(dataframe, dataframe.shape[1])
    # # if any columns needed encoded, a new dataframe is returned with categorical values encoded nominally
    # encoded_dataframe: pd.DataFrame = encoding.encode_nominal_data(dataframe, encoding_cols)
    #
    # # call the driver algorithm that sets up KFCV over a dataset for each decision tree algo
    # perform_k_fold_cross_validation(algorithm=1,
    #                                 num_folds=5,
    #                                 dataframe=dataframe,
    #                                 is_classify=False,
    #                                 explanatory_col=2,
    #                                 response_col=9)
    ##############################################################################################




