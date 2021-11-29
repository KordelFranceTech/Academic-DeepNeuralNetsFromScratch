
from Lab4.Neural_Network.neural_net import NeuralNet
from Lab4.Neural_Network.optimizer_function import Adam
from Lab4.Neural_Network.loss_function import CrossEntropyLoss
from Lab4.Neural_Network.layer import DenseLayer, ActivationLayer
from Lab4.Neural_Network.utilities import train_test_split, to_categorical, Plot

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


def main():
    optimizer = Adam()

    data = datasets.load_digits()
    x_input = data.data
    y_target = data.target
    y_target = to_categorical(y_target.astype('int'))

    n_samples, n_features = x_input.shape
    n_hidden = 10

    x_train, x_test, y_train, y_test = train_test_split(x_input, y_target, test_size=0.4, seed=1)

    mlp = NeuralNet(optimizer_function=optimizer, loss_function=CrossEntropyLoss, val_data=(x_test, y_test))
    mlp.add_layer(DenseLayer(n_hidden, input_shape=(n_features,)))
    mlp.add_layer(ActivationLayer('leaky_relu'))
    mlp.add_layer(DenseLayer(10))
    mlp.add_layer(ActivationLayer('softmax'))
    print()
    mlp.model_analytics(title='MLP')

    train_err, val_err = mlp.fit_neural_net(x_train, y_train, epochs=50, batch_size=256)
    n = len(train_err)
    training, = plt.plot(range(n), train_err, label='training error')
    validation, = plt.plot(range(n), val_err, label='validation error')
    plt.legend(handles = [training, validation])
    plt.title('error plot')
    plt.ylabel('error')
    plt.xlabel('iterations')
    plt.show()

    _, accuracy = mlp.test_single_batch(x_test, y_test)
    print(f'accuracy: {accuracy}')

    y_hats = np.argmax(mlp.calculate_y_hat(x_test), axis=1)
    Plot().plot_in_2d(x_test, y_hats, title='multilayer perceptron', accuracy=accuracy, legend_labels=range(10))


if __name__ == '__main__':
    main()

