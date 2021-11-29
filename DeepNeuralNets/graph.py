# graph.py
# Kordel France
########################################################################################################################
# This file contains a function to graph and display algorithmic efficiency data
########################################################################################################################

import matplotlib.pyplot as plt
from Lab3.config import DEBUG_MODE


def graph_runtime_data(x_vals, y_vals, scatter: bool, label: str):
    """
    Graphs a series of x values and y values with the specified label and an option for a scatter plot.
    :param x_vals: explanatory variable.
    :param y_vals: response variable.
    :param scatter: bool - set True if plot should be a scatter plot
    :param label: str - the label of the graph describing what is being plotted
    """
    # print values to screen to veriy they are correct
    if DEBUG_MODE:
        print(f'x_vals: {x_vals}')
        print(f'y_vals: {y_vals}')

    # set up the graph and plot the data
    if not scatter:
        plt.plot(x_vals, y_vals)
    else:
        plt.scatter(x_vals, y_vals)
    plt.title(label)
    plt.legend()
    plt.show()