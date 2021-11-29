# linear_network.py
# Kordel France
########################################################################################################################
# This file establishes the class and driver function for a linear discriminator network.
########################################################################################################################


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


from Lab4.config import DEBUG_MODE


def fit_linear_network(x0_data_train: list,
                       x0_data_test: list,
                       x1_data_train: list,
                       x1_data_test: list,
                       y0_data_train: list,
                       y0_data_test: list,
                       y1_data_train: list,
                       y1_data_test: list):
    """
    Fits a linear discriminator network between two sets of data
    :param x0_data_train - list of training values
    :param x1_data_train - list of training values
    :param y0_data_train - list of training values
    :param y1_data_train - list of training values
    :param x0_data_test - list of testing values
    :param x1_data_test - list of testing values
    :param y0_data_test - list of testing values
    :param y1_data_test - list of testing values
    """

    # build classifier
    print((x0_data_test))
    print((x1_data_test))
    print((y0_data_test))
    print((y1_data_test))
    slope0, intercept0 = compute_linear_regression_equation(x0_data_train, y0_data_train)
    slope1, intercept1 = compute_linear_regression_equation(x1_data_train, y1_data_train)
    intersect_x, intersect_y = compute_intersection_point(slope0, intercept0, slope1, intercept1)
    # theta: float = math.atan((slope0 - slope1) / (1 + (slope0 * slope1)))
    theta: float = (slope0 + slope1) / 2
    beta: float = intersect_y - (theta * intersect_x)
    alpha: float = theta * 0.5

    # test classifier
    x0_acc_list: list = []
    x0_acc_count: float = 0.0
    x1_acc_list: list = []
    x1_acc_count: float = 0.0

    for index_i in range(0, len(x0_data_test)):
        y_hat = theta * x0_data_test[index_i] + beta
        if y_hat < y0_data_test[index_i]:
            x0_acc_list.append(0)
        else:
            x0_acc_list.append(1)
            x0_acc_count += 1

    for index_i in range(0, len(x1_data_test)):
        y_hat = theta * x1_data_test[index_i] + beta
        if y_hat > y1_data_test[index_i]:
            x1_acc_list.append(0)
        else:
            x1_acc_list.append(1)
            x1_acc_count += 1

    # compute accuracy of the network
    x0_accuracy: float = x0_acc_count / len(x0_data_train)
    x1_accuracy: float = x1_acc_count / len(x1_data_train)

    # if DEBUG_MODE:
    print(f'x0 accuracy: {x0_accuracy}')
    print(f'x1 accuracy: {x1_accuracy}')

    theta0_min = min(x1_data_train)
    theta1_min = min(x1_data_test)
    theta_min = min(theta0_min, theta1_min)

    theta0_max = max(x1_data_train)
    theta1_max = max(x1_data_test)
    theta_max = max(theta0_max, theta1_max)

    step_size: float = (theta_max - theta_min) / float(len(x1_data_train) + len(x1_data_test))
    x_thetas: list = []
    y_thetas: list = []

    # compute slope of discrimnator
    for index_i in range(0, len(x1_data_train) + len(x1_data_test)):
        x_theta: float = index_i * step_size
        y_theta: float = theta * x_theta + beta
        x_thetas.append(x_theta)
        y_thetas.append(y_theta)

    # plot line and present
    if DEBUG_MODE:
        plt.scatter(x0_data_train, y0_data_train, color='blue')
        plt.scatter(x0_data_test, y0_data_test, color='purple')
        plt.scatter(x1_data_train, y1_data_train, color='orange')
        plt.scatter(x1_data_test, y1_data_test, color='red')
        plt.scatter(x_thetas, y_thetas, color='black')

        plt.title("linear network")
        plt.legend()
        plt.show()

    # final results
    print(f'\nfeature0:\n\tslope: {slope0}, intercept: {intercept0}')
    print(f'\nfeature1:\n\tslope: {slope1}, intercept: {intercept1}')
    print(f'\nclassifier:\n\tslope: {theta}, intercept: {beta}')
    return theta, beta


def compute_intersection_point(slope0: float, intercept0: float, slope1: float, intercept1: float):
    """
    Computes the intersection point between new lines of form y=mx+b
    :param slope0: float - m of equation 1
    :param intercept0: float - intercept of equation 1
    :param slope1: float - m of equation 2
    :param intercept1: float - intercept of equation 2
    """

    # define a line between two points
    def line(p0, p1):
        a = p0[1] - p1[1]
        b = p1[0] - p0[0]
        c = p0[0] * p1[1] - p1[0] * p0[1]
        return a, b, -c

    # establish vertices of each end of line
    line0_p0_x: float = 0.0
    line0_p0_y: float = slope0 * line0_p0_x + intercept0
    line0_p1_x: float = 1.0
    line0_p1_y: float = slope0 * line0_p1_x + intercept0
    line1_p0_x: float = 0.0
    line1_p0_y: float = slope1 * line1_p0_x + intercept1
    line1_p1_x: float = 1.0
    line1_p1_y: float = slope1 * line1_p1_x + intercept1

    # establish line
    line0: line = line([line0_p0_x, line0_p0_y], [line0_p1_x, line0_p1_y])
    line1: line = line([line1_p0_x, line1_p0_y], [line1_p1_x, line1_p1_y])

    # compute deltas and intersection points
    d = line0[0] * line1[1] - line0[1] * line1[0]
    dx = line0[2] * line1[1] - line0[1] * line1[0]
    dy = line0[0] * line1[2] - line0[2] * line1[0]

    if d != 0:
        x = dx / d
        y = dy / d
        return x, y





def compute_linear_regression_equation(x_data: list, y_data: list):
    """
    Fits a linear regression line to the specified data in the form y = mx + b
    :param dataframe: pd.DataFrame - (the dataset to fit regression line to)
    :param x_column: int - (explanatory variable)
    :param y_vals: int - (response variable)
    """

    x_vals: list = x_data
    y_vals: list = y_data

    # print fit status to the screen
    print('fitting linear regression equation')
    print(f' x values: {x_vals}')
    print(f' y values: {y_vals}')

    # initialize statistics
    n = len(y_vals)
    sum_xy = 0
    sum_x = 0
    sum_y = 0
    sum_x2 = 0

    # build statistics
    for index in range(0, n):
        x_val: float = float(x_vals[index])
        y_val: float = float(y_vals[index])
        sum_x += x_val
        sum_y += y_val
        sum_xy += (x_val * y_val)
        sum_x2 += (x_val ** 2)

    m = (n * sum_xy) - (sum_x * sum_y)
    m /= ((n * sum_x2) - (sum_x ** 2))
    b = sum_y - (m * sum_x)
    b /= n

    # build the estimates of the response variable
    y_hats = []
    for index in range(0, n):
        y_hat: float = float(m) * float(x_vals[index])
        y_hat += float(b)
        y_hats.append(y_hat)

    # compute mean squared error
    mse: float = 0
    for index in range(0, n):
        error = y_vals[index] - y_hats[index]
        error **= 2
        mse += error

    mse /= n

    # equation constructed, now calculate correlation
    x_bar: float = sum_x / n
    y_bar: float = sum_y / n
    r: float = 0
    r_numerator: float = 0.0
    r_denominator: float = 0.0
    r_denom0: float = 0.0
    r_denom1: float = 0.0

    for index in range(0, n):
        r_numerator += ((x_vals[index] - x_bar) * (y_vals[index] - y_bar))
        r_denom0 += (x_vals[index] - x_bar) ** 2
        r_denom1 += (y_vals[index] - y_bar) ** 2

    r_denominator += (r_denom0 * r_denom1)
    r_denominator **= 0.5
    if r_denominator != 0.0:
        r += r_numerator / r_denominator
        r *= 100

    # correlation calculated, now display final regression equation
    equation: str = f'y={m}x+{b}'

    # plot the values to show the linear regression equation
    if DEBUG_MODE:
        plt.scatter(x_vals, y_vals)
        plt.plot(x_vals, y_hats)
        plt.title("linear regression: y=mx+b")
        plt.legend()
        plt.show()

    print(f'\tm: {m}\tb:{b}')
    print(f'\tequation: {equation} with correlation: {r}%')
    return m, b

