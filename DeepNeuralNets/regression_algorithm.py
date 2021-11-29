# regression_algorithm.py
# Kordel France
########################################################################################################################
# This file establishes helper functions used for the facilitation of different regression algorithms.
########################################################################################################################


import matplotlib.pyplot as plt
import Lab4.helpers
from Lab4.config import DEBUG_MODE
from Lab4.Metric import Metric


def fit_linear_regression_equation(x_vals, y_vals):
    """
    Fits a linear regression line to the specified data in the form y = mx + b
    :param x_vals: [float] - a list of x values (explanatory variable)
    :param y_vals: [float] - a list of y values (response variable)
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
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
        Lab1.helpers.COST_COUNTER += 1

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
        Lab1.helpers.COST_COUNTER += 1

    # compute mean squared error
    mse: float = 0
    for index in range(0, n):
        error = y_vals[index] - y_hats[index]
        error **= 2
        mse += error
        Lab1.helpers.COST_COUNTER += 1

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
        Lab1.helpers.COST_COUNTER += 1

    r_denominator += (r_denom0 * r_denom1)
    r_denominator **= 0.5
    if r_denominator != 0.0:
        r += r_numerator / r_denominator
        r *= 100
    equation: str = f'y={m}x+{b}'

    # option to graph in DEBUG_MODE
    if DEBUG_MODE:
        plt.scatter(x_vals, y_vals)
        plt.plot(x_vals, y_hats)
        plt.title("linear regression: y=mx+b")
        plt.legend()
        plt.show()
        print(f'\tm: {m}\tb:{b}')
        print(f'\tequation: {equation} with correlation: {r}%')

    # return the equation, mse, both coefficients for slope & intercept, and the correlation
    metric = Metric(explanatory_col=0,
                    response_col=0,
                    is_regression=True,
                    regresion_option=0,
                    equation=equation,
                    coeff0=m,
                    coeff1=b,
                    correlation_train=r,
                    mse_train=mse,
                    class_score_train=0,
                    class_value_train='',
                    f1_score_train=0,
                    correlation_test=0,
                    mse_test=0,
                    class_score_test=0,
                    class_value_test='',
                    f1_score_test=0,
                    correlation_val=0,
                    mse_val=0.0,
                    class_score_val=0,
                    class_value_val='',
                    f1_score_val=0)
    return metric

import numpy as np
def fit_inverse_regression_equation(xVals, yVals):
    """
    Fits an inverse regression line to the specified data in the form y = a + b/x
    :param x_vals: [float] - a list of x values (explanatory variable)
    :param y_vals: [float] - a list of y values (response variable)
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print('fitting inverse regression equation')
        print(f' x values: {xVals}')
        print(f' y values: {yVals}')

    n = len(xVals)
    sumXinv2 = 0.0
    sumXinvY = 0.0
    xInvBar = 0.0
    sumY2 = 0.0
    yBar = 0.0

    for i in range(0, n - 1):
        sumXinvY += (1 / xVals[i]) * yVals[i]
    for i in range(0, n - 1):
        sumXinv2 += ((1 / xVals[i]) * (1 / xVals[i]))
    for i in range(0, n - 1):
        sumY2 += pow(yVals[i], 2)
    for i in range(0, n - 1):
        xInvBar += (1 / xVals[i])

    xInvBar /= n
    xInvBar2 = pow(xInvBar, 2)
    for i in range(0, n - 1):
        yBar += yVals[i]

    yBar /= n
    sxx = sumXinv2 - (n * pow(xInvBar, 2))
    syy = sumY2 - (n * pow(yBar, 2))
    sxy = sumXinvY - (n * xInvBar * yBar)
    b = sxy / sxx
    a = yBar - (b * xInvBar)
    r = sxy / (np.sqrt(sxx) * np.sqrt(syy))
    x = np.array(xVals)
    y = np.array(yVals)

    if DEBUG_MODE:
        print('sxx: ' + str(sxx))
        print('syy: ' + str(syy))
        print('sxy: ' + str(sxy))
        print('b: ' + str(b))
        print('a: ' + str(a))
        print('r: ' + str(r))

    def inverse_law(x, a, b):
        return a + (b / x)

    shouldGraph = True
    yHats = []

    for xPrime in x:
        yHats.append(inverse_law(xPrime, a, b))

    # compute mean squared error
    mse: float = 0
    for index in range(0, n):
        error = yVals[index] - yHats[index]
        error **= 2
        mse += error
    mse /= n

    if DEBUG_MODE:
        plt.scatter(xVals, yVals)
        plt.plot(x, yHats)
        plt.title("Time vs Current ax^b CHECKED")
        plt.xlabel('Time')
        plt.ylabel('Current')
        plt.legend()
        plt.show()
        print(str(a))
        print(str(b))
        print(f'The checked equation of regression line is y={a} + {b}/x with correlation {100 * r}%')

    # return the equation, mse, both coefficients for slope & intercept, and the correlation
    equation: str = f'y={a} + {b}/x'
    metric = Metric(explanatory_col=0,
                    response_col=0,
                    is_regression=True,
                    regresion_option=1,
                    equation=equation,
                    coeff0=a,
                    coeff1=b,
                    correlation_test=r,
                    mse_test=mse,
                    class_score_test=0,
                    class_value_test='',
                    f1_score_test=0,
                    correlation_train=0,
                    mse_train=0,
                    class_score_train=0,
                    class_value_train='',
                    f1_score_train=0,
                    correlation_val=0,
                    mse_val=0.0,
                    class_score_val=0,
                    class_value_val='',
                    f1_score_val=0)
    return metric


def fit_exponential_regression_equation(xxVals, yyVals):
    """
    Fits a linear regression line to the specified data in the form y = ab^x
    :param x_vals: [float] - a list of x values (explanatory variable)
    :param y_vals: [float] - a list of y values (response variable)
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print('fitting exponential regression equation')
        print(f' x values: {xxVals}')
        print(f' y values: {yyVals}')

    n = len(xxVals)
    sumX = 0.0
    sumX2 = 0.0
    sumXlnY = 0.0
    sumLnxLny = 0.0
    sumLnx = 0.0
    sumLny = 0.0
    sumLnx2 = 0.0
    sumLny2 = 0.0

    for i in range(0, n - 1):
        sx0 = xxVals[i]
        sumX += sx0
    for i in range(0, n - 1):
        sx20 = xxVals[i] * xxVals[i]
        sumX2 += sx20
    for i in range(0, n - 1):
        t = xxVals[i]
        v = np.log(yyVals[i])
        sumXlnY += (t * v)
    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        lny = np.log(yyVals[i])
        sumLnxLny += (lnx * lny)
    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        sumLnx += lnx
    for i in range(0, n - 1):
        lny = np.log(yyVals[i])
        sumLny += lny
    for i in range(0, n - 1):
        lny = np.log(yyVals[i])
        sumLny2 += (lny * lny)
    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        sumLnx2 += (lnx * lnx)

    lnxBar = sumLnx / n
    lnyBar = sumLny / n
    xBar = sumX / n
    sxx = sumX2 - (n * xBar * xBar)
    syy = sumLny2 - (n * (lnyBar ** 2))
    sxy = sumXlnY - (n * xBar * lnyBar)
    b = pow(np.e, sxy / sxx)
    a = pow(np.e, lnyBar - (xBar * np.log(b)))
    r = sxy / (pow(sxx, 0.5) * pow(syy, 0.5))
    xx = np.array(xxVals)
    yy = np.array(yyVals)

    if DEBUG_MODE:
        print('sxx: ' + str(sxx))
        print('syy: ' + str(syy))
        print('sxy: ' + str(sxy))
        print('b: ' + str(b))
        print('a: ' + str(a))
        print('r: ' + str(r))

    def exponent_law(x, a, b):
        return np.power(a * b, x)

    shouldGraph = True
    yHats = []

    for xPrime in xx:
        yHats.append(exponent_law(xPrime, a, b))

    # compute mean squared error
    mse: float = 0
    for index in range(0, n):
        error = yyVals[index] - yHats[index]
        error **= 2
        mse += error
    mse /= n

    if DEBUG_MODE:
        plt.scatter(xx, yHats)
        # plt.plot(x, exponent_law(x, a, b), 'r-')
        plt.scatter(xxVals ,yyVals ,label='Time vs Current')
        plt.title("Time vs Current ab^x CHECKED")
        plt.xlabel('Time')
        plt.ylabel('Current')
        plt.legend()
        plt.show()
        print(str(a))
        print(str(b))
        print(f'The checked equation of regression line is y={a}*{b}^x with correlation {100 * r}%')

    # return the equation, mse, both coefficients for slope & intercept, and the correlation
    equation: str = f'y={a}*{b}^x'
    metric = Metric(explanatory_col=0,
                    response_col=0,
                    is_regression=True,
                    regresion_option=2,
                    equation=equation,
                    coeff0=a,
                    coeff1=b,
                    correlation_test=r,
                    mse_test=mse,
                    class_score_test=0,
                    class_value_test='',
                    f1_score_test=0,
                    correlation_train=0,
                    mse_train=0,
                    class_score_train=0,
                    class_value_train='',
                    f1_score_train=0,
                    correlation_val=0,
                    mse_val=0.0,
                    class_score_val=0,
                    class_value_val='',
                    f1_score_val=0)
    return metric


def fit_euler_exponential_regression_equation(xxVals, yyVals):
    """
    Fits a euler exponential regression line to the specified data in the form y = ae^bx
    :param x_vals: [float] - a list of x values (explanatory variable)
    :param y_vals: [float] - a list of y values (response variable)
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print('fitting euler exponential regression equation')
        print(f' x values: {xxVals}')
        print(f' y values: {yyVals}')

    n = len(xxVals)
    sumX = 0.0
    sumX2 = 0.0
    sumXlnY = 0.0
    sumLnxLny = 0.0
    sumLnx = 0.0
    sumLny = 0.0
    sumLnx2 = 0.0
    sumLny2 = 0.0

    for i in range(0, n - 1):
        sx0 = xxVals[i]
        sumX += sx0
    for i in range(0, n - 1):
        sx20 = pow(xxVals[i], 2)
        sumX2 += sx20
    for i in range(0, n - 1):
        t = xxVals[i]
        v = np.log(yyVals[i])
        sumXlnY += (t * v)
    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        lny = np.log(yyVals[i])
        sumLnxLny += (lnx * lny)
    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        sumLnx += lnx
    for i in range(0, n - 1):
        lny = np.log(yyVals[i])
        sumLny += lny
    for i in range(0, n - 1):
        lny = np.log(yyVals[i])
        sumLny2 += (lny * lny)
    for i in range(0, n - 1):
        lnx = np.log(yyVals[i])
        sumLnx2 += (lnx * lnx)

    lnxBar = sumLnx / n
    lnyBar = sumLny / n
    xBar = sumX / n
    sxx = sumX2 - (n * xBar * xBar)
    syy = sumLny2 - (n * (lnyBar ** 2))
    sxy = sumXlnY - (n * xBar * lnyBar)
    b = pow(np.e, sxy / sxx)
    a = pow(np.e, lnyBar - (xBar * b))
    r = sxy / (pow(sxx, 0.5) * pow(syy, 0.5))
    xx = np.array(xxVals)
    yy = np.array(yyVals)

    if DEBUG_MODE:
        print('sxx: ' + str(sxx))
        print('syy: ' + str(syy))
        print('sxy: ' + str(sxy))
        print('b: ' + str(b))
        print('a: ' + str(a))
        print('r: ' + str(r))

    def euler_exponent_law(xx, a, b):
        return a * np.power(np.e, b * xx)

    shouldGraph = True
    yHats = []

    for xPrime in xx:
        yHats.append(euler_exponent_law(xPrime, a, b))

    # compute mean squared error
    mse: float = 0
    for index in range(0, n):
        error = yyVals[index] - yHats[index]
        error **= 2
        mse += error
    mse /= n

    if DEBUG_MODE:
        print('yHats: ' + str(yHats))
        # plt.scatter(xx, yHats)
        plt.plot(xx, euler_exponent_law(xx, a, b), 'r-')
        plt.scatter(xx ,yy ,label='Time vs Current')
        plt.title("Time vs Current ae^bx CHECKED")
        plt.xlabel('Time')
        plt.ylabel('Current')
        plt.legend()
        plt.show()
        print(str(a))
        print(str(b))
        print(f'The checked equation of regression line is y={a}*e^({b}*x) with correlation {100 * r}%')

    # return the equation, mse, both coefficients for slope & intercept, and the correlation
    equation: str = f'y={a}*e^({b}*x)'
    metric = Metric(explanatory_col=0,
                    response_col=0,
                    is_regression=True,
                    regresion_option=3,
                    equation=equation,
                    coeff0=a,
                    coeff1=b,
                    correlation_test=r,
                    mse_test=mse,
                    class_score_test=0,
                    class_value_test='',
                    f1_score_test=0,
                    correlation_train=0,
                    mse_train=0,
                    class_score_train=0,
                    class_value_train='',
                    f1_score_train=0,
                    correlation_val=0,
                    mse_val=0.0,
                    class_score_val=0,
                    class_value_val='',
                    f1_score_val=0)
    return metric


def fit_power_regression_equation(xxVals, yyVals):
    """
    Fits a linear regression line to the specified data in the form y = ax^b
    :param x_vals: [float] - a list of x values (explanatory variable)
    :param y_vals: [float] - a list of y values (response variable)
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print('fitting power regression equation')
        print(f' x values: {xxVals}')
        print(f' y values: {yyVals}')

    n = len(xxVals)
    sumLnxLny = 0.0
    sumLnx = 0.0
    sumLny = 0.0
    sumLnx2 = 0.0
    sumLny2 = 0.0

    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        lny = np.log(yyVals[i])
        sumLnxLny += (lnx * lny)
    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        sumLnx += lnx
    for i in range(0, n - 1):
        lny = np.log(yyVals[i])
        sumLny += lny
    for i in range(0, n - 1):
        lny = np.log(yyVals[i])
        sumLny2 += (lny * lny)
    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        sumLnx2 += (lnx * lnx)

    lnxBar = sumLnx / n
    lnyBar = sumLny / n
    sxx = sumLnx2 - (n * (lnxBar ** 2))
    syy = sumLny2 - (n * (lnyBar ** 2))
    sxy = sumLnxLny - (n * lnxBar * lnyBar)
    b = sxy / sxx
    a = pow(np.e, lnyBar - (b * lnxBar))
    r = sxy / (np.sqrt(sxx) * np.sqrt(syy))
    xx = np.array(xxVals)
    yy = np.array(yyVals)

    if DEBUG_MODE:
        print('sxx: ' + str(sxx))
        print('syy: ' + str(syy))
        print('sxy: ' + str(sxy))
        print('b: ' + str(b))
        print('a: ' + str(a))
        print('r: ' + str(r))

    def power_law(xx, a, b):
        return a * np.power(xx, b)

    shouldGraph = True
    yHats = []

    for xPrime in xx:
        yHats.append(power_law(xPrime, a, b))

    # compute mean squared error
    mse: float = 0
    for index in range(0, n):
        error = yyVals[index] - yHats[index]
        error **= 2
        mse += error
    mse /= n

    if DEBUG_MODE:
        print('yHats: ' + str(yHats))
        plt.scatter(xx, yHats)
        # plt.plot(xx, power_law(x, a, b), 'r-')
        plt.scatter(xx ,yy ,label='Time vs Current')
        plt.title("Time vs Current ax^b CHECKED")
        plt.xlabel('Time')
        plt.ylabel('Current')
        plt.legend()
        plt.show()
        print(str(a))
        print(str(b))
        print(f'The checked equation of regression line is y={a}x^{b} with correlation {100 * r}%')

    # return the equation, mse, both coefficients for slope & intercept, and the correlation
    equation: str = f'y={a}x^{b}'
    metric = Metric(explanatory_col=0,
                    response_col=0,
                    is_regression=True,
                    regresion_option=4,
                    equation=equation,
                    coeff0=a,
                    coeff1=b,
                    correlation_test=r,
                    mse_test=mse,
                    class_score_test=0,
                    class_value_test='',
                    f1_score_test=0,
                    correlation_train=0,
                    mse_train=0,
                    class_score_train=0,
                    class_value_train='',
                    f1_score_train=0,
                    correlation_val=0,
                    mse_val=0.0,
                    class_score_val=0,
                    class_value_val='',
                    f1_score_val=0)
    return metric



def eval_linear_regression_equation(x_vals, y_vals, metric: Metric):
    """
    Fits a linear regression line to the specified data in the form y = mx + b
    :param x_vals: [float] - a list of x values (explanatory variable)
    :param y_vals: [float] - a list of y values (response variable)
    :param metric: Metric - an object containing statistical metrics and performance info
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print('fitting linear regression equation')
        print(f' x values: {x_vals}')
        print(f' y values: {y_vals}')

    # initialize statistics
    n = len(y_vals)
    sum_x = 0
    sum_y = 0

    # build the estimates of the response variable
    y_hats = []
    for index in range(0, n):
        y_hat: float = float(metric.coeff0) * float(x_vals[index])
        y_hat += float(metric.coeff1)
        y_hats.append(y_hat)
        Lab1.helpers.COST_COUNTER += 1

    # compute mean squared error
    mse: float = 0
    for index in range(0, n):
        error = y_vals[index] - y_hats[index]
        error **= 2
        mse += error
        Lab1.helpers.COST_COUNTER += 1

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
        Lab1.helpers.COST_COUNTER += 1

    r_denominator += (r_denom0 * r_denom1)
    r_denominator **= 0.5
    if r_denominator != 0.0:
        r += r_numerator / r_denominator
        r *= 100

    # option to graph in DEBUG_MODE
    if DEBUG_MODE:
        print(f'\tm: {metric.coeff0}\tb:{metric.coeff1}')
        print(f'\tequation: {metric.equation} with correlation: {r}%')

    # return the equation, mse, both coefficients for slope & intercept, and the correlation
    if metric.correlation_test <= 0:
        metric.correlation_test = r
        metric.mse_test = mse
    else:
        metric.correlation_val = r
        metric.mse_val = mse

    return metric


def eval_inverse_regression_equation(xVals, yVals, metric: Metric):
    """
    Fits an inverse regression line to the specified data in the form y = a + b/x
    :param x_vals: [float] - a list of x values (explanatory variable)
    :param y_vals: [float] - a list of y values (response variable)
    :param metric: Metric - an object containing statistical metrics and performance info
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print('fitting inverse regression equation')
        print(f' x values: {xVals}')
        print(f' y values: {yVals}')

    n = len(xVals)
    sumXinv2 = 0.0
    sumXinvY = 0.0
    xInvBar = 0.0
    sumY2 = 0.0
    yBar = 0.0

    for i in range(0, n - 1):
        sumXinvY += (1 / xVals[i]) * yVals[i]
    for i in range(0, n - 1):
        sumXinv2 += ((1 / xVals[i]) * (1 / xVals[i]))
    for i in range(0, n - 1):
        sumY2 += pow(yVals[i], 2)
    for i in range(0, n - 1):
        xInvBar += (1 / xVals[i])

    xInvBar /= n
    xInvBar2 = pow(xInvBar, 2)
    for i in range(0, n - 1):
        yBar += yVals[i]

    yBar /= n
    sxx = sumXinv2 - (n * pow(xInvBar, 2))
    syy = sumY2 - (n * pow(yBar, 2))
    sxy = sumXinvY - (n * xInvBar * yBar)
    r = sxy / (np.sqrt(sxx) * np.sqrt(syy))
    x = np.array(xVals)
    y = np.array(yVals)

    if DEBUG_MODE:
        print('sxx: ' + str(sxx))
        print('syy: ' + str(syy))
        print('sxy: ' + str(sxy))
        print('b: ' + str(metric.coeff1))
        print('a: ' + str(metric.coeff0))
        print('r: ' + str(r))

    def inverse_law(x, a, b):
        return a + (b / x)

    yHats = []
    for xPrime in x:
        yHats.append(inverse_law(xPrime, metric.coeff0, metric.coeff1))

    # compute mean squared error
    mse: float = 0
    for index in range(0, n):
        error = yVals[index] - yHats[index]
        error **= 2
        mse += error
    mse /= n

    if DEBUG_MODE:
        print \
            (f'The checked equation of regression line is y={metric.coeff0} + {metric.coeff1}/x with correlation {100 * r}%')

    # return the equation, mse, both coefficients for slope & intercept, and the correlation
    if metric.correlation_test <= 0:
        metric.correlation_test = r
        metric.mse_test = mse
    else:
        metric.correlation_val = r
        metric.mse_val = mse

    return metric


def eval_exponential_regression_equation(xxVals, yyVals, metric: Metric):
    """
    Fits a linear regression line to the specified data in the form y = ab^x
    :param xx_vals: [float] - a list of x values (explanatory variable)
    :param yy_vals: [float] - a list of y values (response variable)
    :param metric: Metric - an object containing statistical metrics and performance info
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print('fitting exponential regression equation')
        print(f' x values: {xxVals}')
        print(f' y values: {yyVals}')

    n = len(xxVals)
    sumX = 0.0
    sumX2 = 0.0
    sumXlnY = 0.0
    sumLnxLny = 0.0
    sumLnx = 0.0
    sumLny = 0.0
    sumLnx2 = 0.0
    sumLny2 = 0.0

    for i in range(0, n - 1):
        sx0 = xxVals[i]
        sumX += sx0
    for i in range(0, n - 1):
        sx20 = xxVals[i] * xxVals[i]
        sumX2 += sx20
    for i in range(0, n - 1):
        t = xxVals[i]
        v = np.log(yyVals[i])
        sumXlnY += (t * v)
    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        lny = np.log(yyVals[i])
        sumLnxLny += (lnx * lny)
    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        sumLnx += lnx
    for i in range(0, n - 1):
        lny = np.log(yyVals[i])
        sumLny += lny
    for i in range(0, n - 1):
        lny = np.log(yyVals[i])
        sumLny2 += (lny * lny)
    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        sumLnx2 += (lnx * lnx)

    lnxBar = sumLnx / n
    lnyBar = sumLny / n
    xBar = sumX / n
    sxx = sumX2 - (n * xBar * xBar)
    syy = sumLny2 - (n * (lnyBar ** 2))
    sxy = sumXlnY - (n * xBar * lnyBar)
    r = sxy / (pow(sxx, 0.5) * pow(syy, 0.5))
    xx = np.array(xxVals)
    yy = np.array(yyVals)

    if DEBUG_MODE:
        print('sxx: ' + str(sxx))
        print('syy: ' + str(syy))
        print('sxy: ' + str(sxy))
        print('b: ' + str(metric.coeff1))
        print('a: ' + str(metric.coeff0))
        print('r: ' + str(r))

    def exponent_law(x, a, b):
        return np.power(a * b, x)

    yHats = []
    for xPrime in xx:
        yHats.append(exponent_law(xPrime, metric.coeff0, metric.coeff1))

    # compute mean squared error
    mse: float = 0
    for index in range(0, n):
        error = yyVals[index] - yHats[index]
        error **= 2
        mse += error
    mse /= n

    if DEBUG_MODE:
        print \
            (f'The checked equation of regression line is y={metric.coeff0}*{metric.coeff1}^x with correlation {100 * r}%')

    # return the equation, mse, both coefficients for slope & intercept, and the correlation
    if metric.correlation_test <= 0:
        metric.correlation_test = r
        metric.mse_test = mse
    else:
        metric.correlation_val = r
        metric.mse_val = mse
    return metric


def eval_euler_exponential_regression_equation(xxVals, yyVals, metric: Metric):
    """
    Fits a euler exponential regression line to the specified data in the form y = ae^bx
    :param x_vals: [float] - a list of x values (explanatory variable)
    :param y_vals: [float] - a list of y values (response variable)
    :param metric: Metric - an object containing statistical metrics and performance info
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print('fitting euler exponential regression equation')
        print(f' x values: {xxVals}')
        print(f' y values: {yyVals}')

    n = len(xxVals)
    sumX = 0.0
    sumX2 = 0.0
    sumXlnY = 0.0
    sumLnxLny = 0.0
    sumLnx = 0.0
    sumLny = 0.0
    sumLnx2 = 0.0
    sumLny2 = 0.0

    for i in range(0, n - 1):
        sx0 = xxVals[i]
        sumX += sx0
    for i in range(0, n - 1):
        sx20 = pow(xxVals[i], 2)
        sumX2 += sx20
    for i in range(0, n - 1):
        t = xxVals[i]
        v = np.log(yyVals[i])
        sumXlnY += (t * v)
    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        lny = np.log(yyVals[i])
        sumLnxLny += (lnx * lny)
    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        sumLnx += lnx
    for i in range(0, n - 1):
        lny = np.log(yyVals[i])
        sumLny += lny
    for i in range(0, n - 1):
        lny = np.log(yyVals[i])
        sumLny2 += (lny * lny)
    for i in range(0, n - 1):
        lnx = np.log(yyVals[i])
        sumLnx2 += (lnx * lnx)

    lnxBar = sumLnx / n
    lnyBar = sumLny / n
    xBar = sumX / n
    sxx = sumX2 - (n * xBar * xBar)
    syy = sumLny2 - (n * (lnyBar ** 2))
    sxy = sumXlnY - (n * xBar * lnyBar)
    r = sxy / (pow(sxx, 0.5) * pow(syy, 0.5))
    xx = np.array(xxVals)
    yy = np.array(yyVals)

    if DEBUG_MODE:
        print('sxx: ' + str(sxx))
        print('syy: ' + str(syy))
        print('sxy: ' + str(sxy))
        print('b: ' + str(metric.coeff1))
        print('a: ' + str(metric.coeff1))
        print('r: ' + str(r))

    def euler_exponent_law(xx, a, b):
        return a * np.power(np.e, b * xx)

    yHats = []
    for xPrime in xx:
        yHats.append(euler_exponent_law(xPrime, metric.coeff0, metric.coeff1))

    # compute mean squared error
    mse: float = 0
    for index in range(0, n):
        error = yyVals[index] - yHats[index]
        error **= 2
        mse += error
    mse /= n

    if DEBUG_MODE:
        print \
            (f'The checked equation of regression line is y={metric.coeff0}*e^({metric.coeff1}*x) with correlation {100 * r}%')

    # return the equation, mse, both coefficients for slope & intercept, and the correlation
    if metric.correlation_test <= 0:
        metric.correlation_test = r
        metric.mse_test = mse
    else:
        metric.correlation_val = r
        metric.mse_val = mse

    return metric


def eval_power_regression_equation(xxVals, yyVals, metric):
    """
    Fits a linear regression line to the specified data in the form y = ax^b
    :param x_vals: [float] - a list of x values (explanatory variable)
    :param y_vals: [float] - a list of y values (response variable)
    :param metric: Metric - an object containing statistical metrics and performance info
    :return metric: Metric - an object containing statistical metrics and performance info
    """
    if DEBUG_MODE:
        print('fitting power regression equation')
        print(f' x values: {xxVals}')
        print(f' y values: {yyVals}')

    n = len(xxVals)
    sumLnxLny = 0.0
    sumLnx = 0.0
    sumLny = 0.0
    sumLnx2 = 0.0
    sumLny2 = 0.0

    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        lny = np.log(yyVals[i])
        sumLnxLny += (lnx * lny)
    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        sumLnx += lnx
    for i in range(0, n - 1):
        lny = np.log(yyVals[i])
        sumLny += lny
    for i in range(0, n - 1):
        lny = np.log(yyVals[i])
        sumLny2 += (lny * lny)
    for i in range(0, n - 1):
        lnx = np.log(xxVals[i])
        sumLnx2 += (lnx * lnx)

    lnxBar = sumLnx / n
    lnyBar = sumLny / n
    sxx = sumLnx2 - (n * (lnxBar ** 2))
    syy = sumLny2 - (n * (lnyBar ** 2))
    sxy = sumLnxLny - (n * lnxBar * lnyBar)
    r = sxy / (np.sqrt(sxx) * np.sqrt(syy))
    xx = np.array(xxVals)
    yy = np.array(yyVals)

    if DEBUG_MODE:
        print('sxx: ' + str(sxx))
        print('syy: ' + str(syy))
        print('sxy: ' + str(sxy))
        print('b: ' + str(metric.coeff1))
        print('a: ' + str(metric.coeff0))
        print('r: ' + str(r))

    def power_law(xx, a, b):
        return a * np.power(xx, b)

    yHats = []
    for xPrime in xx:
        yHats.append(power_law(xPrime, metric.coeff0, metric.coeff1))

    # compute mean squared error
    mse: float = 0
    for index in range(0, n):
        error = yyVals[index] - yHats[index]
        error **= 2
        mse += error
    mse /= n

    if DEBUG_MODE:
        print \
            (f'The checked equation of regression line is y={metric.coeff0}x^{metric.coeff1} with correlation {100 * r}%')

    # return the equation, mse, both coefficients for slope & intercept, and the correlation
    if metric.correlation_test <= 0:
        metric.correlation_test = r
        metric.mse_test = mse
    else:
        metric.correlation_val = r
        metric.mse_val = mse
    return metric
