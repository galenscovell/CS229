"""
Linear Regression with one variable

@author GalenS <galen.scovell@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')



def scatterplot(x, y):
    """
    Make scatterlot from initial data.
    :param x: x values
    :type  x: ndarray, 2d array [[1., x-val], [1., xval], etc.]
    :param y: y values
    :type  y: ndarray, 2d array [[y-val], [y-val], etc.]
    """
    plt.figure(figsize=(12, 8), dpi=80)
    plt.scatter(x[:, 1], y, s=30, c='r', marker='x', linewidths=1)
    plt.grid(True)
    plt.xlim(4, 24)
    plt.ylabel('Profit ($10k)')
    plt.xlabel('Population (10k)')
    plt.show()
    plt.close()


def compute_cost(x, y, theta=[[0], [0]]):
    """
    Compute cost J from current theta value.
    :param     x: x values
    :type      x: ndarray, 2d array [[1., x-val], [1., xval], etc.]
    :param     y: y values
    :type      y: ndarray, 2d array [[y-val], [y-val], etc.]
    :param theta: 
    :type  theta: ndarray, 2d array [[x-val, y-val], etc.]
    :return: float cost
    """
    m = y.size
    j = 0
    h = x.dot(theta)
    j = 1 / (2 * m) * np.sum(np.square(h - y))
    return j


def gradient_descent(x, y, theta=[[0], [0]], alpha=0.01, iter_num=1500):
    """
    Minimize cost using gradient descent.
    :param x: x values
    :type  x: ndarray, 2d array [[1., x-val], [1., xval], etc.]
    :param y: y values
    :type  y: ndarray, 2d array [[y-val], [y-val], etc.]
    :param    theta: 
    :type     theta: ndarray, 2d array [[x-val, y-val], etc.]
    :param    alpha: 
    :type     alpha: float
    :param iter_num: number of iterations for gradient descent
    :type  iter_num: int
    :return: tuple theta and j_history
    """
    m = y.size
    j_history = np.zeros(iter_num)
    for i in np.arange(iter_num):
        h = x.dot(theta)
        theta = theta - alpha * (1 / m) * (x.T.dot(h - y))
        j_history[i] = compute_cost(x, y, theta)
    return theta, j_history


def plot_costs(j_history, iter_num=1500):
    """
    Plot line of costs calculated in gradient descent (J's).
    :param j_history: costs calculated from descent
    :type  j_history: list of floats
    :param  iter_num: number of iterations for gradient descent
    :type   iter_num: int
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(j_history)), j_history)
    plt.grid(True)
    plt.title('J (Cost)')
    plt.xlabel('Iteration')
    plt.ylabel('Cost function')
    plt.xlim([0, 1.05 * iter_num])
    plt.ylim([4, 7])
    plt.show()
    plt.close()


def plot_descent(x, y, theta):
    """
    Plot gradient descent thetas as line over dataset scatterplot.
    :param     x: x values
    :type      x: ndarray, 2d array [[1., x-val], [1., xval], etc.]
    :param     y: y values
    :type      y: ndarray, 2d array [[y-val], [y-val], etc.]
    :param theta: 
    :type  theta: ndarray, 2d array [[x-val, y-val], etc.]
    """
    xx = np.arange(5, 23)          # 
    yy = theta[0] + theta[1] * xx  # 

    plt.figure(figsize=(12, 8), dpi=80)
    plt.scatter(x[:, 1], y, s=30, c='r', marker='x', linewidths=1)
    plt.plot(xx, yy, label='Linear Regression (Gradient Descent)')

    plt.grid(True)
    plt.xlim(4, 24)  # Extend plot slightly beyond data bounds
    plt.xlabel('Population of City (10k)')
    plt.ylabel('Profit ($10k)')
    plt.legend(loc=4)
    plt.show()
    plt.close()


def make_prediction(theta, value):
    """
    Make a prediction based on gradient descent theta results.
    :param theta: Calculated theta from gradient descent
    :param theta: 
    :type  theta: ndarray, 2d array [[x-val, y-val], etc.]
    :param value: Given value to predict based off of
    :type  value: int
    :return: float prediction
    """
    return theta.T.dot([1, value]) * 10000



if __name__ == '__main__':
    # Read in data and visualize
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    x = np.c_[np.ones(data.shape[0]), data[:,0]]  # data.shape[0] = 97 (rows in 1st column)
                                                  # np.ones(data.shape[0]) = list of 97 1's
                                                  # data[:, 0] = all data in 1st column
                                                  # np.c_[] = combine results above:
                                                  #     list of lists, each inner list
                                                  #         is [1., column val]
    y = np.c_[data[:, 1]]  # list of lists, each inner list is single entry [2nd column val]
    # scatterplot(x, y)

    # Gradient descent and visualize
    theta, j_history = gradient_descent(x, y)
    plot_costs(j_history)
    plot_descent(x, y, theta)

    # Make some predictions
    print('Predicted profit for 3.5k population: {0}'.format(make_prediction(theta, 3.5)))
    print('Predicted profit for 7k population: {0}'.format(make_prediction(theta, 7)))

