"""
Linear Regression with one variable

@author GalenS <galen.scovell@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
sns.set_style('white')


ITERATIONS = 1500  # Number of iterations to use for gradient descent
ALPHA = 0.01       # Learning rate: how big steps are, larger is more aggressive


def scatterplot(x, y):
    """
    Make scatterlot from initial data.
    :param x: x values
    :type  x: 2d ndarray [[1., x-val], [1., x-val], ...]
    :param y: y values
    :type  y: 2d ndarray [[y-val], [y-val], ...]
    """
    plt.figure(figsize=(14, 8), dpi=80)
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
    :type      x: 2d ndarray [[1., x-val], [1., x-val], ...]
    :param     y: y values
    :type      y: 2d ndarray [[y-val], [y-val], ...]
    :param theta: current theta value to use in computation
    :type  theta: 2d ndarray [[theta0 float], [theta1 float]]
    :return: float cost
    """
    m = y.size
    h = x.dot(theta)
    j = 1 / (2 * m) * np.sum(np.square(h - y))
    return j


def gradient_descent(x, y, theta=[[0], [0]]):
    """
    Minimize cost using gradient descent.
    :param     x: x values
    :type      x: 2d ndarray [[1., x-val], [1., x-val], ...]
    :param     y: y values
    :type      y: 2d ndarray [[y-val], [y-val], ...]
    :param theta: starting theta values
    :type  theta: 2d ndarray [[theta0 float], [theta1 float]]
    :return: tuple, theta 2d array and j_history array
    """
    m = y.size
    j_history = np.zeros(ITERATIONS)  # Make list of 0's length of ITERATIONS
    for i in np.arange(ITERATIONS):
        h = x.dot(theta)
        theta = theta - ALPHA * (1 / m) * (x.T.dot(h - y))
        j_history[i] = compute_cost(x, y, theta)
    return theta, j_history


def plot_costs(j_history):
    """
    Plot line of costs calculated in gradient descent (J's).
    :param j_history: costs calculated from descent
    :type  j_history: list of floats
    """
    plt.figure(figsize=(14, 8))
    plt.plot(range(len(j_history)), j_history)
    plt.grid(True)
    plt.title('J (Cost)')
    plt.xlabel('Iteration')
    plt.ylabel('Cost function')
    plt.xlim([0, 1.05 * ITERATIONS])
    plt.ylim([4, 7])
    plt.show()
    plt.close()


def plot_descent(x, y, theta):
    """
    Plot gradient descent thetas as line over dataset scatterplot.
    :param     x: x values
    :type      x: 2d ndarray [[1., x-val], [1., x-val], ...]
    :param     y: y values
    :type      y: 2d ndarray [[y-val], [y-val], ...]
    :param theta: calculated theta values
    :type  theta: 2d ndarray [[theta0 float], [theta1 float]]
    """
    # Compute prediction for each point in xx range using calculated theta values
    # h(x) = theta0 + theta1 * x
    xx = np.arange(5, 23)
    yy = theta[0] + theta[1] * xx

    plt.figure(figsize=(14, 8), dpi=80)
    plt.scatter(x[:, 1], y, s=30, c='r', marker='x', linewidths=1)
    plt.plot(xx, yy, label='Hypothesis: h(x) = {0:.2f} + {1:.2f}x'.format(float(theta[0]), float(theta[1])))

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
    :param theta: calculated theta values
    :type  theta: 2d ndarray [[theta0 float], [theta1 float]]
    :param value: Given value to predict based off of
    :type  value: int
    :return: float prediction
    """
    return theta.T.dot([1, value]) * 10000


def plot_3d(x, y):
    """
    Plot x vs y vs z (cost, j value) 3D plot.
    :param     x: x values
    :type      x: 2d ndarray [[1., x-val], [1., x-val], ...]
    :param     y: y values
    :type      y: 2d ndarray [[y-val], [y-val], ...]
    """
    # Create grid coordinates
    x_axis = np.linspace(-10, 10, 50)
    y_axis = np.linspace(-1, 4, 50)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing='xy')
    z = np.zeros((x_axis.size, y_axis.size))

    # Calculate z-values based on grid coefficients
    for (i, j), v in np.ndenumerate(z):
        z[i, j] = compute_cost(x, y, theta=[[xx[i, j]], [yy[i, j]]])

    # Construct plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, z, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
    ax.set_zlabel('Cost')
    ax.set_zlim(z.min(), z.max())
    ax.view_init(elev=15, azim=230)
    plt.title('X vs. Y vs. Cost')
    ax.set_xlabel(r'$\theta_0$', fontsize=17)
    ax.set_ylabel(r'$\theta_1$', fontsize=17)
    plt.show()
    plt.close()



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
    scatterplot(x, y)

    # Gradient descent and visualize
    theta, j_history = gradient_descent(x, y)
    plot_costs(j_history)
    plot_descent(x, y, theta)

    # Make some predictions
    print('Predicted profit for 3.5k population: {0}'.format(make_prediction(theta, 3.5)))
    print('Predicted profit for 7k population: {0}'.format(make_prediction(theta, 7)))

    plot_3d(x, y)

