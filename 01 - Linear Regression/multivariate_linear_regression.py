"""
Linear Regression with multiple variables.

@author GalenS <galen.scovell@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
sns.set_style('white')


ITERATIONS = 200
ALPHA = 0.1


def compute_cost(x, y, theta=[[0], [0], [0]]):
    """
    Normal cost function with more features.
    :return: float cost, j
    """
    m = y.size
    h = x.dot(theta)
    j = 1 / (2 * m) * np.sum(np.square(h - y))
    return j


def plot_costs(j_history):
    """
    Plot cost history over gradient descent.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(range(len(j_history)), j_history)
    plt.grid(True)
    plt.title('J (Cost)')
    plt.xlabel('Iteration')
    plt.ylabel('Cost function')
    plt.xlim([-0.05 * ITERATIONS, 1.05 * ITERATIONS])
    plt.ylim([-0.05 * min(j_history), 1.05 * max(j_history)])
    plt.show()
    plt.close()


def gradient_descent(x, y, theta=[[0], [0], [0]]):
    """
    Normal gradient descent with more features.
    :return: tuple (final theta ndarray, cost history list)
    """
    m = y.size
    j_history = []
    for i in range(ITERATIONS):
        h = x.dot(theta)
        theta = theta - (ALPHA / m) * (x.T.dot(h - y))
        j_history.append(compute_cost(x, y, theta))
    return theta, j_history


def predict(theta, x1, x2):
    """
    Make prediction with scaled query features.
    :return: float prediction
    """
    return theta.T.dot([1.0, x1, x2])


def scale_training_features(x):
    """
    Scale features used for training.
    Calculation is (feature_val - avg_feature_val) / feature_stdev
    :return: feature ndarray with scaled values
    """
    scaled_x = []
    avg_std = []
    for f in range(1, 3):
        avg = sum(x[:, f]) / x[:, f].size
        rng = max(x[:, f]) - min(x[:, f])
        std = statistics.stdev(x[:, f])
        avg_std.append([avg, std])
        scaled_feature = []
        for i in x[:, f]:
            scaled = (i - avg) / std
            scaled_feature.append(scaled)
        scaled_x.append(scaled_feature)
    return avg_std, np.c_[np.ones(x.shape[0]), scaled_x[0], scaled_x[1]]


def scale_query_features(avg_std, x1, x2):
    """
    Scale features used to make a prediction.
    Calculation is (feature_val - avg_feature_val) / feature_stdev
    :return: tuple (scaled x1 float, scaled x2 float)
    """
    x1_avg = avg_std[0][0]
    x1_std = avg_std[0][1]
    x2_avg = avg_std[1][0]
    x2_std = avg_std[1][1]
    scaled_x1 = (x1 - x1_avg) / x1_std
    scaled_x2 = (x2 - x2_avg) / x2_std
    return (scaled_x1, scaled_x2)


if __name__ == '__main__':
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    x = np.c_[np.ones(data.shape[0]), data[:, 0], data[:, 1]]
    y = np.c_[data[:, 2]]

    avg_std, x = scale_training_features(x)
    theta, j_history = gradient_descent(x, y)
    plot_costs(j_history)

    print('Hypothesis: h(x) = {0:.2f}x0 + {1:.2f}x1 + {2:.2f}x2'.format(
        float(theta[0]), float(theta[1]), float(theta[2])))

    query = scale_query_features(avg_std, 1650, 3)
    print('Prediction of house price with 1650sqft and 3 bedrooms: $%0.2f' % float(
        predict(theta, query[0], query[1])))

    query = scale_query_features(avg_std, 670, 1)
    print('Prediction of house price with 670sqft and 1 bedrooms: $%0.2f' % float(
        predict(theta, query[0], query[1])))

    query = scale_query_features(avg_std, 1000, 2)
    print('Prediction of house price with 1000sqft and 2 bedrooms: $%0.2f' % float(
        predict(theta, query[0], query[1])))
