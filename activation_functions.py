""" activation_functions.py
---------------------

Helper classes for different cost function called by stochastic gradient
descent algorithm.

"""

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))