""" cost_functions.py
---------------------

Helper classes for different cost function called by stochastic gradient
descent algorithm.

"""
import numpy as np

class MeanSquaredError(object):

    @staticmethod
    def cost(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def cost_derivative(output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        # logging.info("outpur_actications: {0}".format(output_activations.shape))
        # logging.info("y: {0}".format(y.shape))
        return (output_activations - y)

class CrossEntropy(object):

    @staticmethod
    def cost(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))