
# built-in libraries
import random
import logging, sys

# third-party libraries
import numpy as np

# user-defined libraries
import cost_functions as cf

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        # set up logging debugger
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    def sgd(self, training_data, epochs, mini_batch_size, learning_rate,
            cost_function=cf.MeanSquaredError,
            test_data=None):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = \
                [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self._gradientDescent(mini_batch, learning_rate, cost_function)
            if test_data:
                print(
                    "Epoch {0}: {1} / {2}".format(
                        j, self._evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def _gradientDescent(self, mini_batch, learning_rate, cost_function):
        """
        update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "learning_rate"
        is the learning rate.
        """
        images = np.array([np.reshape(image, (784, )) for (image, label) in mini_batch]).transpose()
        labels = np.array([np.reshape(label, (10, )) for (image, label) in mini_batch]).transpose()

        delta_nabla_b, delta_nabla_w = self._backprop(images, labels, cost_function)

        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, delta_nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, delta_nabla_b)]

    def _forwardprop(self, inputs):
        """
        :param inputs:
        a matrix X = [x1, ... , xm], where xi is a single training input, 784*1

        :return:
        (A, Z), where A is a list of matrices of activations of each training input;
        Z is the list of matrices of intermediate z values of each training input
        """
        activation = inputs
        activations = [activation]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self._sigmoid(z)
            activations.append(activation)

        return (activations, zs)

    def _backprop(self, image, label, cost_function):
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.

        :param image:
        a matrix X = [x1, ... , xm], where xi is a single training input;
        e.g. in our case, X is a 784*10 matrix, Xij is the activation of
        ith input neuron in the jth training example

        :param label:
        a matrix Y = [y1, ... , ym], where yi is the label/desired output;
        e.g. Y in this case is a 10*10 matrix, each yi is a 10*1 vector

        :returns:
        """
        nabla_b = [0 for i in range(len(self.biases))]
        nabla_w = [0 for i in range(len(self.weights))]

        activations, zs = self._forwardprop(image)

        # backward pass
        # TODO: add or change _cost_derivative function to all the cost function classes
        delta = cost_function.cost_derivative(activations[-1], label) * self._sigmoid_prime(zs[-1])
        error_b_sum = np.sum(delta, axis=1)
        nabla_b[-1] = np.reshape(error_b_sum, (error_b_sum.shape[0], 1))
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * self._sigmoid_prime(zs[-l])
            error_b_sum = np.sum(delta, axis=1)
            nabla_b[-l] = np.reshape(error_b_sum, (error_b_sum.shape[0], 1))
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def _feedforward(self, a):
        """
        calculate the output of the network
        :param a: a numpy ndarray (n, 1)
        :return: a single number corresponding to the output of the network
        """
        for b, w in zip(self.biases, self.weights):
            a = self._sigmoid(np.dot(w, a) + b)
        return a

    def _evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self._feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self._sigmoid(z) * (1 - self._sigmoid(z))

