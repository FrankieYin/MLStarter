
# built-in libraries
import random, math
import logging, sys

# third-party libraries
import numpy as np

# user-defined libraries
import cost_functions as cf
import activation_functions as af

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)/math.sqrt(x) # standard deviation is squeezed down
                        for x, y in zip(sizes[:-1], sizes[1:])]

        self.dropout_enabled = False
        self.l2_enabled = True
        self.dropout_size = 2 # 三贤者系统 vs 二分心智！

        # The tolerance might be to aggressive, as some networks tend to plateau for some number of epochs
        # only to then start improving again
        self.no_improvement_tolerance = 10
        # we take the minor "improvements" as errors/fluctuations
        self.significant_improvement_rate = 0.005
        # variable for learning rate schedule
        self.slowdown_factor = 2

        # set up logging debugger
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    # TODO: change to momentum-based stochastic gradient descent
    # TODO: grid search for hyper-parameters
    def sgd(self, training_data, epochs, mini_batch_size, learning_rate,
            lmbda = 0.0, # the regularisation constant
            cost_function=cf.CrossEntropy,
            test_data=None,
            validation_data=None,
            dropout_enabled=False,
            l2_enabled=True,
            early_stopping=False,
            learning_schedule=True):
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
        self.dropout_enabled = dropout_enabled
        self.l2_enabled = l2_enabled

        if test_data: n_test = len(test_data)
        n = len(training_data)

        best_accuracy = 0
        best_epoch = None
        no_improvement = 0
        n_slowdown = 0
        epoch = 1
        while True:
            random.shuffle(training_data)
            mini_batches = \
                [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self._gradientDescent(mini_batch, learning_rate, cost_function, n, lmbda)
            if test_data:
                n_success = self._evaluate(test_data)
                print(
                    "Epoch {0}: {1} / {2}".format(
                        epoch, n_success, n_test))

                if n_success > int(best_accuracy*(1+self.significant_improvement_rate)):
                    best_accuracy = n_success
                    best_epoch = epoch
                    no_improvement = 0
                else:
                    no_improvement += 1

                # Early-stopping using the no-improvement-in-n-epoch technique:
                # Early-stopping helps to set the number of epochs (one of the less significant hyper-parameters).
                # In particular, it means we don't need to explicitly think about how number of epochs depends on other
                # hyper-parameters, it's taken care of automatically.
                #
                # At early stages, it's better to turn off early-stopping, as it helps to inform about overfitting and
                # regularisation.
                #
                # Another way of early-stopping is to halve the learning_rate every time the conditions satisfy
                # the no-improvement-in-n rules. When the learning_rate is 1/128 of the original value, we terminate.
                if early_stopping:
                    if no_improvement >= self.no_improvement_tolerance:
                        if learning_schedule:
                            if n_slowdown == 7:
                                break
                            else:
                                learning_rate /= self.slowdown_factor
                                n_slowdown += 1
                        else:
                            print("Stopping the network: no improvement in {0} epochs."
                                  .format(self.no_improvement_tolerance))
                            break
                elif epoch == epochs:
                    break
            else:
                print("Epoch {0} complete".format(epoch))

            epoch += 1

        if test_data:
            print("Best accuracy: {0}% on epoch {1}".format(best_accuracy*100/n_test, best_epoch))
        print("Total epochs trained: {0}".format(epoch))

    def _gradientDescent(self, mini_batch, learning_rate, cost_function, num_training, lmbda):
        """
        update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "learning_rate"
        is the learning rate.
        """
        images = np.array([np.reshape(image, (784, )) for (image, label) in mini_batch]).transpose()
        labels = np.array([np.reshape(label, (10, )) for (image, label) in mini_batch]).transpose()

        # if we are using dropout for regularisation
        if self.dropout_enabled:

            # copy the current weights and biases for restoration later on
            weights = [w for w in self.weights]
            biases = [b for b in self.biases]
            neurons_deleted = []

            # we randomly choose some neurons in the hidden layer to delete
            for i in range(1, self.num_layers-1):
                n = self.sizes[i]
                neurons_to_delete = random.sample(range(n), math.floor(n*(self.dropout_size-1)/self.dropout_size))
                neurons_deleted.append(neurons_to_delete)
                # delete rows in the weight matrix connecting to the previous layer
                self.weights[i-1] = np.delete(self.weights[i-1], neurons_to_delete, axis=0)
                self.biases[i-1] = np.delete(self.biases[i-1], neurons_to_delete, axis=0)
                # delete columns in the weight matrix connecting to the next layer
                self.weights[i] = np.delete(self.weights[i], neurons_to_delete, axis=1)

        delta_nabla_b, delta_nabla_w = self._backprop(images, labels, cost_function)

        if self.l2_enabled:
            # l2 regularisation, weight decay on weights
            self.weights = [(1 - learning_rate * (lmbda/num_training)) * w - (learning_rate / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, delta_nabla_w)]
        else:
            self.weights = [w - (learning_rate / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, delta_nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, delta_nabla_b)]

        if self.dropout_enabled:
            # we restore the deleted neurons
            for i in range(1, self.num_layers-1):
                neurons_to_restore = neurons_deleted[i-1]
                neurons_to_restore.sort() # sort to avoid index changing during insertion
                for j in neurons_to_restore:
                    # insert before current jth row
                    self.weights[i-1] = np.insert(self.weights[i-1], j, weights[i-1][j], axis=0)
                    self.biases[i-1] = np.insert(self.biases[i-1], j, biases[i-1][j], axis=0)
                    # insert before current jth column
                    self.weights[i] = np.insert(self.weights[i], j, weights[i][:, j], axis=1)

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
            activation = af.sigmoid(z)
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
        delta = cost_function.error(activations[-1], label, zs[-1])
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
            delta = np.dot(self.weights[-l+1].transpose(), delta) * af.sigmoid_prime(zs[-l])
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
            # To compensate for the increase in number of neurons activated
            # we divide the weights and biases by the dropout size.
            if self.dropout_enabled:
                a = af.sigmoid(np.dot(w/self.dropout_size, a) + b/self.dropout_size)
            else:
                a = af.sigmoid(np.dot(w, a) + b)
        return a

    def _evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self._feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

