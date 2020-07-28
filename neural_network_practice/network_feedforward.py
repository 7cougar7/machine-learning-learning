import numpy as np
from neural_network_practice.Neuron import Neuron


class OurNeuralNetwork:
    """
        A neural network with:
            - 2 inputs
            - a hidden layer with 2 neurons (h1, h2)
            - an output layer with 1 neuron (o1)
        Each neuron has the same weights and bias:
            - w = [0, 1]
            - b = 0
    """

    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weights=weights, biases=bias)
        self.h2 = Neuron(weights=weights, biases=bias)
        self.o1 = Neuron(weights=weights, biases=bias)

    def feedforward(self, x):
        # Calculate output of hidden layer nodes
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # Calculate output of output layer nodes
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1


network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))
exit()
