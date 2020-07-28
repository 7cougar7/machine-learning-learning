import numpy as np


def sigmoid(x):
    # Sigmoid activation function: f(x) = 1/(1+e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Derivative of activation function:
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length
    return ((y_true - y_pred) ** 2).mean()


class Neuron:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
        self.sigmoid_val = 0
        self.sum_val = 0

    def feedforward(self, inputs):
        # Weight inputs, add bias, and use activation function
        total = np.dot(self.weights, inputs) + self.biases
        self.sum_val = total
        self.sigmoid_val = sigmoid(total)
        return

    def get_weight(self, index):
        return self.weights[index]

    def set_weight(self, index, value):
        self.weights[index] = value


class NeuralNetwork:
    '''
    A neural network with:
      - 2 inputs
      - a hidden layer with 2 neurons (h1, h2)
      - an output layer with 1 neuron (o1)

    *** DISCLAIMER ***:
    The code below is intended to be simple and educational, NOT optimal.
    Real neural net code looks nothing like this. DO NOT use this code.
    Instead, read/run it to understand how this specific network works.
    '''

    def __init__(self):
        # # Weights
        # self.w1 = np.random.normal()
        # self.w2 = np.random.normal()
        # self.w3 = np.random.normal()
        # self.w4 = np.random.normal()
        # self.w5 = np.random.normal()
        # self.w6 = np.random.normal()
        #
        # # biases
        # self.b1 = np.random.normal()
        # self.b2 = np.random.normal()
        # self.b3 = np.random.normal()

        self.h1 = Neuron(
            weights=np.array([np.random.normal(), np.random.normal()]),
            biases=np.random.normal()
        )
        self.h2 = Neuron(
            weights=np.array([np.random.normal(), np.random.normal()]),
            biases=np.random.normal()
        )
        self.o1 = Neuron(
            weights=np.array([np.random.normal(), np.random.normal()]),
            biases=np.random.normal()
        )


    def feedforward(self, x):
        # x is a numpy array with 2 elements
        self.h1.feedforward(x)
        self.h2.feedforward(x)
        self.o1.feedforward(np.array([self.h1.sigmoid_val, self.h2.sigmoid_val]))
        return self.o1.sigmoid_val

    def train(self, data, all_y_trues):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
          Elements in all_y_trues correspond to those in data.
        '''
        learn_rate = 0.1
        epochs = 25000  # Number of times to loop through the entire dataset
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Forward feed each neuron
                self.h1.feedforward(x)
                self.h2.feedforward(x)
                self.o1.feedforward(np.array([self.h1.sigmoid_val, self.h2.sigmoid_val]))
                y_pred = self.o1.sigmoid_val

                # Calc partial derivatives
                # Naming: dL__dw1 means "partial L / partial w1"
                d_l__dy_pred = -2 * (y_true - y_pred)

                # Neuron o1
                deriv_o1 = deriv_sigmoid(self.o1.sum_val)
                d_y_pred__d_w5 = self.h1.sigmoid_val * deriv_o1
                d_y_pred__d_w6 = self.h2.sigmoid_val * deriv_o1
                d_y_pred__d_b3 = deriv_o1

                d_y_pred__d_h1 = self.o1.get_weight(0) * deriv_o1  # w5 * ...
                d_y_pred__d_h2 = self.o1.get_weight(1) * deriv_o1  # w6 * ...

                # Neuron h1
                deriv_h1 = deriv_sigmoid(self.h1.sum_val)
                d_h1__d_w1 = x[0] * deriv_h1
                d_h1__d_w2 = x[1] * deriv_h1
                d_h1__d_b1 = deriv_h1

                # Neuron h2
                deriv_h2 = deriv_sigmoid(self.h2.sum_val)
                d_h2__d_w3 = x[0] * deriv_h2
                d_h2__d_w4 = x[1] * deriv_h2
                d_h2__d_b2 = deriv_h2

                # ~~~ Update weights and biases
                learning_multiplier = learn_rate * d_l__dy_pred

                # Neuron h1
                h1_rate = learning_multiplier * d_y_pred__d_h1
                self.h1.set_weight(0, self.h1.get_weight(0) - (h1_rate * d_h1__d_w1))
                self.h1.set_weight(1, self.h1.get_weight(1) - (h1_rate * d_h1__d_w2))
                self.h1.biases -= (h1_rate * d_h1__d_b1)

                # Neuron h2
                h2_rate = learning_multiplier * d_y_pred__d_h2
                self.h2.set_weight(0, self.h1.get_weight(0) - (h2_rate * d_h2__d_w3))
                self.h2.set_weight(1, self.h1.get_weight(1) - (h2_rate * d_h2__d_w4))
                self.h2.biases -= (h2_rate * d_h2__d_b2)

                # Neuron o1
                self.o1.set_weight(0, self.o1.get_weight(0) - (learning_multiplier * d_y_pred__d_w5))
                self.o1.set_weight(1, self.o1.get_weight(1) - (learning_multiplier * d_y_pred__d_w6))
                self.o1.biases -= (learning_multiplier * d_y_pred__d_b3)

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))


# Define dataset
data = np.array([
    [-2, -1],  # Alice
    [25, 6],  # Bob
    [17, 4],  # Charlie
    [-15, -6],  # Diana
])
all_y_trues = np.array([
    1,  # Alice
    0,  # Bob
    0,  # Charlie
    1,  # Diana
])

# Train the neural network
network = NeuralNetwork()

emily = np.array([-7, -3])  # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily))  # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank))  # 0.039 - M

network.train(data, all_y_trues)

# Make some predictions
emily = np.array([-7, -3])  # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily))  # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank))  # 0.039 - M


soomin = np.array([3, -3])  # 155 pounds, 68 inches
tilo = np.array([15, 6])  # 155 pounds, 68 inches
print("Soomin: %.3f" % network.feedforward(soomin))  # 0.951 - F
print("Tilo: %.3f" % network.feedforward(tilo))  # 0.951 - F