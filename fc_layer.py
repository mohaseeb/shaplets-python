from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
import numpy as np


class FcLayer:
    def __init__(self, input_size, output_size):
        """

        :param input_size:
        :param output_size:
        """
        self.input_size = input_size
        self.output_size = output_size
        # layer weights
        self.W = None
        # layer biases
        self.W_0 = None
        self.set_weights(np.random.normal(loc=0, scale=1, size=(output_size, input_size)),
                         np.random.normal(loc=0, scale=1, size=(output_size, 1)))
        # layer input holder
        self.current_input = None
        # layer output holder
        self.current_output = None

    def set_weights(self, W, W_0):
        self.W = W
        self.W_0 = W_0

    def forward(self, layer_input):
        """

        :param layer_input:
        :return:
        """
        self.current_input = layer_input
        self.current_output = np.dot(self.W, self.current_input.T).T + self.W_0
        return self.current_output

    def backward(self, output_matrix):
        """

        :param output_matrix:
        :return:
        """
        pass

    def update_params(self, update_matrix):
        """

        :param update_matrix:
        :return:
        """
        pass

    def get_params(self):
        """

        :return:
        """
        return self.W, self.W_0
