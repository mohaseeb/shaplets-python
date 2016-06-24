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
        # derivative of Loss w.r.t. inputs
        self.dL_dinput = None
        # derivative of Loss w.r.t. weights
        self.dL_dW = None
        # derivative of Loss w.r.t. biases
        self.dL_dW_0 = None

    def set_weights(self, W, W_0):
        self.W = W
        self.W_0 = W_0

    def forward(self, layer_input):
        """

        :param layer_input:
        :return:
        """
        self.current_input = layer_input
        self.current_output = np.dot(self.W, self.current_input.T) + self.W_0
        return self.current_output.T

    def backward(self, dL_dout):
        """

        :param dL_dout: (1 X output_size)
        :return: dL_dinputs (1 X input_size), dL_dW (output_size X input_size), dL_dW_0 (output_size, 1)
        """
        # dL_dW calculations
        # TODO
        # dL_dW_0 calculations
        # TODO
        # dL_dinputs calculations
        self.dL_dinput = np.dot(dL_dout, self.W)
        return self.dL_dinput

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
