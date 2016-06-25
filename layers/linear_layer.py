from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
import numpy as np


class LinearLayer:
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
        self.dL_dparams = None

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
        dout_dparams = np.zeros((self.output_size, self.W.size + self.W_0.size))
        output_W_index = 0
        output_W_0_index = self.W.size
        for output_id in range(self.output_size):
            dout_dparams[output_id, output_W_index:output_W_index + self.input_size] = self.current_input
            dout_dparams[output_id, output_W_0_index] = 1
            output_W_index += self.input_size
            output_W_0_index += 1
        self.dL_dparams = np.dot(dL_dout, dout_dparams)
        # dL_dinputs calculations
        self.dL_dinput = np.dot(dL_dout, self.W)
        return self.dL_dinput, self.dL_dparams

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
        return np.concatenate((self.W.reshape((1, self.W.size)),
                               self.W_0.reshape((1, self.W_0.size))), axis=1)

    def set_params(self, params):
        self.W = np.reshape(params[:, 0:params.size - self.W_0.size], (self.output_size, self.input_size))
        self.W_0 = np.reshape(params[:, params.size - self.W_0.size:], (self.output_size, 1))
