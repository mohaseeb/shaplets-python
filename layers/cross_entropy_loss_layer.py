from __future__ import print_function
from __future__ import division
import numpy as np


class CrossEntropyLossLayer:
    def __init__(self, lamda, train_size):
        self.lamda = lamda
        self.I = train_size
        self.output_size = 1
        # layer input holder
        self.current_input_probabilities = None
        # layer output holder
        self.current_output = None
        # derivative of Loss w.r.t. inputs
        self.dL_dinput = None
        # target probabilities
        self.current_target_probabilities = None
        # parameters to be penalized in the loss function
        self.regularized_params = None

    def set_current_target_probabilities(self, target_probabilities):
        self.current_target_probabilities = target_probabilities

    def set_regularized_params(self, regularized_params):
        self.regularized_params = regularized_params

    def forward(self, layer_input):
        self.current_input_probabilities = layer_input
        self.current_output = np.sum(
            -1 * self.current_target_probabilities * np.log(self.current_input_probabilities) + (
                self.current_target_probabilities - 1) * np.log(
                1 - self.current_input_probabilities))
        # regularization part
        self.current_output += self.lamda * np.sum(self.regularized_params ** 2) / (
            self.I * self.current_input_probabilities.size)
        return self.current_output

    def backward(self, dL_dout):
        self.dL_dinput = -self.current_target_probabilities / self.current_input_probabilities + \
                         (1 - self.current_target_probabilities) / (1 - self.current_input_probabilities)
        self.dL_dinput *= dL_dout
        return self.dL_dinput

    def update_params(self):
        pass
