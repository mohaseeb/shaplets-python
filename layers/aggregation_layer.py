from __future__ import print_function
import numpy as np


class AggregationLayer:
    def __init__(self, layers):
        self.layers = layers
        self.layers_number = len(layers)
        self.number_params = self._get_total_number_params()
        # layer input holder
        self.current_input = None
        # layer output holder
        self.current_output = None
        # derivative of Loss w.r.t. inputs
        self.dL_dinput = None

    def _get_total_number_params(self):
        total = 0
        for layer_number in range(self.layers_number):
            total += self.layers[layer_number].get_size()
        return total

    def forward(self, layer_input):
        self.current_input = layer_input
        self.current_output = np.zeros((1, self.layers_number))
        for layer_number in range(self.layers_number):
            self.current_output[0, layer_number] = self.layers[layer_number].forward(self.current_input)
        return self.current_output

    def backward(self, dL_dout):
        dL_dparams = np.zeros((1, self.number_params))
        layer_segment_id = 0
        for layer_number in range(self.layers_number):
            layer_size = self.layers[layer_number].get_size()
            dL_dparams[0, layer_segment_id:layer_segment_id + layer_size] = self.layers[layer_number].backward(
                dL_dout[0, layer_number])[:]
            layer_segment_id += layer_size
        return dL_dparams

    def get_params(self):
        """

        :return:
        """
        params = np.zeros((1, self.number_params))
        layer_segment_id = 0
        for layer_number in range(self.layers_number):
            layer_size = self.layers[layer_number].get_size()
            params[0, layer_segment_id:layer_segment_id + layer_size] = self.layers[layer_number].get_params()[:]
            layer_segment_id += layer_size
        return params

    def set_params(self, params):
        """

        :param params:
        :return:
        """
        layer_segment_id = 0
        for layer_number in range(self.layers_number):
            layer_size = self.layers[layer_number].get_size()
            self.layers[layer_number].set_params(params[0, layer_segment_id:layer_segment_id + layer_size])
            layer_segment_id += layer_size

    def update_params(self):
        for layer_number in range(self.layers_number):
            self.layers[layer_number].update_params()
