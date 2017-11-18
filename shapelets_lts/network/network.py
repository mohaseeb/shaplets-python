from . import CrossEntropyLossLayer
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.regularized = []

    def add_layer(self, layer, regularized=False):
        self.layers.append(layer)
        self.regularized.append(regularized)

    def remove_loss_layer(self):
        if isinstance(self.layers[-1], CrossEntropyLossLayer):
            del self.layers[-1]

    def forward(self, sample, target):
        layer_input = sample
        for layer_id in range(len(self.layers)):
            if isinstance(self.layers[layer_id], CrossEntropyLossLayer):
                self.layers[layer_id].set_current_target_probabilities(target)
                self.layers[layer_id].set_regularized_params(self._get_regularized_params())
            layer_input = self.layers[layer_id].forward(layer_input)
        return layer_input

    def backward(self):
        dL_dlayer_output = 1
        for layer_id in range(len(self.layers) - 1, -1, -1):
            dL_dlayer_output = self.layers[layer_id].backward(dL_dlayer_output)

    def update_params(self):
        for layer_id in range(len(self.layers)):
            self.layers[layer_id].update_params()

    def _get_regularized_params(self):
        regularized = []
        for layer_id in range(len(self.layers)):
            if self.regularized[layer_id]:
                regularized.append(self.layers[layer_id].get_params())
        return np.concatenate(regularized, axis=1)
