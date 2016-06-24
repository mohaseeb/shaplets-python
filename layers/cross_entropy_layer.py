import numpy as np


class CrossEntropyLayer:
    def __init__(self, input_size):
        self.input_size = input_size
        self.output_size = 1
        # layer input holder
        self.current_input = None
        # layer output holder
        self.current_output = None
        # derivative of Loss w.r.t. inputs
        self.dL_dinput = None

    def forward(self, layer_input, targets):
        self.current_input = layer_input
        self.current_output = np.sum(
            -1 * targets * np.log(self.current_input) + (targets - 1) * np.log(1 - self.current_input))
        return self.current_output

    def backward(self, dL_dout):
        pass
