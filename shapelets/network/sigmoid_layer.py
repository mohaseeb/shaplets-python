from shapelets.util import utils


class SigmoidLayer:
    def __init__(self, input_size):
        self.input_size = input_size
        self.output_size = input_size
        # layer input holder
        self.current_input = None
        # layer output holder
        self.current_output = None
        # derivative of Loss w.r.t. inputs
        self.dL_dinput = None

    def forward(self, layer_input):
        self.current_input = layer_input
        self.current_output = utils.sigmoid(self.current_input)
        return self.current_output

    def backward(self, dL_dout):
        dout_dinput = self.current_output * (1 - self.current_output)
        self.dL_dinput = dL_dout * dout_dinput
        return self.dL_dinput

    def update_params(self):
        pass
