from layers.cross_entropy_loss_layer import CrossEntropyLossLayer


class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, sample, target):
        layer_input = sample
        for layer_id in range(len(self.layers)):
            if type(self.layers[layer_id]) == CrossEntropyLossLayer:
                layer_input = self.layers[layer_id].forward(layer_input, target)
            else:
                layer_input = self.layers[layer_id].forward(layer_input)
        return layer_input

    def backward(self):
        dL_dlayer_output = 1
        for layer_id in range(len(self.layers) - 1, -1, -1):
            dL_dlayer_output = self.layers[layer_id].backward(dL_dlayer_output)

    def update_params(self):
        for layer_id in range(len(self.layers)):
            self.layers[layer_id].update_params()
