import numpy as np
from layers.cross_entropy_layer import CrossEntropyLayer


def test_forward():
    n_inputs = 5
    # create a cross entropy layer
    layer = CrossEntropyLayer(n_inputs)
    # create a layer input,
    layer_input = np.array([0.1, 0.4, 0.5])
    target = np.array([0, 0, 1])
    # execute the layer
    layer_output = layer.forward(layer_input, target)
    # compare the layer output to the expected one
    output_truth = 1.30933331998  # calculated by hand
    assert (layer_output, output_truth)
