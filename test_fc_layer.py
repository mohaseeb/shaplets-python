from fc_layer import FcLayer
import numpy as np


def test_forward():
    n_inputs = 15
    n_outputs = 4
    # create a layer
    fc_layer = FcLayer(n_inputs, n_outputs)
    # initialize weights
    W = np.ones((n_outputs, n_inputs))
    W_0 = np.ones((1, n_outputs))
    fc_layer.set_weights(W, W_0)
    # create a layer input
    layer_input = np.ones((1, n_inputs))
    # execute the layer
    layer_output = fc_layer.forward(layer_input)
    # compare the layer output to the expected one
    expected_output = np.array([[16., 16., 16., 16.]])
    assert (np.array_equal(layer_output, expected_output))


def test_backword():
    assert (1==2)


def test_fc_layer_initializaiton():
    n_inputs = 15
    n_outputs = 4
    fc_layer = FcLayer(n_inputs, n_outputs)
    W, W_0 = fc_layer.get_params()
    assert (W.shape == (n_outputs, n_inputs))
    assert (W_0.shape == (n_outputs, 1))
