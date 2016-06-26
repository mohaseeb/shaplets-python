from __future__ import division
from __future__ import print_function

import numpy as np

from shapelets.network import LinearLayer
from shapelets.util import utils


def test_fc_layer_initialization():
    n_inputs = 15
    n_outputs = 4
    learning_rate = 0.01
    regularization_parameter = 0.1
    training_size = 10
    fc_layer = LinearLayer(n_inputs, n_outputs, learning_rate, regularization_parameter, training_size)
    W = fc_layer.get_params()
    assert (W.shape == (1, n_outputs * (n_inputs + 1)))


def test_forward():
    n_inputs = 15
    n_outputs = 4
    learning_rate = 0.01
    regularization_parameter = 0.1
    training_size = 10
    fc_layer = LinearLayer(n_inputs, n_outputs, learning_rate, regularization_parameter, training_size)
    # initialize weights
    W = np.ones((n_outputs, n_inputs))
    W_0 = np.ones((n_outputs, 1))
    fc_layer.set_weights(W, W_0)
    # create a layer input
    layer_input = np.ones((1, n_inputs))
    # execute the layer
    layer_output = fc_layer.forward(layer_input)
    # compare the layer output to the expected one
    expected_output = np.array([[16., 16., 16., 16.]])
    assert (np.array_equal(layer_output, expected_output))


def test_backword():
    """
    :return:
    """
    # create a layer
    n_inputs = 3
    n_outputs = 2
    learning_rate = 0.01
    regularization_parameter = 0.1
    training_size = 10
    fc_layer = LinearLayer(n_inputs, n_outputs, learning_rate, regularization_parameter, training_size)
    # create a layer input
    layer_input = np.random.normal(loc=0, scale=1, size=(1, n_inputs))
    # create dL_layer_doutput
    dL_layer_output = np.random.normal(loc=0, scale=1, size=(1, n_outputs))
    # do a forward and a backward pass
    fc_layer.forward(layer_input)
    dL_input = fc_layer.backward(dL_layer_output)
    dL_dparams = fc_layer.get_dL_dparams()
    # verify dL_dinput ######
    doutput_input_truth = utils.approximate_derivative_wrt_inputs(fc_layer.forward, layer_input, n_outputs,
                                                                  h=0.01)  # n_outputs X n_inputs
    dL_input_truth = np.dot(dL_layer_output, doutput_input_truth)
    result = np.isclose(dL_input, dL_input_truth)
    assert result.all()
    # verify dL_dW
    dout_dparams_truth = utils.approximate_derivative_wrt_params(fc_layer, layer_input, n_outputs, h=0.0001)
    dL_dparams_truth = np.dot(dL_layer_output, dout_dparams_truth)
    result = np.isclose(dL_dparams, dL_dparams_truth, rtol=1e-05, atol=1e-03)
    assert result.all()
