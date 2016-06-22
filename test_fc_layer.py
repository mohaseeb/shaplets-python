from __future__ import print_function
from __future__ import division
from fc_layer import FcLayer
import numpy as np


def test_fc_layer_initialization():
    n_inputs = 15
    n_outputs = 4
    fc_layer = FcLayer(n_inputs, n_outputs)
    W, W_0 = fc_layer.get_params()
    assert (W.shape == (n_outputs, n_inputs))
    assert (W_0.shape == (n_outputs, 1))


def test_forward():
    n_inputs = 15
    n_outputs = 4
    # create a layer
    fc_layer = FcLayer(n_inputs, n_outputs)
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
    n_inputs = 15
    n_outputs = 4
    # create a layer
    fc_layer = FcLayer(n_inputs, n_outputs)
    # create a layer input
    layer_input = np.random.normal(loc=0, scale=1, size=(1, n_inputs))
    # create dL_layer_doutput
    dL_layer_output = np.random.normal(loc=0, scale=1, size=(1, n_outputs))
    # do a forward and a backward
    fc_layer.forward(layer_input)
    dL_input = fc_layer.backward(dL_layer_output)
    # verify dL_dinput ######
    doutput_input_truth = _approximate_derivative(fc_layer.forward, layer_input, n_outputs,
                                                  h=0.01)  # n_outputs X n_inputs
    dL_input_truth = np.dot(dL_layer_output, doutput_input_truth)
    result = np.isclose(dL_input, dL_input_truth)
    assert result.all()


def _approximate_derivative(function, inputs, n_outputs, h):
    """
    https://en.wikipedia.org/wiki/Finite_difference#Relation_with_derivatives
    :param function:
    :param inputs:
    :param n_outputs:
    :param h:
    :return:
    """
    n_inputs = inputs.size
    dFunction_dinputs = np.zeros((n_outputs, n_inputs))
    for input_id in range(n_inputs):
        f1 = function(inputs)  # 1 X n_outputs
        inputs[0, input_id] += h
        f2 = function(inputs)  # 1 X n_outputs
        inputs[0, input_id] -= h
        dFunction_dinputs[:, input_id] = (f2 - f1) / h
    return dFunction_dinputs
