from scipy.special import expit

import numpy as np

from shapelets.network import SigmoidLayer
from shapelets.util import utils


def test_forward():
    n_inputs = 5
    # create a layer
    sig_layer = SigmoidLayer(n_inputs)
    # create an input
    layer_input = np.zeros((1, n_inputs)) + 0.458
    output = sig_layer.forward(layer_input)
    # compare to expected output
    output_truth = np.zeros((1, n_inputs)) + expit(0.458)
    assert (np.isclose(output, output_truth).all())


def test_backward():
    n_inputs = 5
    # create a layer
    sig_layer = SigmoidLayer(n_inputs)
    # create a layer input
    layer_input = np.random.normal(loc=0, scale=1, size=(1, n_inputs))
    # create dL_layer_doutput
    dL_doutput = np.random.normal(loc=0, scale=1, size=(1, n_inputs))
    # perform a forward and a backward pass
    sig_layer.forward(layer_input)
    dL_dinput = sig_layer.backward(dL_doutput)
    # verify dL_dinput ######
    doutput_dinput_truth = utils.approximate_derivative_wrt_inputs(sig_layer.forward, layer_input, n_inputs,
                                                                   h=0.00001)  # n_outputs X n_inputs
    dL_dinput_truth = np.dot(dL_doutput, doutput_dinput_truth)
    result = np.isclose(dL_dinput, dL_dinput_truth)
    assert result.all()
