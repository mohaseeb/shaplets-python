import numpy as np

from shapelets.network import CrossEntropyLossLayer
from shapelets.util import utils


def test_forward():
    training_data_size = 10
    lamda = 0.01
    params = np.array([[1, 2, 3, 4]])
    # create a cross entropy layer
    layer = CrossEntropyLossLayer(lamda, training_data_size)
    # create a layer input,
    input_probabilities = np.array([0.1, 0.4, 0.5])
    target_probabilities = np.array([0, 0, 1])
    # execute the layer
    layer.set_current_target_probabilities(target_probabilities)
    layer.set_regularized_params(params)
    layer_output = layer.forward(input_probabilities)
    # compare the layer output to the expected one
    output_truth = 1.30933331998  # calculated by hand
    assert (layer_output, output_truth)


def test_backward():
    training_data_size = 10
    lamda = 0.01
    params = np.array([[1, 2, 3, 4]])
    # create a layer
    layer = CrossEntropyLossLayer(lamda, training_data_size)
    # create a layer input,
    input_probabilities = np.array([[0.7, 0.4, 0.5, 0.1]])
    target_probabilities = np.array([[0, 1, 0, 0]])
    # create dL_layer_doutput
    dL_doutput = 1
    # perform a forward and a backward pass
    layer.set_current_target_probabilities(target_probabilities)
    layer.set_regularized_params(params)
    layer.forward(input_probabilities)
    dL_dinput = layer.backward(dL_doutput)
    # verify dL_dinput ######
    doutput_dinput_truth = utils.approximate_derivative_wrt_inputs(layer.forward, input_probabilities, 1,
                                                                   h=0.000001)  # n_outputs X n_inputs
    dL_dinput_truth = np.dot(dL_doutput, doutput_dinput_truth)
    result = np.isclose(dL_dinput, dL_dinput_truth)
    assert result.all()
