import numpy as np
import math


def approximate_derivative_wrt_inputs(function, inputs, n_outputs, h):
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


def approximate_derivative_wrt_params(layer, inputs, n_outputs, h):
    """
    https://en.wikipedia.org/wiki/Finite_difference#Relation_with_derivatives
    :param n_outputs:
    :param layer:
    :param inputs:
    :param h:
    :return:
    """
    params = layer.get_params()
    n_params = params.size
    dlayer_dparams = np.zeros((n_outputs, n_params))
    for param_id in range(n_params):
        f1 = layer.forward(inputs)  # 1 X n_outputs
        params[0, param_id] += h
        layer.set_params(params)
        f2 = layer.forward(inputs)  # 1 X n_outputs
        params[0, param_id] -= h
        layer.set_params(params)
        dlayer_dparams[:, param_id] = (f2 - f1) / h
    return dlayer_dparams


def sigmoid(X):
    return np.array([[(1 / (1 + math.exp(-x))) for x in X[0, :]]])
