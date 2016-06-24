import numpy as np
import math


def approximate_derivative(function, inputs, n_outputs, h):
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


def sigmoid(X):
    return np.array([[(1 / (1 + math.exp(-x))) for x in X[0, :]]])
