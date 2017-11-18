from __future__ import print_function
import numpy as np
import math
from sklearn.cluster import KMeans


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


def get_one_active_representation(labels):
    classes = np.unique(labels)
    one_active_labels = np.zeros((labels.size, classes.size))
    for label_id in range(labels.size):
        one_active_labels[label_id, np.where(classes == labels[label_id])] = 1
    return one_active_labels


def get_centroids_of_segments(data, L, K):
    """

    :param data: the dataset
    :param L: segment length
    :param K: number of centroids
    :return: the top K centroids of the clustered segments
    """
    data_segmented = segment_dataset(data, L)
    centroids = get_centroids(data_segmented, K)
    return centroids


def segment_dataset(data, L):
    """

    :param data:
    :param L: segment length
    :return:
    """
    # number of time series, time series size
    I, Q = data.shape
    # number of segments in a time series
    J = Q - L + 1
    S = np.zeros((J * I, L))
    # create segments
    for i in range(I):
        for j in range(J):
            S[i * J + j, :] = data[i, j:j + L]
    return S


def get_centroids(data, k):
    clusterer = KMeans(n_clusters=k)
    clusterer.fit(data)
    return clusterer.cluster_centers_
