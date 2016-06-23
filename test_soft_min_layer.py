import numpy as np
from soft_min_layer import SoftMinLayer


def test_forward():
    soft_layer = SoftMinLayer(np.ones((1, 5)))
    T = np.ones((1, 10)) + 1
    assert (soft_layer.forward(T) == 1)


def test_backward():
    # create the soft min layer
    L = 5
    shapelet = np.random.normal(loc=0, scale=1, size=(1, L))
    soft_layer = SoftMinLayer(shapelet)
    # create a layer input
    Q = 10
    T = np.random.normal(loc=0, scale=1, size=(1, Q))
    # create dL_dM
    dL_dM = np.random.normal()
    # do a forward and a backward pass
    soft_layer.forward(T)
    dL_dS = soft_layer.backward(dL_dM)
    # verify dL_dS ######
    dM_dS_truth = _approximate_dlayer_dparams(soft_layer, T, h=0.01)
    dL_dS_truth = dL_dM * dM_dS_truth
    print(dL_dS)
    print(dL_dS_truth)
    result = np.isclose(dL_dS, dL_dS_truth)
    assert result.all()


def test_shapelet_dist_sqr_error():
    soft_layer = SoftMinLayer(np.ones((1, 10)))
    assert (soft_layer.dist_sqr_error(np.zeros((1, 10))) == 1)


def _approximate_dlayer_dparams(layer, inputs, h):
    """
    https://en.wikipedia.org/wiki/Finite_difference#Relation_with_derivatives
    :param layer:
    :param inputs:
    :param h:
    :return:
    """
    params = layer.get_params()
    n_params = params.size
    dlayer_dparams = np.zeros((1, n_params))
    for param_id in range(n_params):
        f1 = layer.forward(inputs)  # scalar
        params[0, param_id] += h
        layer.set_params(params)
        print(params)
        f2 = layer.forward(inputs)  # 1 X n_outputs
        params[0, param_id] -= h
        layer.set_params(params)
        dlayer_dparams[:, param_id] = (f2 - f1) / h
    return dlayer_dparams
