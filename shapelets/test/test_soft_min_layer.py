import numpy as np

from shapelets.network import SoftMinLayer
from shapelets.util import utils


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
    dM_dS_truth = utils.approximate_derivative_wrt_params(soft_layer, T, 1, h=0.00001)
    dL_dS_truth = dL_dM * dM_dS_truth
    result = np.isclose(dL_dS, dL_dS_truth, rtol=1e-05, atol=1e-04)
    assert result.all()


def test_shapelet_dist_sqr_error():
    soft_layer = SoftMinLayer(np.ones((1, 10)))
    assert (soft_layer.dist_sqr_error(np.zeros((1, 10))) == 1)
