import numpy as np
from soft_min_layer import SoftMinLayer


def test_forward():
    soft_layer = SoftMinLayer(np.ones((1, 5)))
    T = np.ones((1, 10)) + 1
    assert (soft_layer.forward(T) == 1)


def test_backward():
    assert (12 == 1)


def test_shapelet_dist_sqr_error():
    soft_layer = SoftMinLayer(np.ones((1, 10)))
    assert (soft_layer.dist_sqr_error(np.zeros((1, 10))) == 1)


def test_shapelet_dist_soft_min():
    soft_layer = SoftMinLayer(np.ones((1, 10)))
    T = np.ones((1, 10)) + 1
    assert (soft_layer.dist_soft_min(T) == 1)
