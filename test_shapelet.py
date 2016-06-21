from Shapelet import Shapelet
import numpy as np


def test_shapelet_initialization():
    shapelet = Shapelet(np.ones((1, 10)))
    assert (shapelet.get_length() == 10)
    assert (shapelet.get_sequence().shape == (1, 10))


def test_shapelet_dist():
    shapelet = Shapelet(np.ones((1, 10)))
    assert (shapelet.dist(np.zeros((1, 10))) == 1)
