import numpy as np

from shapelets.network.soft_min_layer import SoftMinLayer


def create_soft_min_layers(sizes):
    layers = []
    for size in sizes:
        layers.append(SoftMinLayer(np.ones((1, size))))
    return layers
