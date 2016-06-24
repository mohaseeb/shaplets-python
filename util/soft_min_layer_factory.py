from layers.soft_min_layer import SoftMinLayer
import numpy as np


def create_soft_min_layers(sizes):
    layers = []
    for size in sizes:
        layers.append(SoftMinLayer(np.ones((1, size))))
    return layers
