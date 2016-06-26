import numpy as np
from util import soft_min_layer_factory

from shapelets.network import AggregationLayer
from shapelets.util import utils


def test_forward():
    # create a set of soft_min_layers
    soft_min_layers = soft_min_layer_factory.create_soft_min_layers([3, 4, 5])
    # create an aggregation
    aggregator = AggregationLayer(soft_min_layers)
    # create a layer input
    Q = 10
    T = np.random.normal(loc=0, scale=1, size=(1, Q))
    # do a forward pass
    output = aggregator.forward(T)
    # compare to the truth output
    output_truth = np.array([[layer.forward(T) for layer in soft_min_layers]])
    assert (np.array_equal(output, output_truth))


def test_backward():
    layer_sizes = [3, 4, 5]
    n_outputs = len(layer_sizes)
    # create a set of soft_min_layers
    soft_min_layers = soft_min_layer_factory.create_soft_min_layers(layer_sizes)
    # create an aggregation
    aggregator = AggregationLayer(soft_min_layers)
    # create a layer input
    Q = 10
    T = np.random.normal(loc=0, scale=1, size=(1, Q))
    # create a dL_dout
    dL_dout = np.random.normal(loc=0, scale=1, size=(1, n_outputs))
    # do a forward and backward passes
    aggregator.forward(T)
    dL_dparams = aggregator.backward(dL_dout)
    # verify dL_dS ######
    dout_dparams_truth = utils.approximate_derivative_wrt_params(aggregator, T, n_outputs, h=0.00001)
    dL_dparams_truth = np.dot(dL_dout, dout_dparams_truth)
    result = np.isclose(dL_dparams, dL_dparams_truth, rtol=1e-05, atol=1e-03)
    assert result.all()
