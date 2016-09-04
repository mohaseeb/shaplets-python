from __future__ import division

import copy

import matplotlib.pyplot as pyplot
import numpy as np
from sklearn.base import BaseEstimator

from shapelets.network import AggregationLayer
from shapelets.network import CrossEntropyLossLayer
from shapelets.network import LinearLayer
from shapelets.network import Network
from shapelets.network import SigmoidLayer
from shapelets.network import SoftMinLayer
from shapelets.util import utils

"""
This class implements the sklearn estimator interface, so sklearn tools like GridsearchCV can be used
"""
class LtsShapeletClassifier(BaseEstimator):
    def __init__(self, K=20, R=3, L_min=30, alpha=-100, eta=0.01, lamda=0.01, epocs=10,
                 shapelet_initialization='segments_centroids', plot_loss=False):
        """

        :param K: number of shapelets
        :param R: scales of shapelet lengths
        :param L_min: minimum shapelet length
        """
        # Shapelet related
        self.K = K
        self.R = R
        self.n_shapelets = None
        self.L_min = L_min
        self.alpha = alpha
        # Training data related
        self.train_data = None
        self.train_labels = None
        self.output_size = None
        self.train_size = None
        # validation data
        self.valid_data = None
        self.valid_labels = None
        # Hyper parameters
        self.epocs = epocs
        self.eta = eta  # learning rate
        self.lamda = lamda  # regularization parameter
        # other
        self.network = None
        self.shapelet_initialization = shapelet_initialization
        self.plot_loss = plot_loss
        self.loss_fig = None

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        self.n_shapelets = self.K * self.R
        self.train_data = X
        self.train_labels = utils.get_one_active_representation(y)
        self.train_size, self.output_size = self.train_labels.shape
        self._init_network()
        self._train_network()
        return self

    def predict(self, X):
        tmp_network = copy.deepcopy(self.network)
        tmp_network.remove_loss_layer()
        predicted_labels = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            predicted_probabilities = tmp_network.forward(np.array([X[i, :]]), None)
            predicted_labels[i, 0] = np.argmax(predicted_probabilities)
        del tmp_network
        return predicted_labels

    def _init_network(self):
        print('Network initialization ...')
        self.network = Network()
        # shapelets layer
        self.network.add_layer(self._get_shapelets_layer())

        # linear layer
        self.network.add_layer(LinearLayer(self.n_shapelets, 24, self.eta, self.lamda, self.train_size),
                               regularized=True)
        # sigmoid layer
        self.network.add_layer(SigmoidLayer(24))

        # linear layer
        self.network.add_layer(LinearLayer(24, self.output_size, self.eta, self.lamda, self.train_size),
                               regularized=True)
        # sigmoid layer
        self.network.add_layer(SigmoidLayer(self.output_size))
        # loss layer
        self.network.add_layer(CrossEntropyLossLayer(self.lamda, self.train_size))

    def _get_shapelets_layer(self):
        if self.shapelet_initialization == 'segments_centroids':
            print('Using training data to initialize shaplets')
            return self._create_shapelets_layer_segments_centroids()
        else:
            print('Randomly initialize shapelets')
            return self._create_shapelets_layer_random()

    def _create_shapelets_layer_segments_centroids(self):
        # Shapelets are included in SoftMinLayers
        min_soft_layers = []
        for r in range(1, self.R + 1):
            L = r * self.L_min
            top_K_centroids_scale_r = utils.get_centroids_of_segments(self.train_data, L, self.K)
            for centroid in top_K_centroids_scale_r:
                min_soft_layers.append(
                    SoftMinLayer(np.array([centroid]), self.eta, self.alpha))
        # shapelets aggregation layer
        aggregator = AggregationLayer(min_soft_layers)
        return aggregator

    def _create_shapelets_layer_random(self):
        # Shapelets are included in SoftMinLayers
        min_soft_layers = []
        for k in range(self.K):
            for r in range(1, self.R + 1):
                min_soft_layers.append(
                    SoftMinLayer(np.random.normal(loc=0, scale=1, size=(1, r * self.L_min)), self.eta, self.alpha))
        # shapelets aggregation layer
        aggregator = AggregationLayer(min_soft_layers)
        return aggregator

    def _train_network(self):
        print('Training ...')
        loss = np.zeros((1, self.epocs * self.train_size))
        valid_accur = np.zeros((1, self.epocs * self.train_size))
        iteration = 0
        for epoc in range(self.epocs):
            l = 10000
            for sample_id in range(self.train_size):
                sample = np.array([self.train_data[sample_id]])
                label = np.array([self.train_labels[sample_id]])
                # perform a forward pass
                l = self.network.forward(sample, label)
                # perform a backward pass
                self.network.backward()
                # perform a parameter update
                self.network.update_params()
                iteration += 1
            loss[0, epoc] = l
            # calculate accuracy in validation set
            if self.valid_data is None:
                valid_epoc_accur = 0
            else:
                valid_epoc_accur = np.sum(np.equal(self.predict(self.valid_data), self.valid_labels)) / \
                                   self.valid_labels.shape[0]
            valid_accur[0, epoc] = valid_epoc_accur
            # print current loss info
            print("epoc=" + str(epoc) + "/" + str(self.epocs - 1) + " (iteration=" + str(iteration) + ") loss=" + str(l)
                  + " validation accuracy=" + str(valid_epoc_accur))
            # plot if needed
            if self.plot_loss:
                self._plot_loss(loss, valid_accur, epoc)
        if self.plot_loss:
            pyplot.savefig('loss.jpg')

    def _plot_loss(self, loss, validation_acc, epocs):
        if self.loss_fig is None:
            self.loss_fig = pyplot.figure()
            pyplot.xlabel("epoc")
            pyplot.ylabel("loss/validation_acc")
            pyplot.ion()
        pyplot.plot(range(epocs + 1), loss[0, 0:epocs + 1], color='red', label='loss')
        pyplot.plot(range(epocs + 1), validation_acc[0, 0:epocs + 1], color='blue', label='validation accuracy')
        pyplot.pause(0.05)
