from __future__ import division

import matplotlib.pyplot as pyplot
import numpy as np
import copy

from shapelets.network import AggregationLayer
from shapelets.network import CrossEntropyLossLayer
from shapelets.network import LinearLayer
from shapelets.network import Network
from shapelets.network import SigmoidLayer
from shapelets.network import SoftMinLayer
from shapelets.util import utils


class LtsShapeletClassifier:
    def __init__(self, K, R, L_min, alpha=-100, learning_rate=0.01, regularization_parameter=0.01, epocs=10,
                 shapelet_initialization=None):
        """

        :param K: number of shapelets
        :param R: scales of shapelet lengths
        :param L_min: minimum shapelet length
        """
        # Shapelet related
        self.K = K
        self.R = R
        self.n_shapelets = self.K * self.R
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
        self.eta = learning_rate
        self.lamda = regularization_parameter
        # other
        self.network = None
        self.shapelet_initialization = shapelet_initialization
        self.plot_loss = None
        self.loss_fig = None

    def fit(self, train_data, train_labels, valid_data, valid_labels, plot_loss=False):
        self.train_data = train_data
        self.train_labels = utils.get_one_active_representation(train_labels)
        self.valid_data = valid_data
        self.valid_labels = valid_labels
        self.train_size, self.output_size = self.train_labels.shape
        self.plot_loss = plot_loss
        self._init_network()
        self._train_network()

    def predict(self, test_data):
        tmp_network = copy.deepcopy(self.network)
        self.network.remove_loss_layer()
        predicted_labels = np.zeros((test_data.shape[0], 1))
        for i in range(test_data.shape[0]):
            predicted_probabilities = self.network.forward(np.array([test_data[i, :]]), None)
            predicted_labels[i, 0] = np.argmax(predicted_probabilities)
        self.network = tmp_network
        return predicted_labels

    def _init_network(self):
        print('Network initialization ...')
        self.network = Network()
        # shapelets layer
        self.network.add_layer(self._get_shapelets_layer())
        # linear layer
        self.network.add_layer(LinearLayer(self.n_shapelets, self.output_size, self.eta, self.lamda, self.train_size),
                               regularized=True)
        # sigmoid layer
        self.network.add_layer(SigmoidLayer(self.output_size))
        # loss layer
        self.network.add_layer(CrossEntropyLossLayer(self.lamda, self.train_size))

    def _get_shapelets_layer(self):
        if self.shapelet_initialization == 'segments_centroids':
            return self._create_shapelets_layer_segments_centroids()
        else:
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
