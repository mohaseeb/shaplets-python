from layers.soft_min_layer import SoftMinLayer
from layers.aggregation_layer import AggregationLayer
from layers.linear_layer import LinearLayer
from layers.sigmoid_layer import SigmoidLayer
from layers.cross_entropy_loss_layer import CrossEntropyLossLayer
from network import Network

import numpy as np


class LtsShapeletClassifier:
    def __init__(self, K, R, L_min, alpha=-100, learning_rate=0.01, regularization_parameter=0.01, epocs=10,
                 loss_plot=False):
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
        # Hyper parameters
        self.epocs = epocs
        self.eta = learning_rate
        self.lamda = regularization_parameter
        # other
        self.network = None
        self.loss_plot = loss_plot

    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.train_size, self.output_size = train_labels.shape
        self._init_network()
        self._train_network()

    def _init_network(self):
        self.network = Network()
        # shapelets layer
        self.network.add_layer(self._create_shapelets_aggregation_layer())
        # linear layer
        self.network.add_layer(LinearLayer(self.n_shapelets, self.output_size, self.eta, self.lamda, self.train_size),
                               regularized=True)
        # sigmoid layer
        self.network.add_layer(SigmoidLayer(self.output_size))
        # loss layer
        self.network.add_layer(CrossEntropyLossLayer(self.lamda, self.train_size))

    def _train_network(self):
        loss = np.zeros((1, self.epocs * self.train_size))
        for epoc in range(self.epocs):
            for sample_id in range(self.train_size):
                sample = np.array([self.train_data[sample_id]])
                label = np.array([self.train_labels[sample_id]])
                # perform a forward pass
                loss[0, epoc * self.train_size + sample_id] = self.network.forward(sample, label)
                # perform a backward pass
                self.network.backward()
                # perform a parameter update
                self.network.update_params()
            # print current loss info
            print("epoc=" + str(epoc) + " loss=" + str(loss[0, (epoc + 1) * self.train_size - 1]))
            # plot if needed
            if self.loss_plot:
                self._plot_loss(loss)

    def _create_shapelets_aggregation_layer(self):
        # Shapelets are included in SoftMinLayers
        min_soft_layers = []
        for k in range(self.K):
            for r in range(1, self.R + 1):
                # TODO use the top K-mean centroids to initialize the shapelets
                min_soft_layers.append(
                    SoftMinLayer(np.random.normal(loc=0, scale=1, size=(1, r * self.L_min)), self.eta, self.alpha))
        # shapelets aggregation layer
        aggregator = AggregationLayer(min_soft_layers)
        return aggregator

    def _plot_loss(self, loss):
        pass
