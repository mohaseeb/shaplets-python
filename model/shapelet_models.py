from layers import SoftMinLayer
from layers import AggregationLayer
from layers import LinearLayer
from layers import SigmoidLayer
from layers import CrossEntropyLossLayer
from network import Network
from util import utils
import numpy as np
import matplotlib.pyplot as pyplot


class LtsShapeletClassifier:
    def __init__(self, K, R, L_min, alpha=-100, learning_rate=0.01, regularization_parameter=0.01, epocs=10):
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
        self.plot_loss = None
        self.loss_fig = None

    def fit(self, train_data, train_labels, plot_loss=False):
        self.train_data = train_data
        self.train_labels = utils.get_one_active_representation(train_labels)
        self.train_size, self.output_size = self.train_labels.shape
        self.plot_loss = plot_loss
        self._init_network()
        self._train_network()

    def predict(self, test_data):
        self.network.remove_loss_layer()
        predicted_labels = np.zeros((test_data.shape[0], 1))
        for i in range(test_data.shape[0]):
            predicted_probabilities = self.network.forward(np.array([test_data[i, :]]), None)
            predicted_labels[i, 0] = np.argmax(predicted_probabilities)
        return predicted_labels + 1

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

    def _create_shapelets_aggregation_layer(self):
        # Shapelets are included in SoftMinLayers
        min_soft_layers = []
        for k in range(self.K):
            for r in range(1, self.R + 1):
                # TODO use the segments top K-mean centroids to initialize the shapelets
                min_soft_layers.append(
                    SoftMinLayer(np.random.normal(loc=0, scale=1, size=(1, r * self.L_min)), self.eta, self.alpha))
        # shapelets aggregation layer
        aggregator = AggregationLayer(min_soft_layers)
        return aggregator

    def _train_network(self):
        loss = np.zeros((1, self.epocs * self.train_size))
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
            # print current loss info
            print("epoc=" + str(epoc) + "/" + str(self.epocs) + " (iteration=" + str(iteration) + ") loss=" + str(l))
            # plot if needed
            if self.plot_loss:
                self._plot_loss(loss, epoc)
        if self.plot_loss:
            pyplot.savefig('loss.jpg')

    def _plot_loss(self, loss, epocs):
        if self.loss_fig is None:
            self.loss_fig = pyplot.figure()
            pyplot.xlabel("epoc")
            pyplot.ylabel("loss")
            pyplot.ion()
        pyplot.plot(range(epocs + 1), loss[0, 0:epocs + 1], color='blue')
        pyplot.pause(0.05)
