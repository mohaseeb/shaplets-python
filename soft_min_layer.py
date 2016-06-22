import numpy as np


class SoftMinLayer:
    def __init__(self, sequence, alpha=-100):
        """

        :param shapelet:
        """
        self.S = sequence
        self.L = np.size(sequence, 1)
        self.alpha = alpha
        # layer input holder
        self.current_input = None
        # layer output holder
        self.current_output = None
        # derivative of Loss w.r.t. inputs
        self.dL_dinput = None
        # derivative of Loss w.r.t. weights
        self.dL_dW = None
        # derivative of Loss w.r.t. biases
        self.dL_dW_0 = None

    def forward(self, layer_input):
        self.current_input = layer_input
        self.current_output = self.dist_soft_min(self.current_input)
        return self.current_output

    def backward(self, dL_dout):
        """

        :param dL_dout:
        :return: dL_dS (1 X self.shapelet.get_length())
        """
        pass

    def dist_soft_min(self, T):
        Q = T.size
        J = Q - self.L
        M_numerator = 0
        M_denominator = 0
        # for each segment of T
        for j in range(J + 1):
            D = self.dist_sqr_error(T[0, j:j + self.L])
            xi = np.exp(self.alpha * D)
            M_numerator += D * xi
            M_denominator += xi
        return M_numerator / M_denominator

    def dist_sqr_error(self, T):
        """

        :param T:
        :return:
        """
        dist = (T - self.S) ** 2
        dist = np.sum(dist) / self.L
        return dist
