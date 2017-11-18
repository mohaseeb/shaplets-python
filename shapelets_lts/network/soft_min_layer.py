import numpy as np


class SoftMinLayer:
    def __init__(self, sequence, learning_rate=0.01, alpha=-100):
        """

        :type alpha:
        :param sequence:
        :param alpha:
        """
        self.S = sequence
        self.L = np.size(sequence, 1)
        self.alpha = alpha
        # learning rate
        self.eta = learning_rate
        # layer input holder
        self.T = None
        # layer output holder
        self.current_output = None
        # derivative of Loss w.r.t. shapelet values
        self.dL_dS = None
        # holder of pre-calculated values to speed up the calculations
        self.J = None  # number of segments in input time-series
        self.D = None  # (1 X J) distances between shapelet and the current time-series segments
        self.xi = None  # (1 X J)
        self.psi = None
        self.M = None  # soft minimum distance

    def forward(self, layer_input):
        self.T = layer_input
        self.M = self.dist_soft_min()
        return self.M

    def backward(self, dL_dout):
        """

        :param dL_dout:
        :return: dL_dS (1 X self.L)
        """
        # (1 X J): derivative of M (soft minimum) w.r.t D_j (distance between shapelet and the segment j of the
        # time-series)
        dM_dD = self.xi * (1 + self.alpha * (self.D - self.M)) / self.psi
        # (J X L) : derivative of D_j w.r.t. S_l (shapelet value at position l)
        dD_dS = np.zeros((self.J, self.L))
        for j in range(self.J):
            dD_dS[j, :] = 2 * (self.S - self.T[0, j:j + self.L]) / self.L
        # (1 X L) : derivative of M w.r.t. S_l
        dM_dS = np.dot(dM_dD, dD_dS)
        # (1 X L) : derivative of L w.r.t S_l. Note dL_dout is dL_dM
        self.dL_dS = dL_dout * dM_dS
        return self.dL_dS

    def dist_soft_min(self):
        Q = self.T.size
        self.J = Q - self.L + 1
        M_numerator = 0
        # for each segment of T
        self.D = np.zeros((1, self.J))
        self.xi = np.zeros((1, self.J))
        self.psi = 0
        for j in range(self.J):
            self.D[0, j] = self.dist_sqr_error(self.T[0, j:j + self.L])
            self.xi[0, j] = np.exp(self.alpha * self.D[0, j])
            M_numerator += self.D[0, j] * self.xi[0, j]
            self.psi += self.xi[0, j]
        M = M_numerator / self.psi
        return M

    def dist_sqr_error(self, T_j):
        """

        :param T:
        :return:
        """
        dist = (T_j - self.S) ** 2
        dist = np.sum(dist) / self.L
        return dist

    def get_params(self):
        """

        :return:
        """
        return self.S

    def set_params(self, param):
        """

        :param param:
        :return:
        """
        self.S = param

    def update_params(self):
        self.S -= self.eta * self.dL_dS

    def get_size(self):
        return self.L
