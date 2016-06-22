from __future__ import division
import numpy as np


class Shapelet:
    """

    """

    def __init__(self, sequence, alpha=-100):
        self.S = sequence
        self.L = np.size(sequence, 1)
        self.alpha = alpha

    def get_sequence(self):
        return self.S

    def get_length(self):
        return self.L

    def dist_sqr_error(self, T):
        """

        :param T:
        :return:
        """
        dist = (T - self.S) ** 2
        dist = np.sum(dist) / self.L
        return dist

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
