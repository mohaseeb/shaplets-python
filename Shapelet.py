from __future__ import division
import numpy as np


class Shapelet:
    """

    """

    def __init__(self, sequence):
        self.S = sequence
        self.L = np.size(sequence, 1)

    def get_sequence(self):
        return self.S

    def get_length(self):
        return self.L

    def dist(self, T):
        """

        :param T:
        :return:
        """
        dist = (T - self.S) ** 2
        dist = np.sum(dist) / self.L
        return dist


