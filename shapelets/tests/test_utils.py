import numpy as np

from shapelets.util import utils


def test_get_centroids():
    cluster_size = 5
    n_dims = 2
    n_clusters = 3
    cluster_1_data = np.random.normal(loc=0, scale=1, size=(cluster_size, n_dims))
    cluster_2_data = np.random.normal(loc=5, scale=1, size=(cluster_size, n_dims))
    cluster_3_data = np.random.normal(loc=9, scale=1, size=(cluster_size, n_dims))
    data = np.concatenate((cluster_1_data, cluster_2_data, cluster_3_data), axis=0)
    centroids = utils.get_centroids(data, n_clusters)
    assert (centroids.shape == (n_clusters, n_dims))


def test_segment_dataset():
    I = 2  # number of time-series
    Q = 4  # time series size
    L = 2  # segment length
    J = Q - L + 1  # segments per time-series
    data = np.random.normal(loc=5, scale=1, size=(I, Q))
    S = utils.segment_dataset(data, 2)  # segment
    assert (S.shape == (J * I, L))
    assert (np.array_equal(S[I * J - 1], data[I - 1, Q - L:]))


def test_get_centroids_of_segments():
    n_samples = 50
    n_dims = 2
    n_clusters = 2
    cluster_1_data = np.random.normal(loc=0, scale=0.01, size=(n_samples, n_dims))
    cluster_2_data = np.random.normal(loc=5, scale=0.01, size=(n_samples, n_dims))
    data = np.concatenate((cluster_1_data, cluster_2_data), axis=1)
    centroids = utils.get_centroids_of_segments(data, n_dims, n_clusters + 1)
    # centroids should be close to (0,0) (0,5) (5,5)
    centroid1_occur = 0
    centroid2_occur = 0
    centroid3_occur = 0
    for centroid in centroids:
        if np.isclose(centroid, (0, 0), rtol=1e-05, atol=1e-02).all():
            centroid1_occur += 1
        if np.isclose(centroid, (0, 5), rtol=1e-05, atol=1e-02).all():
            centroid2_occur += 1
        if np.isclose(centroid, (5, 5), rtol=1e-05, atol=1e-02).all():
            centroid3_occur += 1
    assert (centroid1_occur == centroid2_occur == centroid3_occur == 1)
