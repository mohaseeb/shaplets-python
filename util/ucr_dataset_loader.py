from numpy import genfromtxt
import numpy as np


def _get_one_active_labels(labels):
    classes = np.unique(labels)
    one_active_labels = np.zeros((labels.size, classes.size))
    for label_id in range(labels.size):
        one_active_labels[label_id, np.where(classes == labels[label_id])] = 1
    return one_active_labels


def load_dataset(dataset_name, dataset_folder):
    train_file_path = dataset_folder + "/" + dataset_name + "/" + dataset_name + "_TRAIN"
    test_file_path = dataset_folder + "/" + dataset_name + "/" + dataset_name + "_TEST"
    # read training data into numpy arrays
    train_raw_arr = genfromtxt(train_file_path, delimiter=',')
    train_labels = _get_one_active_labels(train_raw_arr[:, 0])
    train_data = train_raw_arr[:, 1:]
    # read test data into numpy arrays
    test_raw_arr = genfromtxt(test_file_path, delimiter=',')
    test_labels = _get_one_active_labels(test_raw_arr[:, 0])
    test_data = test_raw_arr[:, 1:]
    return train_data, train_labels, test_data, test_labels
