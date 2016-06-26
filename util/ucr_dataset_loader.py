from numpy import genfromtxt


def load_dataset(dataset_name, dataset_folder):
    train_file_path = dataset_folder + "/" + dataset_name + "/" + dataset_name + "_TRAIN"
    test_file_path = dataset_folder + "/" + dataset_name + "/" + dataset_name + "_TEST"
    # read training data into numpy arrays
    train_raw_arr = genfromtxt(train_file_path, delimiter=',')
    train_labels = train_raw_arr[:, 0]
    train_data = train_raw_arr[:, 1:]
    # read test data into numpy arrays
    test_raw_arr = genfromtxt(test_file_path, delimiter=',')
    test_labels = test_raw_arr[:, 0]
    test_data = test_raw_arr[:, 1:]
    return train_data, train_labels, test_data, test_labels
