from __future__ import division, print_function

from os.path import expanduser

from sklearn.metrics import classification_report

from shapelets_lts.classification import LtsShapeletClassifier
from shapelets_lts.util import ucr_dataset_loader, plot_sample_shapelets

"""
This example uses dataset from the UCR archive "UCR Time Series Classification
Archive" format.  

- Follow the instruction on the UCR page 
(http://www.cs.ucr.edu/~eamonn/time_series_data/) to download the dataset. You 
need to be patient! :) 
- Update the vars below to point to the correct dataset location in your  
machine.

Otherwise update _load_train_test_datasets() below to return your own dataset.
"""

ucr_dataset_base_folder = expanduser('~/ws/data/UCR_TS_Archive_2015/')
ucr_dataset_name = 'Gun_Point'


def main():
    # load the data
    print('\nLoading data...')
    x_train, y_train, x_test, y_test = _load_train_test_datasets()

    # create a classifier
    Q = x_train.shape[1]
    K = int(0.15 * Q)
    L_min = int(0.2 * Q)
    clf = LtsShapeletClassifier(
        K=K,
        R=3,
        L_min=L_min,
        epocs=30,
        lamda=0.01,
        eta=0.01,
        shapelet_initialization='segments_centroids',
        plot_loss=True
    )

    # train the classifier
    print('\nTraining...')
    clf.fit(x_train, y_train)

    # evaluate on test data
    print('\nEvaluating...')
    y_pred = clf.predict(x_test)
    print(
        'classification report...\n{}'
        ''.format(classification_report(y_true=y_test, y_pred=y_pred))
    )

    # plot sample shapelets
    print('\nPlotting sample shapelets...')
    plot_sample_shapelets(shapelets=clf.get_shapelets(), sample_size=36)


def _load_train_test_datasets():
    """
    :return: numpy arrays, train_data, train_labels, test_data, test_labels
        train_data and test_data shape is: (n_samples, n_features)
        train_labels and test_labels shape is: (n_samples)
    """
    return ucr_dataset_loader.load_dataset(
        dataset_name=ucr_dataset_name,
        dataset_folder=ucr_dataset_base_folder
    )


if __name__ == '__main__':
    main()
