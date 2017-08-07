from __future__ import division, print_function

from os.path import expanduser

from sklearn.metrics import confusion_matrix, classification_report

from shapelets.classification import LtsShapeletClassifier
from shapelets.util import ucr_dataset_loader

"""
This example expects a dataset in the "UCR Time Series Classification
Archive" format. UCR page (http://www.cs.ucr.edu/~eamonn/time_series_data/) 

Follow the instructions at the top of the page UCR page to download the 
UCR dataset.

Update the vars below to point to the correct dataset location in your machine.
"""

ucr_dataset_base_folder = expanduser('~/UCR_TS_Archive_2015')
ucr_dataset_name = 'Gun_Point'


def _evaluate_LtcShapeletClassifier(base_folder, dataset_name):
    # load the data
    train_data, train_label, test_data, test_label = (
        ucr_dataset_loader.load_dataset(dataset_name, base_folder)
    )

    # create a classifier (the parameter values as per Table1 for the GunPoint
    # dataset ). 200 epocs yielded 0.99
    #  accuracy
    Q = train_data.shape[1]
    K = int(0.15 * Q)
    L_min = int(0.2 * Q)
    classifier = LtsShapeletClassifier(
        K=K,
        R=3,
        L_min=L_min,
        epocs=200,
        lamda=0.01,
        eta=0.01,
        shapelet_initialization='segments_centroids',
        plot_loss=True
    )

    # train the classifier
    classifier.fit(train_data, train_label)

    # evaluate on test data
    prediction = classifier.predict(test_data)
    print(classification_report(test_label, prediction))
    print(confusion_matrix(test_label, prediction))


if __name__ == '__main__':
    _evaluate_LtcShapeletClassifier(ucr_dataset_base_folder, ucr_dataset_name)
