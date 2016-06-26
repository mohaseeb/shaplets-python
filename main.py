from util import ucr_dataset_loader
from model import LtsShapeletClassifier
from sklearn.metrics import confusion_matrix, classification_report


def _evaluate_LtcShapeletClassifier(dataset_name, base_folder):
    # load the data
    train_data, train_label, test_data, test_label = ucr_dataset_loader.load_dataset(dataset_name, base_folder)
    # create a classifier (the parameter values as per Table1 for the GunPoint dataset )
    Q = train_data.shape[1]
    K = int(0.15 * Q)
    L_min = int(0.2 * Q)
    classifier = LtsShapeletClassifier(K=K, R=3, L_min=L_min, epocs=2, regularization_parameter=0.1,
                                       learning_rate=0.01, shapelet_initialization='segments_centroids')
    # train the classifier
    classifier.fit(train_data, train_label, plot_loss=True)
    # evaluate on test data
    prediction = classifier.predict(test_data)
    print(classification_report(test_label, prediction))
    print(confusion_matrix(test_label, prediction))


if __name__ == '__main__':
    _evaluate_LtcShapeletClassifier('Gun_Point', 'data')
