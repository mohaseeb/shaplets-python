from util import ucr_dataset_loader
from model import LtsShapeletClassifier
from sklearn.metrics import confusion_matrix, classification_report


def _evaluate_LtcShapeletClassifier(dataset_name, base_folder):
    # load the data
    train_data, train_label, test_data, test_label = ucr_dataset_loader.load_dataset(dataset_name, base_folder)
    # create a classifier
    classifier = LtsShapeletClassifier(K=2, R=7, L_min=10, epocs=100, regularization_parameter=0.001,
                                       learning_rate=0.01)
    # train the classifier
    classifier.fit(train_data, train_label, plot_loss=True)
    # evaluate on test data
    prediction = classifier.predict(test_data)
    print(classification_report(test_label, prediction))
    print(confusion_matrix(test_label, prediction))


if __name__ == '__main__':
    _evaluate_LtcShapeletClassifier('Gun_Point', 'data')
