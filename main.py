from util import ucr_dataset_loader
from model.shapelet_models import LtsShapeletClassifier


def _evaluate_LtcShapeletClassifier(dataset_name, base_folder):
    # load the data
    train_data, train_label, test_data, test_label = ucr_dataset_loader.load_dataset(dataset_name, base_folder)
    # create a classifier
    classifier = LtsShapeletClassifier(K=2, R=3, L_min=10, epocs=100)
    # train the classifier
    classifier.fit(train_data, train_label)


if __name__ == '__main__':
    _evaluate_LtcShapeletClassifier('Gun_Point', 'data')
