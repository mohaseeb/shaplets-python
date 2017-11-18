# shaplets
Python implementation of [the Learning Time-Series Shapelets method by Josif Grabocka et al.](http://www.ismll.uni-hildesheim.de/pub/pdfs/grabocka2014e-kdd.pdf), that learns a shapelet-based time-series classifier with gradient descent. 

This implementation view the model as a layered network, where each layer implements a forward, a backword and parameters update methods. This abstraction makes it easy to port the implementation to a frameworks like Torch or Tensorflow. Check the diagram at the bottom of this page.

Note, the loss in this implementation is an updated version of the one in the paper to allow training a single network for all the classes in the dataset (rather than one network/class). The impact on performance was not estimated. For details check shapelets/network/cross_entropy_loss_layer.py

## Installation ##
```bash
git clone git@github.com:mohaseeb/shaplets-python.git
cd shaplets-python
pip install .
```
## Usage ##
```python
from shapelets_lts.classification import LtsShapeletClassifier
# create an LtsShapeletClassifier instance
classifier = LtsShapeletClassifier(
    K=20, 
    R=3, 
    L_min=30, 
    epocs=2, 
    regularization_parameter=0.01,
    learning_rate=0.01, 
    shapelet_initialization='segments_centroids'
)
# train the classifier. train_data (a numpy matrix) shape is (# train samples X time-series length), train_label (a numpy matrix) is (# train samples X 1).
classifier.fit(train_data, train_label, plot_loss=True)
# evaluate on test data. test_data (a numpy matrix) shape is (# test samples X time-series length)
prediction = classifier.predict(test_data)
```
Also have a look at example.py in [the implementation](https://github.com/mohaseeb/shaplets-python). For a stable training, make sure all the features in dataset are [standardized](https://en.wikipedia.org/wiki/Feature_scaling#Standardization) (i.e. each has zero mean and unit variance).

Although I believe the architecture is good, I think the implementation is way from optimal, and there is plenty of room for improvement. Off the top of my head, the usage of python arrays/lists has to be improved.

For stable training, make sure all the timeseries in the dataset are [standardized](https://en.wikipedia.org/wiki/Feature_scaling#Standardization) (i.e. each has zero mean and unit variance). 

## Model diagram
![Network diagram](lts-diag.png)
