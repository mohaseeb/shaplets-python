# shaplets
Python implementation of [the Learning Time-Series Shapelets method by Josif Grabocka et al.](http://www.ismll.uni-hildesheim.de/pub/pdfs/grabocka2014e-kdd.pdf), that learns a shapelet-based time-series classifier with gradient descent. 

This implementation views the model as a layered graph, where each layer 
implements a forward, backword and parameters update methods (see below 
diagram). This abstraction simplifies thinking about the algorithm and implementing it.
![Network diagram](lts-diag.png)

## Differences from the paper ##
* This implmenetation employs two (LinearLayer + SigmoidLayer) pairs instead of one (LinearLayer + SigmoidLayer) pair as in the paper (and shown in above diagram). This (using two pairs) has yielded improved results on some datasets. To have a similar setup as in the paper, simply update `shapelets_lts/classification/shapelet_models.py:LtsShapeletClassifier._init_network()`. 
* The loss in this implementation is an updated version of the one in the 
paper to allow training a single model for all the classes in the dataset (rather than one model/class). The impact on performance was not analysed. For details check `shapelets_lts/network/cross_entropy_loss_layer.py`

## Installation ##
```bash
git clone git@github.com:mohaseeb/shaplets-python.git
cd shaplets-python
pip install .
# or, for dev
# pip install .[dev]
```
## Usage ##
```python
from shapelets_lts.classification import LtsShapeletClassifier

# create an LtsShapeletClassifier instance
classifier = LtsShapeletClassifier(
    K=20,
    R=3,
    L_min=30,
    epocs=50,
    lamda=0.01,
    eta=0.01,
    shapelet_initialization='segments_centroids',
    plot_loss=True
)

# train the classifier. 
# train_data.shape -> (# train samples X time-series length) 
# train_label.shape -> (# train samples)
classifier.fit(train_data, train_label, plot_loss=True)

# evaluate on test data. 
# test_data.shape -> (# test samples X time-series length)
prediction = classifier.predict(test_data)

# retrieve the learnt shapelets
shapelets = classifier.get_shapelets()


# and plot sample shapelets
from shapelets_lts.util import plot_sample_shapelets
plot_sample_shapelets(shapelets=shapelets, sample_size=36)
```
Also have a look at example.py. For a stable training, make sure all the features in dataset are [standardized](https://en.wikipedia.org/wiki/Feature_scaling#Standardization) (i.e. each has zero mean and unit variance).

Example plot from plot_sample_shapelets.
![sample_shapelets](sample_shapelets.png)
