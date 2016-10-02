import cPickle as pickle
import os
import numpy as np
import sys
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lasagne
import theano
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def float32(k):
    return np.cast['float32'](k)

class PlotLosses(object):
    def __init__(self, figsize=(8, 6)):
        plt.plot([],[])
    def __call__(self, nn, train_history):
        train_loss=np.array([i["train_loss"] for i in nn.train_history_])
        valid_loss=np.array([i["valid_loss"] for i in nn.train_history_])
        
        plt.gca().cla()
        plt.plot(train_loss, label="train")
        plt.plot(valid_loss, label="test")
        
        plt.legend()
        plt.draw()

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name=name
        self.start, self.stop=start, stop
        self.ls=None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls=np.linspace(self.start, self.stop, nn.max_epochs)
        epoch=train_history[-1]['epoch']
        new_value=float32(self.ls[epoch-1])
        getattr(nn, self.name).set_value(new_value)

def load_dataset():
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    return X_train, y_train, X_val, y_val, X_test, y_test



def unpickle(file):
    f = open(file, 'rb')
    dict = pickle.load(f)
    f.close()
    return dict


net = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('drop1', layers.DropoutLayer),
            ('dense1', layers.DenseLayer),
            ('drop2', layers.DropoutLayer),
            ('output', layers.DenseLayer)
            ],
    input_shape=(None, 1, 28, 28),

    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),

    maxpool1_pool_size=(2, 2),

    conv2d2_num_filters=32,
    conv2d2_filter_size=(5, 5),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d2_W=lasagne.init.GlorotUniform(),

    maxpool2_pool_size=(2, 2),

    drop1_p=0.5,

    dense1_num_units=256,
    dense1_nonlinearity=lasagne.nonlinearities.rectify,

    drop2_p=0.5,

    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,

    update=nesterov_momentum,
    update_momentum=theano.shared(float32(0.9)),
    update_learning_rate=theano.shared(float32(0.01)),
    on_epoch_finished=[AdjustVariable('update_learning_rate',
                                      start=0.03, 
                                      stop=0.0001), 
                       AdjustVariable('update_momentum',
                                      start=0.9,
                                      stop=0.999)],
    max_epochs=10,
    verbose=True,
    
)

X_train, y_train, X_val, y_val, X_test, y_test=load_dataset()

network = net.fit(X_train, y_train)


print classification_report(y_test, predictions)
print accuracy_score(y_test, predictions)