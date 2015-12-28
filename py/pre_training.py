#-------------------------------------------------------------------------------
# Name:        deep_auto_encoder
# Purpose:
#
# Author:      rf
#
# Created:     11/08/2015
# Copyright:   (c) rf 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from chainer import ChainList, optimizers
import numpy as np

#constructing newral network
class PretrainingChain(ChainList):
    def __init__(self, n_units, epoch=20, batch_size=10):
        ChainList.__init__(self)
        self.n_units = n_units[0:-1]
        self.last_unit = n_units[-1]
        self.total_layer = len(n_units)
        self.collect_child_model()
        self.set_optimizer()
        self.epoch = epoch
        self.batch_size = batch_size

    def set_optimizer(self):
        self.optimizer = optimizers.AdaDelta()
        self.optimizer.setup(self)

    def collect_child_model(self):
        self.child_models = []
        for i, n_unit in enumerate(self.n_units):
            if i == 0: continue
            self.child_models.append(ChildChainList(F.Linear(self.n_units[i-1], n_unit)))

    def forward(self, x_data, train=True):
        data = x_data
        for model in self:
            data = F.dropout(F.relu(model(data)), train=train)
        return data

    def pre_training(self, sample, test):
        now_sample = sample
        now_test = test
        for child in self.child_models:
            child.learn_as_autoencoder(now_sample, now_test)
            #add only first link
            self.add_link(child[0].copy())
            now_sample = self.forward(Variable(sample), False).data
            if len(test):
                now_test = self.forward(Variable(test), False).data
        self.add_last_layer()

    def add_last_layer(self):
        self.add_link(F.Linear(self.n_units[-1], self.last_unit))

    def loss_function(self, x, y):
        return F.softmax_cross_entropy(x, y)

    def learn(self, x_train, y_train, x_test, y_test, isClassification=False):
        train_size = x_train.shape[0]
        train_data_size = x_train.shape[1]

        for epoch in range(self.epoch):
            perm = np.random.permutation(train_size)
            train_loss = 0.
            test_loss = 0.
            test_accuracy = 0.
            for i in range(0, train_size, self.batch_size):
                x = Variable(x_train[perm[i:i+self.batch_size]])
                y = Variable(y_train[perm[i:i+self.batch_size]])
                self.zerograds()
                loss = self.loss_function(self.forward(x, train=True), y)
                loss.backward()
                self.optimizer.update()
                train_loss += loss.data * self.batch_size
            train_loss /= train_size

            if len(x_test):
                x = Variable(x_test)
                test_loss = self.loss_function(self.forward(x, train=False), y).data
                print('test_loss: ' + str(test_loss))
                if isClassification:
                    test_accuracy = F.accuracy(self.forward(x), t).data
                    print('test_accuracy: ' + str(test_accuracy))

class ChildChainList(ChainList):
    def __init__(self, link, epoch=20, batch_size=10):
        ChainList.__init__(self, link)
        self.optimizer = optimizers.AdaDelta()
        self.optimizer.setup(self)
        self.loss_function = F.mean_squared_error
        self.epoch = epoch
        self.batch_size = batch_size

    def forward(self, x_data, train):
        return F.dropout(F.relu(self[0](x_data)), train=train)

    def forward_as_autoencoder(self, x_data, train):
        h = self.forward(x_data, train)
        return F.dropout(F.relu(self[1](h)), train=train)

    def add_dummy_output_link(self, train_data_size):
        self.add_link(F.Linear(self[0].W.data.shape[0] , train_data_size))

    def learn_as_autoencoder(self, x_train, x_test=None):
        optimizer = self.optimizer
        train_size = x_train.shape[0]
        train_data_size = x_train.shape[1]
        self.add_dummy_output_link(train_data_size)

        for epoch in range(self.epoch):
            perm = np.random.permutation(train_size)
            train_loss = 0
            test_loss = 0
            test_accuracy = 0
            for i in range(0, train_size, self.batch_size):
                x = Variable(x_train[perm[i:i+self.batch_size]])
                self.zerograds()
                loss = self.loss_function(self.forward_as_autoencoder(x, train=True), x)
                loss.backward()
                self.optimizer.update()
                train_loss += loss.data * self.batch_size
            train_loss /= train_size

            if len(x_test):
                x = Variable(x_test)
                test_loss = self.loss_function(self.forward_as_autoencoder(x, train=False), x).data
        if test_loss != 0:
            print('Pre-training test loss: ' + str(test_loss))

def make_sample():
    from sklearn.datasets import fetch_mldata
    print('fetch MNIST dataset')
    sample = fetch_mldata('MNIST original')
    print('data fetch success')
    sample.data   = sample.data.astype(np.float32)
    sample.data  /= 255
    sample.target = sample.target.astype(np.int32)
    return sample

if __name__ == '__main__':
    train_data_size = 100
    sample = make_sample()
    perm = np.random.permutation(len(sample.data))
    x_train = sample.data[perm[0:train_data_size]]
    y_train = sample.target[perm[0:train_data_size]]
    x_test = sample.data[perm[train_data_size:2000]]
    y_test = sample.target[perm[train_data_size:2000]]
    pc = PretrainingChain([784,11,12,784])
    pc.pre_training(x_train, x_test)
    pc.learn(x_train, y_train, x_test, y_test, True)

