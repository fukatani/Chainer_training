#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      rf
#
# Created:     11/08/2015
# Copyright:   (c) rf 2015
# Licence:     <your licence>
# ref http://aidiary.hatenablog.com/entry/20151005/1444051251
#-------------------------------------------------------------------------------

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import sys
import pickle

#constructing newral network
class Mychain(object):
    def forward(self, x_data, y_data, train=True):
        x, t = Variable(x_data), Variable(y_data)
        h1 = F.dropout(F.relu(self.model.l1(x)),  train=train)
        h2 = F.dropout(F.relu(self.model.l2(h1)), train=train)
        y  = self.model.l3(h2)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def set_model(self):
        try:
            with open("self.model", "r") as f:
                print('load from pickled data.')
                self.model = pickle.load(f)
        except (IOError, EOFError):
            input_matrix_size = 784
            output_matrix_size = 10
            n_units   = 200
            self.model = FunctionSet(l1=F.Linear(input_matrix_size, n_units),
                            l2=F.Linear(n_units, n_units),
                            l3=F.Linear(n_units, output_matrix_size))

    def set_optimizer(self):
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())

    def learning(self, train_data_size, batchsize, n_epoch):
        sample = self.sample
        optimizer = self.optimizer

        x_train, x_test = np.split(sample.data,   [train_data_size])
        y_train, y_test = np.split(sample.target, [train_data_size])
        test_data_size = y_test.size

        train_loss = []
        train_acc  = []
        test_loss = []
        test_acc  = []

        # Learning loop
        for epoch in xrange(1, n_epoch+1):
            print 'epoch', epoch

            # training
            perm = np.random.permutation(train_data_size)
            sum_accuracy = 0
            sum_loss = 0

            for i in xrange(0, train_data_size, batchsize):
                x_batch = x_train[perm[i:i+batchsize]]
                y_batch = y_train[perm[i:i+batchsize]]

                # initialize gradient
                optimizer.zero_grads()
                # forward and calculate error
                loss, acc = self.forward(x_batch, y_batch)
                # calc grad by back prop
                loss.backward()
                optimizer.update()

                train_loss.append(loss.data)
                train_acc.append(acc.data)
                sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
                sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

            # display accuracy for training
            print 'train mean loss={}, accuracy={}'.format(sum_loss / train_data_size, sum_accuracy / train_data_size)

            # evaluation
            sum_accuracy = 0
            sum_loss     = 0
            for i in xrange(0, test_data_size, batchsize):
                x_batch = x_test[i:i+batchsize]
                y_batch = y_test[i:i+batchsize]

                # calc accuracy for test
                loss, acc = self.forward(x_batch, y_batch, train=False)
                test_loss.append(loss.data)
                test_acc.append(acc.data)
                sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
                sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

            # display accuracy for test
            print('test  mean loss={}, accuracy={}'.format(sum_loss / test_data_size, sum_accuracy / test_data_size))

        # display accuracy for test as graph
        plt.style.use('ggplot')
        plt.figure(figsize=(8,6))
        plt.plot(range(len(train_acc)), train_acc)
        plt.plot(range(len(test_acc)), test_acc)
        plt.legend(["train_acc","test_acc"],loc=4)
        plt.title("Accuracy of digit recognition.")
        plt.plot()
        plt.show()

    def set_sample(self):
        print('fetch MNIST dataset')
        sample = fetch_mldata('MNIST original')
        print('data fetch success')
        sample.data   = sample.data.astype(np.float32)
        sample.data  /= 255
        sample.target = sample.target.astype(np.int32)
        self.sample = sample

    def __init__(self, pickle_enable=False):
        # setup chainer
        self.set_sample()
        self.set_model()
        self.set_optimizer()

        #self.learning(train_data_size=60000, batchsize=100, n_epoch=3)
        self.learning(train_data_size=50000, batchsize=100, n_epoch=3)

        # Save final self.model
        if pickle_enable:
            pickle.dump(self.model, open('self.model', 'w'), -1)

if __name__ == '__main__':
    Mychain()
