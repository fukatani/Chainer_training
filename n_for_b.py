#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      rf
#
# Created:     11/08/2015
# Copyright:   (c) rf 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import sys
import data_manager
import pickle
import os

#constructing newral network
class Mychain(object):
    def forward(self, x_data, y_data, train=True, answer=False):
        x, t = Variable(x_data), Variable(y_data)
        h1 = F.dropout(F.relu(self.model.l1(x)),  train=train)
        h2 = F.dropout(F.relu(self.model.l2(h1)), train=train)
        y  = self.model.l3(h2)
        if answer:
            return [np.argmax(data) for data in y.data]
        else:
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def set_model(self):
        try:
            with open("self.model", "r") as f:
                print('load from pickled data.')
                self.model = pickle.load(f)
        except (IOError, EOFError):
            n_units   = 200
            self.model = FunctionSet(l1=F.Linear(self.input_matrix_size, n_units),
                            l2=F.Linear(n_units, n_units),
                            l3=F.Linear(n_units, self.output_matrix_size))

    def set_optimizer(self):
        self.optimizer = optimizers.AdaDelta()
        #self.optimizer = optimizers.Adam(alpha=0.01)
        self.optimizer.setup(self.model.collect_parameters())

    def time_record(func):
        import datetime
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = datetime.datetime.today()
            result = func(*args, **kwargs)
            end = datetime.datetime.today()
            print('time elapsed ' + str(end-start))
        return wrapper

    @time_record
    def learning(self, train_data_size, batchsize, n_epoch):
        sample = self.sample
        optimizer = self.optimizer

        perm = np.random.permutation(len(sample.data))
        x_train = sample.data[perm[0:train_data_size]]
        y_train = sample.target[perm[0:train_data_size]]
        x_test = sample.data[perm[train_data_size:-1]]
        y_test = sample.target[perm[train_data_size:-1]]
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
                if self.is_clastering:
                    loss, acc = self.forward(x_batch, y_batch)
                else:
                    loss = self.forward(x_batch, y_batch)
                # calc grad by back prop
                loss.backward()
                optimizer.update()

                train_loss.append(loss.data)
                sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
                if self.is_clastering:
                    train_acc.append(acc.data)
                    sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

            # display accuracy for training
            if not self.is_clastering:
                print('train mean loss={}'.format(sum_loss / train_data_size))
            else:
                print('train mean loss={}, accuracy={}'.format(sum_loss / train_data_size, sum_accuracy / train_data_size))
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
        if self.plot_enable:
            self.disp_plot(train_acc, test_acc)

    def disp_plot(self, train_acc, test_acc):
        # display accuracy for test as graph
        plt.style.use('ggplot')
        plt.figure(figsize=(8,6))
        plt.plot(range(len(train_acc)), train_acc)
        plt.plot(range(len(test_acc)), test_acc)
        plt.legend(["train_acc","test_acc"],loc=4)
        plt.title("Accuracy of inoue/fukatani recognition.")
        plt.plot()

        if self.save_as_png:
            plt.savefig('./Image/learn_result.png')
        else:
            plt.show()

    def set_sample(self):
        print('fetch data')
        self.sample = data_manager.data_manager('./numbers', 1000, 'overlap', True).make_sample()
        self.input_matrix_size = self.sample.input_matrix_size
        self.output_matrix_size = self.sample.output_matrix_size

    def disp_w(self):
        plt.close('all')
        plt.style.use('ggplot')
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.plot(self.model.l1.W[i]), np.arange(0, len(self.model.l1.W[i]))
        plt.title("Weight of l1.")
        plt.plot()
        plt.show()

    def extract_test_sample(self, test_data_size=9):
        perm = np.random.permutation(self.sample.sample_size)
        x_batch = self.sample.data[perm[0:test_data_size]]
        y_batch = self.sample.target[perm[0:test_data_size]]
        return x_batch, y_batch

    def final_test(self, x_batch, y_batch=None):
        plt.close('all')
        plt.style.use('fivethirtyeight')
        #size = 28
        if y_batch is None:
            y_batch = np.zeros(len(x_batch))
        for i in range(x_batch.size / x_batch[0].size):
            x = x_batch[i:i+1]
            y = y_batch[i:i+1]
            recog_answer = self.forward(x, y, train=False, answer=True)[0]
            answer = y[0]

            plt.subplot(3, 3, i+1)
            plt.plot(np.arange(0, self.input_matrix_size, 1), x[0])
            plt.title(self.get_final_test_title(answer, recog_answer), size=8)
            plt.tick_params(labelbottom="off")
            plt.tick_params(labelleft="off")
        if self.save_as_png:
            plt.savefig('./Image/final_test.png')
        plt.show()

    def get_final_test_title(self, answer, recog_answer):
        return "ans=%d, recog=%d"%(answer, recog_answer)

    def __init__(self, pickle_enable=False, plot_enable=True, save_as_png=True, final_test_enable=True, is_clastering=True):
        # setup chainer
        self.set_sample()
        self.set_model()
        self.set_optimizer()
        self.plot_enable = plot_enable
        self.save_as_png = save_as_png
        self.is_clastering = is_clastering
        if save_as_png and not os.path.exists('./Image'):
            os.mkdir('./Image')

        self.learning(train_data_size=100, batchsize=10, n_epoch=10)
        #self.disp_w()
        if final_test_enable:
            x_batch, y_batch = self.extract_test_sample()
            self.final_test(x_batch, y_batch)

        # Save final self.model
        if pickle_enable:
            pickle.dump(self.model, open('self.model', 'w'), -1)

if __name__ == '__main__':
    Mychain()
