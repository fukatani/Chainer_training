#-------------------------------------------------------------------------------
# Name:        auto_encoder
# Purpose:
#
# Author:      rf
#
# Created:     11/08/2015
# Copyright:   (c) rf 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

#%matplotlib inline
from n_for_b import Mychain
from data_manager import data_manager, Abstract_sample
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import numpy as np

#constructing newral network
class Autoencoder(Mychain):
    def forward(self, x_data, y_data, train=True, answer=False):
        x, t = Variable(x_data), Variable(y_data)
        h1 = F.dropout(F.relu(self.model.l1(x)),  train=train)
        h2 = F.dropout(self.model.l2(h1), train=train)
        y  = self.model.l3(h2)
        if answer:
            return y.data, F.mean_squared_error(y, t)
        else:
            return F.mean_squared_error(y, t)#, F.accuracy(y, t)

    def set_sample(self):
        print('fetch data')
        self.train_sample, self.test_sample = dm_for_ae('./numbers'
            , 1000, self.train_size, attenate_flag=True, slide=4, keywords=self.keywords).make_sample()
        self.input_matrix_size = self.train_sample.input_matrix_size
        self.output_matrix_size = self.train_sample.output_matrix_size

    def get_final_test_title(self, answer, recog, loss):
        return str(loss.data)

    def final_test_plot(self, x, y):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(0, self.input_matrix_size, 1), x[0])
        plt.plot(np.arange(0, self.input_matrix_size, 1), y)

    def __init__(self, pickle_enable=False,
                 plot_enable=True,
                 save_as_png=True,
                 final_test_enable=True,
                 is_clastering=False,
                 train_size=100,
                 batch_size=10,
                 n_epoch=10,
                 n_units=200,
                 **keywords):
        self.keywords = keywords
        Mychain.__init__(self,
                         pickle_enable,
                         plot_enable,
                         save_as_png,
                         final_test_enable,
                         is_clastering,
                         train_size,
                         batch_size,
                         n_epoch,
                         n_units)

class dm_for_ae(data_manager):

    def offset_minus(self, data):
        return np.array([element - np.average(element) for element in data])

    def process_sample_backend_ae(func):
        import datetime
        from functools import wraps
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            train, test = func(self, *args, **kwargs)
            train.data  -= np.min(train.data)
            train.data  /= np.max(train.data)
            train.target = np.array(train.data)
            if self.denoised_enable: # Add noise
                train.data += (np.random.normal(size = train.data.shape) / self.noise_coef)

            test.data  -= np.min(test.data)
            test.data  /= np.max(test.data)
            test.target = np.array(test.data)
            if self.offset_cancel:
                train.data = self.offset_minus(train.data)
                train.target = self.offset_minus(train.target)
                test.data = self.offset_minus(test.data)
                test.target = self.offset_minus(test.target)
            return train, test
        return wrapper

    @process_sample_backend_ae
    def make_sample(self):
        """ [Functions]
            Make sample for analysis by chainer.
        """
        data_dict = self.get_data()
        sample_size = len(data_dict.keys())
        #train_size = 50

        #initialize
        train_data = np.zeros([self.train_size, self.data_size], dtype=np.float32)
        test_data = np.zeros([sample_size - self.train_size, self.data_size], dtype=np.float32)
        train_target = np.zeros([sample_size, self.data_size], dtype=np.float32)
        test_target = np.zeros([sample_size - self.train_size, self.data_size], dtype=np.float32)

        sample_index = 0
        if self.randomization:
            sample_data = np.array(data_dict.values())
            perm = np.random.permutation(sample_size)
            train_data = sample_data[perm[0:self.train_size]]
            test_data = sample_data[perm[self.train_size:]]
        elif self.order:
            for name, array in data_dict.items():
                if sample_index < self.train_size:
                    train_data[sample_index] = array
                else:
                    test_data[sample_index-self.train_size] = array
                sample_index += 1
        elif self.all_same:
            for name, array in data_dict.items():
                if sample_index < self.train_size:
                    train_data[sample_index] = data_dict.values()[sample_index % self.sample_kinds]
                else:
                    test_data[sample_index-self.train_size] = data_dict.values()[sample_index % self.sample_kinds]
                sample_index += 1
        else:
            for name, array in data_dict.items():
                if name[0:2] == 'fu' and train_data.shape[0] > sample_index:
                    train_data[sample_index] = array
                elif name[0:2] == 'in' and train_data.shape[0] > sample_index:
                    test_data[sample_index-self.train_size] = array
                sample_index += 1

        return (Abstract_sample(train_data, train_target, train_target[0].size),
                Abstract_sample(test_data, test_target, test_target[0].size))

if __name__ == '__main__':
    Autoencoder(train_size=98, n_epoch=10, n_units=300, same_sample=1, offset_cancel=True)
