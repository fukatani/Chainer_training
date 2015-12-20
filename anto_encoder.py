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
        h2 = F.dropout(F.relu(self.model.l2(h1)), train=train)
        y  = self.model.l3(h2)
        if answer:
            return y.data
        else:
            return F.mean_squared_error(y, t)#, F.accuracy(y, t)

    def set_sample(self):
        print('fetch data')
        self.sample = dm_for_ae('./numbers', 1000, 'overlap', True, slide=4).make_sample()
        self.input_matrix_size = self.sample.input_matrix_size
        self.output_matrix_size = self.sample.output_matrix_size

    def get_final_test_title(self, answer, recog):
        return ""

    def final_test_plot(self, x, y):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(0, self.input_matrix_size, 1), x[0])
        plt.plot(np.arange(0, self.input_matrix_size, 1), y)

    def __init__(self, pickle_enable=False,
                 plot_enable=True,
                 save_as_png=True,
                 final_test_enable=True,
                 is_clastering=False,
                 train_data_size=100,
                 batch_size=10,
                 n_epoch=10,
                 n_units=200):
        Mychain.__init__(self,
                         pickle_enable,
                         plot_enable,
                         save_as_png,
                         final_test_enable,
                         is_clastering,
                         train_data_size,
                         batch_size,
                         n_epoch,
                         n_units)

class dm_for_ae(data_manager):
    def process_sample_backend_ae(func):
        import datetime
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            sample = func(*args, **kwargs)
            sample.data  -= np.min(sample.data)
            sample.data  /= np.max(sample.data)
            sample.data   = sample.data.astype(np.float32)
            sample.target = np.array(sample.data)
            return sample
        return wrapper

    @process_sample_backend_ae
    def make_sample(self):
        """ [Functions]
            Make sample for analysis by chainer.
        """
        data_dict = self.get_data()
        sample_size = len(data_dict.keys())

        #initialize
        data = np.zeros([sample_size, self.data_size], dtype=np.float32)
        target = np.zeros([sample_size, self.data_size], dtype=np.float32)

        sample_index = 0
        for name, array in data_dict.items():
            data[sample_index] = array
            sample_index += 1
        return Abstract_sample(data, target, target[0].size)

    def process_sample_backend_ae2(func):
        import datetime
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            train, test = func(*args, **kwargs)
            train.data  -= np.min(train.data)
            train.data  /= np.max(train.data)
            train.data   = train.data.astype(np.float32)
            train.target = np.array(train.data)
            test.data  -= np.min(test.data)
            test.data  /= np.max(test.data)
            test.data   = test.data.astype(np.float32)
            test.target = np.array(test.data)
            return train, test
        return wrapper

    @process_sample_backend_ae2
    def make_sample2(self):
        """ [Functions]
            Make sample for analysis by chainer.
        """
        data_dict = self.get_data()
        sample_size = len(data_dict.keys())
        train_size = 50

        #initialize
        train_data = np.zeros([sample_size, train_size], dtype=np.float32)
        test_data = np.zeros([sample_size, self.data_size - train_size], dtype=np.float32)
        train_target = np.zeros([sample_size, train_size], dtype=np.float32)
        test_target = np.zeros([sample_size, self.data_size - train_size], dtype=np.float32)

        sample_index = 0
        for name, array in data_dict.items():
            if sample_index < train_size:
                train_data[sample_index] = array
            else:
                test_data[sample_index-train_size] = array
            sample_index += 1
        return (Abstract_sample(train_data, train_target, train_target[0].size),
                Abstract_sample(test_data, test_target, test_target[0].size))

if __name__ == '__main__':
    Autoencoder(train_data_size=100, n_epoch=20, n_units=200)
