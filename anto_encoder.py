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
            return [np.argmax(data) for data in y.data]
        else:
            return F.mean_squared_error(y, t), F.accuracy(y, t)

    def set_sample(self):
        print('fetch data')
        self.sample = dm_for_ae('./numbers', 1000, 'overlap', True).make_sample()
        self.input_matrix_size = self.sample.input_matrix_size
        self.output_matrix_size = self.sample.output_matrix_size

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

if __name__ == '__main__':
    Autoencoder()
