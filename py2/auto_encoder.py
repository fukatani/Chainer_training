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
from b_classfy import b_classfy, set_sample
from data_manager import data_manager, Abstract_sample
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import numpy as np
import util

class Autoencoder(b_classfy):
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
        self.train_sample, self.test_sample = dm_for_ae(util.DATA_DIR
            , 1000, self.train_size, attenate_flag=True, slide=4, keywords=self.keywords).make_sample()
        self.input_matrix_size = self.train_sample.input_matrix_size
        self.output_matrix_size = self.train_sample.output_matrix_size

    def get_final_test_title(self, answer, recog, loss):
        return str(loss.data)

    def final_test_plot(self, x, y):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(0, self.input_matrix_size, 1), x[0])
        plt.plot(np.arange(0, self.input_matrix_size, 1), y)

if __name__ == '__main__':
    p_x_train, p_x_test, x_train, x_test, y_train, y_test, im, om = \
                                                    set_sample(1, 1, 120, 40)
    bc = Autoencoder([im, 150, 100, om])
    bc.pre_training(p_x_train, p_x_test)
    bc.learn(x_train, x_train, x_test, x_test, isClassification=False)
    #bc.disp_w()
    bc.final_test(x_test[0:9], y_test[0:9])
