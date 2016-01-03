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
    def add_last_layer(self):
        self.add_link(F.Linear(self.n_units[-1], self.last_unit, nobias=self.nobias))

    def loss_function(self, x, y):
        return F.softmax_cross_entropy(x, y)

    def get_final_test_title(self, answer, recog, loss):
        return str(loss.data)

    def final_test_plot(self, x, y):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(0, self.input_matrix_size, 1), x[0])
        plt.plot(np.arange(0, self.input_matrix_size, 1), y)

if __name__ == '__main__':
    p_x_train, p_x_test, x_train, x_test, y_train, y_test, im, om = \
                                                    set_sample(1, 1, 120, 40)
    bc = Autoencoder([im, 150, 100, om], isClassification=False)
    bc.pre_training(p_x_train, p_x_test)
    bc.learn(x_train, x_train, x_test, x_test)
    #bc.disp_w()
    bc.final_test(x_test[0:9], y_test[0:9])
