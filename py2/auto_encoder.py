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
from b_classfy import b_classfy
from chainer import cuda, Variable, optimizers
import chainer.functions as F
import numpy as np
import util

class Autoencoder(b_classfy):
    def add_last_layer(self):
        self.add_link(F.Linear(self.n_units[-1], self.last_unit, nobias=self.nobias))

    def loss_function(self, x, y):
        return F.mean_squared_error(x, y)

    def get_final_test_title(self, answer, recog, loss):
        return str(loss.data)

    def final_test_plot(self, x, y):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(0, x.shape[1], 1), x[0])
        plt.plot(np.arange(0, x.shape[1], 1), y[0])

    def forward(self, x_data, train=True):
        if not isinstance(x_data, Variable):
            x_data = Variable(x_data)
        for i, model in enumerate(self):
            if i == len(self) - 1 and self.pre_trained:
                x_data = model(x_data)
            else:
                x_data = F.dropout(F.relu(model(x_data)), train=train)
        return x_data

if __name__ == '__main__':
    p_x_train, p_x_test, x_train, x_test, y_train, y_test, im, om = \
                                                    util.set_sample(1, 1, 120, 40, split_mode='pp', offset_cancel=True, same_sample=10)
    bc = Autoencoder([im, 200, 200, im], epoch=100, is_classification=False, nobias=False)
    bc.pre_training(p_x_train, p_x_test)
    bc.learn(x_train, x_train, x_test, x_test)
    #bc.disp_w()
    bc.final_test(x_test[0:9], x_test[0:9])
