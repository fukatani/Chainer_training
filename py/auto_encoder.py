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

    def get_final_test_title(self, answer, recog, data):
        return str(data)

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
    p_x_train0, p_x_test0, x_train0, x_test0, y_train0, y_test0, im, om = \
        util.set_sample(60, 1, 40, 20,
                        split_mode='pp',
                        #offset_cancel=True,
                        first_cancel=True,
                        #normal_constant=1709.0,
                        same_sample=70,
                        spec_target=0,
                        )
    p_x_train1, p_x_test1, x_train1, x_test1, y_train1, y_test1, im, om = \
        util.set_sample(60, 1, 40, 20,
                        split_mode='pp',
                        first_cancel=True,
                        div_reference=True,
                        #normal_constant=1709.0,
                        )

    bc = Autoencoder([im, 150, 120, im], epoch=100, is_classification=False, nobias=True)
    bc.pre_training(p_x_train0, p_x_test0)
    bc.learn(x_train0, x_train0, x_test0, x_test0)
    #bc.disp_w()
    bc.final_test(x_test1[0:9], x_test1[0:9])
    #bc.final_test(x_test0[0:9], x_test0[0:9])
    bc.serialize_to_hdf5("ae_model.gz")
