#-------------------------------------------------------------------------------
# Name:        Mychain
# Purpose:
#
# Author:      rf
#
# Created:     01/02/2016
# Copyright:   (c) rf 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import chainer.functions as F
import sys
import data_manager
import pickle
import os
import util
import six
from AbstractChain import AbstractChain

#constructing newral network
class b_classfy(AbstractChain):

    def add_last_layer(self):
        self.add_link(F.Linear(self.n_units[-1], self.last_unit))

    def loss_function(self, x, y):
        return F.softmax_cross_entropy(x, y)

    def disp_learn_result(self, train_acc, test_acc):
        """
        display accuracy for test as graph
        """
        plt.style.use('ggplot')
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(train_acc)), train_acc)
        plt.plot(range(len(test_acc)), test_acc)
        plt.legend(["train_acc","test_acc"], loc=4)
        plt.title("Accuracy of inoue/fukatani recognition.")
        plt.plot()

        if self.save_as_png:
            plt.savefig(os.path.join(util.IMAGE_DIR, 'learn_result.png'))
        else:
            plt.show()

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
        perm = np.random.permutation(self.test_sample.sample_size)
        x_batch = self.test_sample.data[perm[0:test_data_size]]
        y_batch = self.test_sample.target[perm[0:test_data_size]]
        return x_batch, y_batch

    def final_test(self, x_batch, y_batch=None):
        plt.close('all')
        plt.style.use('fivethirtyeight')
        #size = 28
        if y_batch is None:
            y_batch = np.zeros(len(x_batch))
        for i in range(x_batch.shape[0]):
            #single test (mini batch size == 1)
            x = x_batch[i:i+1]
            y = y_batch[i:i+1]
            recog_answer, loss = self.forward(x, y, train=False, answer=True)
            answer = y[0]

            plt.subplot(3, 3, i+1)
            self.final_test_plot(x, recog_answer[0])
            plt.title(self.get_final_test_title(answer, recog_answer[0], loss), size=8)
            plt.tick_params(labelbottom="off")
            plt.tick_params(labelleft="off")
            if 'dump_final_result' in self.keywords.keys():
                if not os.path.exists(util.DUMP_DIR):
                    os.mkdir(util.DUMP_DIR)
                np.savetxt(''.join((util.DUMP_DIR + 'final_result', str(i), '.dump')), y)
        if self.save_as_png:
            plt.savefig(os.path.join(util.IMAGE_DIR, 'final_test.png'))
        plt.show()

    def get_final_test_title(self, answer, recog_answer, loss=None):
        return "ans=%d, recog=%d"%(answer, recog_answer)

    def final_test_plot(self, x, y):
        plt.plot(np.arange(0, self.input_matrix_size, 1), x[0])

    def __init__(self,
                 pickle_enable=False,
                 plot_enable=True,
                 save_as_png=True,
                 is_clastering=True,
                 batch_size=20,
                 n_epoch=10,
                 n_units=[1000,200,2],
                 **keywords):

        AbstractChain.__init__(n_units=n_units,
                               epoch=epoch,
                               batch_size=batch_size,
                               visualize=True
                               )

        #configuration
        self.plot_enable = plot_enable
        self.save_as_png = save_as_png
        self.is_clastering = is_clastering
        self.n_units = n_units
        if save_as_png and not os.path.exists(util.IMAGE_DIR):
            os.mkdir(util.IMAGE_DIR)
        if keywords:
            self.keywords = keywords
        else:
            self.keywords = {}

##        # setup chainer
##        nobias = 'nobias' in keywords.keys()
##        self.set_sample()
##
##        #self.disp_w()
##        if final_test_enable:
##            x_batch, y_batch = self.extract_test_sample()
##            self.final_test(x_batch, y_batch)
##
##        # Save final self.model
##        if pickle_enable:
##            pickle.dump(self.model, open('self.model', 'w'), -1)

def set_sample(pre_train_size, pre_test_size, train_size, test_size):
    print('fetch data')
    train_sample, test_sample = data_manager.data_manager(
        util.DATA_DIR, 1000, self.train_size, attenate_flag=True).make_sample()

if __name__ == '__main__':
    bc = b_classfy()
    p_x_train, p_x_test, x_test, x_train, y_train, y_test = set_sample()
    bc.pre_training()
    bc.learn(x_train, y_train, x_test, y_test, isClassification=True)

