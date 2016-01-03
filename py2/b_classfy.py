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
from chainer import Variable
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
        self.add_link(F.Linear(self.n_units[-1], self.last_unit, nobias=self.nobias))

    def loss_function(self, x, y):
        return F.softmax_cross_entropy(x, y)

    def forward(self, x_data, train=True):
        if not isinstance(x_data, Variable):
            x_data = Variable(x_data)
        for model in self:
            x_data = F.dropout(F.relu(model(x_data)), train=train)
        return x_data

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
            plt.plot(self[0].W.data[i]), np.arange(0, len(self[0].W.data[i]))
        plt.title("Weight of l1.")
        plt.plot()
        plt.show()

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
            if self.is_classification:
                recog_answer = np.argmax(self.forward(x, train=False).data)
            else:
                recog_answer = self.forward(x, train=False).data
            loss = self.loss_function(self.forward(x, train=False), Variable(y))
            answer = y[0]

            plt.subplot(3, 3, i+1)
            self.final_test_plot(x, recog_answer)
            plt.title(self.get_final_test_title(answer, recog_answer, loss), size=8)
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
        plt.plot(np.arange(0, x.shape[1], 1), x[0])

    def __init__(self,
                 n_units,
                 is_classification=True,
                 pickle_enable=False,
                 plot_enable=True,
                 save_as_png=True,
                 batch_size=10,
                 epoch=10,
                 **keywords):

        AbstractChain.__init__(self,
                               n_units=n_units,
                               is_classification=is_classification,
                               epoch=epoch,
                               batch_size=batch_size,
                               visualize=True,
                               **keywords)

        #configuration
        self.plot_enable = plot_enable
        self.save_as_png = save_as_png
        if save_as_png and not os.path.exists(util.IMAGE_DIR):
            os.mkdir(util.IMAGE_DIR)

        # Save final self.model
        if pickle_enable:
            pickle.dump(self.model, open('self.model', 'w'), -1)

def set_sample(pre_train_size, pre_test_size, train_size, test_size, auto_encoder=False, **keywords):
    print('fetch data')
    sections = [pre_train_size, pre_test_size, train_size, test_size]
    sample = data_manager.data_manager(util.DATA_DIR,
                                       data_size=300,
                                       split_mode='pp',
                                       attenate_flag=True,
                                       auto_encoder=auto_encoder,
                                       offset_cancel=True,
                                       keywords=keywords
                                       ).make_sample(sections)
    p_x_train, p_x_test, x_train, x_test, _ = sample.data
    _, _, y_train, y_test, _ = sample.target
    return p_x_train, p_x_test, x_train, x_test, y_train, \
           y_test, sample.input_matrix_size, sample.output_matrix_size

if __name__ == '__main__':
    p_x_train, p_x_test, x_train, x_test, y_train, y_test, im, om = \
                                                    set_sample(1, 1, 100, 40)
    bc = b_classfy([im, 150, 100, om], isClassification=True)
    bc.pre_training(p_x_train, p_x_test)
    bc.learn(x_train, y_train, x_test, y_test)
    #bc.disp_w()
    bc.final_test(x_test[0:9], y_test[0:9])

