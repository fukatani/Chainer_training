#-------------------------------------------------------------------------------
# test_ra.py
#
# the test for all pyverilog_toolbox function
#
# Copyright (C) 2015, Ryosuke Fukatani
# License: Apache 2.0
#-------------------------------------------------------------------------------


import sys
import b_classfy
import util
import auto_encoder
import unittest

class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def test_n_for_b(self):
        p_x_train, p_x_test, x_train, x_test, y_train, y_test, im, om = \
                                                        util.set_sample(1, 1, 100, 40)
        bc = b_classfy.b_classfy([im, 150, 150, om], isClassification=True)
        bc.pre_training(p_x_train, p_x_test)

        self.assertGreater(bc.learn(x_train, y_train, x_test, y_test), 0.8)

    def test_auto_encoder(self):
        p_x_train, p_x_test, x_train, x_test, y_train, y_test, im, om = \
                                                        util.set_sample(60, 1, 40, 20, split_mode='pp', offset_cancel=True, same_sample=10)
        bc = Autoencoder.Autoencoder([im, 200, 150, im], epoch=40, is_classification=False, nobias=False)
        bc.pre_training(p_x_train, p_x_test)
        bc.learn(x_train, x_train, x_test, x_test)
        #bc.disp_w()
        self.assertLess(bc.final_test(x_test[0:9], x_test[0:9]), 0.1)

if __name__ == '__main__':
    unittest.main()
