#-------------------------------------------------------------------------------
# test_ra.py
#
# the test for all pyverilog_toolbox function
#
# Copyright (C) 2015, Ryosuke Fukatani
# License: Apache 2.0
#-------------------------------------------------------------------------------


import sys
import n_for_b
import auto_encoder
import unittest

class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def test_n_for_b(self):
        mc = n_for_b.Mychain(pickle_enable=False,
                        save_as_png=False,
                        plot_enable=False,
                        final_test_enable=False,
                        n_epoch=6
                        )
        self.assertGreater(mc.last_accuracy, 0.8)

    def test_auto_encoder(self):
        ae = auto_encoder.Autoencoder(train_size=98,
                                      n_epoch=6,
                                      n_units=300,
                                      same_sample=10,
                                      offset_cancel=True,
                                      is_clastering=False,
                                      input_data_size=300,
                                      split_mode='pp',
                                      save_as_png=False,
                                      plot_enable=False,
                                      final_test_enable=False
                                      )
        self.assertLess(ae.last_loss, 0.01)

if __name__ == '__main__':
    unittest.main()
