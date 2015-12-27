#-------------------------------------------------------------------------------
# Name:        util
# Purpose:
#
# Author:      rf
#
# Created:     14/11/2015
# Copyright:   (c) rf 2015
# Licence:     Apache Licence 2.0
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plot

def get_sum_line(filename):
    """
    Count all line in file.
    """
    return sum(1 for line in open(filename, 'r'))

def simple_plot(data):
    """
    Display time series plot.
    """
    plt.plot(np.arange(0, data.size, 1), data)
    plt.show()

def simple_plot_multi(all_data):
    """
    Display time series plot (multi series).
    """
    for data in all_data:
        plt.plot(np.arange(0, data.size, 1), data)
    plt.show()

def calc_moving_average(data, width):
    ref = np.ones(width) / width
    return np.convolve(data, ref, 'valid')

if __name__ == '__main__':
    #simple_plot(np.array([1,2,3]))
    a = np.array([1,2,3,4,3,6,9,8,10,7,6,5,3,1])
    b = calc_moving_average(a, 5)
    simple_plot_multi([a, b])