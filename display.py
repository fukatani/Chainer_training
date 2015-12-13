#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:     Conventional graph display script by using matplotlib
#
# Author:      rf
#
# Created:     15/13/2015
# Copyright:   (c) rf 2015
# Licence:     Apache Licence 2.0
#-------------------------------------------------------------------------------

import os
import matplotlib.pyplot as plt
import numpy as np

def get_xy(directory, max_data_num=10000):
    file_list = []
    for (root, dirs, files) in os.walk(directory):
        for file in files:
            file_name = os.path.join(root, file)
            readfile = open(file_name, 'r')
            y = np.zeros(max_data_num)
            x = np.linspace(0, max_data_num-1, max_data_num)
            for i, line in enumerate(readfile):
                y[i] = str(line)
                if i >= max_data_num - 1:
                    break
            readfile.close()
            yield x, y, file

def plot(directory):

    plt.title('Title')
    plt.xlabel('Time(msec)')
    plt.ylabel('Amplitude')
    for x, y, filename in get_xy(directory):
        plt.plot(x, y, label=filename)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot('C:/Users/rf/Documents/github/Chainer_training/numbers')