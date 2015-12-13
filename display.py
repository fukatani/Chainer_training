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

def get_xy(directory, max_data_num_set=100000):
    file_list = []
    for (root, dirs, files) in os.walk(directory):
        for file in files:
            file_name = os.path.join(root, file)
            max_data_num =get_sum_line(file_name)
            readfile = open(file_name, 'r')
            y = np.zeros(max_data_num)
            x = np.linspace(0, max_data_num-1, max_data_num)
            for i, line in enumerate(readfile):
                y[i] = str(line)
                if i >= max_data_num - 1 or i >= max_data_num_set - 1:
                    break
            readfile.close()
            yield x, y, file

def get_sum_line(filename):
    return sum(1 for line in open(filename, 'r'))

def data_split(directory, data_size=2000):
    #TODO
    split_result_dir = directory + '/split_result/'
    def get_write_file_name(root, read_file_name, index):
        file_name = os.path.join(split_result_dir, read_file_name)
        return file_name.replace('.dat', '_split_' + str(index) + '.dat')

    if not os.path.exists(split_result_dir):
        os.mkdir(split_result_dir)
    file_list = []
    for (root, dirs, files) in os.walk(directory):

        for file in files:
            if '_split' in file:
                continue
            read_file = open(os.path.join(root, file), 'r')
            file_index = 0
            write_file = open(get_write_file_name(root, file, file_index), 'w')
            i = 0
            for line in read_file:
                write_file.write(line)
                i += 1
                if i == data_size:
                    write_file.close()
                    file_index += 1
                    write_file = open(get_write_file_name(root, file, file_index), 'w')
                    i = 0
            else:# delete last file
                write_file.close()
                os.remove(get_write_file_name(root, file, file_index))
            read_file.close()

def plot(directory, split_flag=False):
    plt.title('Title')
    plt.xlabel('Time(msec)')
    plt.ylabel('Amplitude')
    if split_flag:
        data_split(directory)
    for x, y, filename in get_xy(directory):
        if split_flag and 'split_' not in filename:
            continue
        plt.plot(x, y, label=filename)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot('C:/Users/rf/Documents/github/Chainer_training/numbers', True)