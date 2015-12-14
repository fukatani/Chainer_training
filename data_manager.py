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

class data_manager(object):
    def get_xy(self):
        file_list = []
        for (root, dirs, files) in os.walk(self.directory):
            for file in files:
                file_name = os.path.join(root, file)
                max_data_num = self.get_sum_line(file_name)
                readfile = open(file_name, 'r')
                y = np.zeros(max_data_num)
                x = np.linspace(0, max_data_num-1, max_data_num)
                for i, line in enumerate(readfile):
                    y[i] = str(line)
                    if i == max_data_num - 1:
                        break
                readfile.close()
                yield x, y, file

    def get_sum_line(self, filename):
        return sum(1 for line in open(filename, 'r'))

    def get_split_file_name(self, read_file_name, index):
        file_name = os.path.join(self.split_result_dir, read_file_name)
        return file_name.replace('.dat', '_split_' + str(index) + '.dat')

    def data_split(self):
        if not os.path.exists(self.split_result_dir):
            os.mkdir(self.split_result_dir)
        file_list = []
        for (root, dirs, files) in os.walk(self.directory):

            for file in files:
                if '_split' in file:
                    continue
                read_file = open(os.path.join(root, file), 'r')
                file_index = 0
                write_file = open(self.get_split_file_name(file, file_index), 'w')
                i = 0
                for line in read_file:
                    write_file.write(line)
                    i += 1
                    if i == self.data_size:
                        write_file.close()
                        file_index += 1
                        write_file = open(self.get_split_file_name(file, file_index), 'w')
                        i = 0
                else:# delete last file
                    write_file.close()
                    os.remove(self.get_split_file_name(file, file_index))
                read_file.close()

    def plot(self):
        plt.title('Title')
        plt.xlabel('Time(msec)')
        plt.ylabel('Amplitude')

        if self.split_flag:
            self.data_split()
            if self.attenate_flag:
                self.attenate()

        for x, y, filename in self.get_xy():
            if self.split_flag and 'split_' not in filename:
                continue
            plt.plot(x, y, label=filename)
        plt.legend()
        plt.show()

    def attenate(self):
        for (root, dirs, files) in os.walk(self.directory):
            for file_name in files:
                if '_split' not in file_name: continue
                for a_index, a_coef in enumerate(self.a_coefs):
                    write_file = open(file_name.replace('_split_', '_split_a' + str(a_index) + '_'), 'w')
                    read_file = open(file_name, 'r')
                    for line in read_file:
                        write_file.write(int(int(line) * a_coef))
                    write_file.close()
                    read_file.close()

    def __init__(self, directory, data_size=10000, split_flag=False, attenate_flag=False):
        self.directory = directory
        self.data_size= data_size
        self.split_flag = split_flag
        self.attenate_flag = attenate_flag
        self.a_coefs = (0.95, 1.05)
        self.split_result_dir = self.directory + '/split_result/'

if __name__ == '__main__':
    data_manager('C:/Users/rf/Documents/github/Chainer_training/numbers', 2000, True, False).plot()