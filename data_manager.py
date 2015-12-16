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

#future replaced data_manager

class data_manager(object):
    def get_xy(self, data_dict):
        for name, data in data_dict.items():
            x = np.arange(0, self.data_size, 1)
            yield x, data, name

    def get_sum_line(self, filename):
        return sum(1 for line in open(filename, 'r'))

    def data_slide_split(self):
        self.splited_data_dict = {}
        for name, data_array in self.raw_data_dict.items():
            i = 0
            while i * self.offset_width + self.data_size < self.array_size:
                self.splited_data_dict[name + '_s' + str(i)] = data_array[i * self.offset_width: i * self.offset_width + self.data_size]
                i += 1

    def attenate(self, data_dict):
        attenated_data_dict = {}
        for name, data_array in data_dict.items():
            for i, coef in enumerate(self.a_coefs):
                attenated_data_dict[name + '_a' + str(i)] = data_array + coef
        return attenated_data_dict

    def read_all_data(self):
        self.raw_data_dict = {}
        for (root, dirs, files) in os.walk(self.directory):
            self.array_size = min([self.get_sum_line(os.path.join(root, name)) for name in files])
            for file_name in files:
                new_array = np.zeros(self.array_size)
                read_file = open(os.path.join(root, file_name), 'r')
                i = 0
                for line in read_file:
                    new_array[i] = int(line)
                    i += 1
                    if i == self.array_size:
                        break
                read_file.close()
                self.raw_data_dict[file_name] = new_array

    def plot(self):
        plt.title('Title')
        plt.xlabel('Time(msec)')
        plt.ylabel('Amplitude')
        data_dict = self.get_data()

        for x, y, name in self.get_xy(data_dict):
            plt.plot(x, y, label=name)
        plt.legend()
        if self.save_as_png:
            plt.savefig('./Image/data.png')
        else:
            plt.show()

    def get_data(self):
        self.data_slide_split()
        return self.attenate(self.splited_data_dict)

    def get_target(self, name):
        return 0 if name[0:2] == 'fu' else 1

    def process_sample_backend(func):
        import datetime
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            sample = func(*args, **kwargs)
            sample.data  -= np.min(sample.data)
            sample.data  /= np.max(sample.data)
            sample.data   = sample.data.astype(np.float32)
            sample.target = sample.target.astype(np.int32)
            return sample
        return wrapper

    @process_sample_backend
    def make_sample(self):#TODO
        """ [Functions]
            Make sample for analysis by chainer.
        """
        data_dict = self.get_data()
        sample_size = len(data_dict.keys())

        #initialize
        data = np.zeros([sample_size, self.data_size], dtype=np.float32)
        target = np.zeros(sample_size)

        sample_index = 0
        for name, array in data_dict.items():
            target[sample_index] = self.get_target(name)
            data[sample_index] = array
            sample_index += 1
        return Abstract_sample(self.data_size, data, target, len(self.group_suffixes))

    def __init__(self, directory, data_size=10000, split_mode='', attenate_flag=False, save_as_png=True):
        self.directory = directory
        self.data_size= data_size
        self.offset_width = self.data_size / 4
        self.attenate_flag = attenate_flag
        self.save_as_png = save_as_png

        self.a_coefs = (-100, 100)
        self.group_suffixes = ('fu', 'in')
        if save_as_png and not os.path.exists('./Image'):
            os.mkdir('./Image')
        self.read_all_data()

class Abstract_sample(object):
    def __init__(self, matrix_size, data, target, output_matrix_size):
        self.matrix_size = matrix_size
        self.data = data
        self.target = target
        self.output_matrix_size = output_matrix_size

if __name__ == '__main__':
    dm = data_manager('./numbers', 1000, 'overlap', True, save_as_png=False)
    #dm.plot()
    dm.make_sample()
