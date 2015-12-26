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
            x = np.arange(0, len(data_dict[name]), 1)
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
        if self.split_mode:
            self.data_slide_split()
            return self.attenate(self.splited_data_dict)
        return self.attenate(self.raw_data_dict)

    def get_target(self, name):
        return 0 if name[0:2] == 'fu' else 1

    def process_sample_backend(func):
        import datetime
        from functools import wraps
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            train, test = func(self, *args, **kwargs)
            train.data  -= np.min(train.data)
            train.data  /= np.max(train.data)
            train.data   = train.data.astype(np.float32)
            train.target = train.target.astype(np.int32)
            test.data  -= np.min(test.data)
            test.data  /= np.max(test.data)
            test.data   = test.data.astype(np.float32)
            test.target = test.target.astype(np.int32)
            return train, test
        return wrapper

    @process_sample_backend
    def make_sample(self):
        """ [Functions]
            Make sample for analysis by chainer.
        """
        data_dict = self.get_data()
        sample_size = len(data_dict.keys())

        #initialize
        sample_data = np.zeros([sample_size, self.data_size], dtype=np.float32)
        sample_target = np.zeros(sample_size)

        sample_index = 0
        for name, array in data_dict.items():
            sample_data[sample_index] = array
            sample_target[sample_index] = self.get_target(name)
            sample_index += 1

        perm = np.random.permutation(sample_size)
        train_data = sample_data[perm[0:self.train_size]]
        train_target = sample_target[perm[0:self.train_size]]
        test_data = sample_data[perm[self.train_size:]]
        test_target = sample_target[perm[self.train_size:]]

        return (Abstract_sample(train_data, train_target, len(self.group_suffixes)),
                Abstract_sample(test_data, test_target, len(self.group_suffixes)))

    def __init__(self,
                 directory,
                 data_size=10000,
                 train_size=100,
                 split_mode=True,
                 attenate_flag=False,
                 save_as_png=True,
                 slide=4,
                 keywords=None):
        self.directory = directory
        self.data_size = data_size
        self.train_size = train_size
        self.split_mode = split_mode
        self.offset_width = self.data_size / slide
        self.attenate_flag = attenate_flag
        self.save_as_png = save_as_png
        self.keywords = keywords

        self.a_coefs = (-100, 100)
        self.group_suffixes = ('fu', 'in')
        if save_as_png and not os.path.exists('./Image'):
            os.mkdir('./Image')
        self.read_all_data()
        self.analyse_keywords()

    def analyse_keywords(self):
        if self.keywords:
            self.randomization = 'random_sample' in self.keywords.keys()
            self.order = 'order_sample' in self.keywords.keys()
            self.all_same = 'same_sample' in self.keywords.keys()
            if self.all_same:
                self.sample_kinds = self.keywords['same_sample']
            self.denoised_enable = 'denoised_enable' in self.keywords.keys()
            if self.denoised_enable:
                self.noise_coef = self.keywords['denoised_enable']
            self.offset_cancel = 'offset_cancel' in self.keywords.keys()
        else:
            self.randomization = False
            self.order = False
            self.all_same = False
            self.denoised_enable = False

class Abstract_sample(object):
    def __init__(self, data, target, output_matrix_size):
        self.data = data
        self.input_matrix_size = data[0].size
        self.target = target
        self.output_matrix_size = output_matrix_size
        self.sample_size = target.size / target[0].size

if __name__ == '__main__':
    dm = data_manager('./numbers', 1000, 100, split_mode=False, attenate_flag=True, save_as_png=False)
    dm.plot()
    #dm.make_sample()
