#-------------------------------------------------------------------------------
# Name:        data_manager
# Purpose:     Data manager for chainer
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
import util
from collections import OrderedDict

class data_manager(object):
    def get_xy(self, data_dict):
        for name, data in data_dict.items():
            x = np.arange(0, len(data_dict[name]), 1)
            yield x, data, name

    def split_by_slide(self):
        """
        ex. [1,2,3,4,5,6,7,8,9] -> [1,2,3,4,5], [2,3,4,5,6], ..., [5,6,7,8,9]
        """
        self.splited_data_dict = OrderedDict()
        for name, data_array in self.raw_data_dict.items():
            i = 0
            while i * self.offset_width + self.data_size < self.array_size:
                self.splited_data_dict[name + '_s' + str(i)] = data_array[i * self.offset_width: i * self.offset_width + self.data_size]
                i += 1

    def split_by_peak(self):
        from scipy import signal
        self.splited_data_dict = OrderedDict()
        for name, data_array in self.raw_data_dict.items():
            m_buttored = util.filter_signal(data_array, 0.01)
            max_indexes = signal.argrelmax(m_buttored)[0]
            i = 0
            for max_index in max_indexes:
                if len(data_array) - max_index < self.data_size:
                    break
                self.splited_data_dict[name + '_s' + str(i)] = data_array[max_index: max_index + self.data_size]
                i += 1

    def attenate(self, data_dict):
        """
        ex. [1,2,3,4,5] -> [2,3,4,5,6]
        """
        attenated_data_dict = OrderedDict()
        for name, data_array in data_dict.items():
            for i, coef in enumerate(self.a_coefs):
                attenated_data_dict[name + '_a' + str(i)] = data_array + coef
        return attenated_data_dict

    def read_all_data(self):
        """
        Get numpy array (raw_data) from *.dat file.
        """
        self.raw_data_dict = OrderedDict()
        self.array_size = min([util.get_sum_line(os.path.join(self.directory, name)) for name in os.listdir(self.directory)])
        for each_file in os.listdir(self.directory):
            if each_file[-4:] != '.dat': continue
            new_array = np.zeros(self.array_size)
            read_file = open(os.path.join(self.directory, each_file), 'r')
            i = 0
            for line in read_file:
                new_array[i] = int(line)
                i += 1
                if i == self.array_size:
                    break
            read_file.close()
            self.raw_data_dict[each_file] = new_array

    def get_data(self):
        """
        Get final data from raw_data.
        """
        if self.split_mode:
            if self.split_mode == 'slide':
                self.split_by_slide()
            else:
                self.split_by_peak()
            if self.attenate_flag:
                return self.attenate(self.splited_data_dict)
            else:
                return self.splited_data_dict
        if self.attenate_flag:
            return self.attenate(self.raw_data_dict)
        else:
            return self.raw_data_dict

    def plot(self):
        plt.title('Title')
        plt.xlabel('Time(msec)')
        plt.ylabel('Amplitude')
        data_dict = self.get_data()

        for x, y, name in self.get_xy(data_dict):
            plt.plot(x, y, label=name)
        plt.legend()
        if self.save_as_png:
            plt.savefig(os.path.join(util.IMAGE_DIR,'data.png'))
        else:
            plt.show()

    def get_target(self, name):
        return 0 if name[0:2] == 'fu' else 1

    def get_spectrogram(self, data_dict):
        #TODO
        from scipy import signal
        spectrogram_dict = OrderedDict()

        for name, data in data_dict.items():
            f, t, Sxx = signal.spectrogram(data, fs=1)
            plt.pcolormesh(t, f, Sxx)
        return spectrogram_dict

    def process_sample_backend(func):
        """
        Data processing after make sample.
        """
        import datetime
        from functools import wraps
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            sample = func(self, *args, **kwargs)
            min_val = min(np.min(data) for data in sample.data)
            max_val = max(np.max(data) for data in sample.data)
            if self.offset_cancel:
                for data, target in zip(sample.data, sample.target):
                    data /= max_val
                    data = np.array([element - np.average(element) for element in data])
            else:
                for data, target in zip(sample.data, sample.target):
                    data -= min_val
                    data /= max_val
            if self.auto_encoder:
                sample.target = sample.data
            return sample
        return wrapper

    @process_sample_backend
    def make_sample(self, section=[100, 100]):
        """ [Functions]
            Make sample for analysis by chainer.
        """
        data_dict = self.get_data()
        sample_size = len(data_dict.keys())
        #initialize
        sample_data = np.zeros([sample_size, self.data_size], dtype=np.float32)
        sample_target = np.zeros(sample_size, dtype=np.int32)
        sample_index = 0

        for name, array in data_dict.items():
            sample_data[sample_index] = array
            sample_target[sample_index] = self.get_target(name)
            sample_index += 1

        if self.randomization:
            orders = np.random.permutation(sample_size)
        elif self.order:
            orders = np.arange(0, sample_size)
        elif self.same_sample:
            orders = np.array([i % self.sample_kinds for i in range(sample_size)])

        indexes = []
        for index in section:
            indexes.append(index + sum(indexes))
        datas = np.split(sample_data[orders], indexes)
        targets = np.split(sample_target[orders], indexes)

        return Abstract_sample(datas, targets, len(self.group_suffixes))

    def __init__(self,
                 directory,
                 data_size=1000,
                 split_mode='slide',
                 attenate_flag=False,
                 save_as_png=True,
                 slide=4,
                 **keywords):
        self.directory = directory
        self.data_size = data_size
        self.split_mode = split_mode
        self.offset_width = self.data_size / slide
        self.attenate_flag = attenate_flag
        self.save_as_png = save_as_png
        self.keywords = keywords

        self.a_coefs = (-100, 100)
        self.group_suffixes = ('fu', 'in')
        if save_as_png and not os.path.exists(util.IMAGE_DIR):
            os.mkdir(util.IMAGE_DIR)
        self.read_all_data()
        self.analyse_keywords()

    def analyse_keywords(self):
        if self.keywords:
            self.order = 'order_sample' in self.keywords.keys()
            self.same_sample = 'same_sample' in self.keywords.keys()
            if self.same_sample:
                self.sample_kinds = self.keywords['same_sample']
            self.denoised_enable = 'denoised_enable' in self.keywords.keys()
            if self.denoised_enable:
                self.noise_coef = self.keywords['denoised_enable']
            if 'offset_cancel' in self.keywords.keys():
                self.offset_cancel = self.keywords['offset_cancel']
            else:
                self.offset_cancel = False
            if 'split_mode' in self.keywords.keys():
                self.split_mode = self.keywords['split_mode']
            if 'auto_encoder' in self.keywords.keys():
                self.auto_encoder = self.keywords['auto_encoder']
            else:
                self.auto_encoder = False
        else:
            self.order = False
            self.same_sample = False
            self.denoised_enable = False
            self.offset_cancel = False
            self.auto_encoder = False
        self.randomization = not (self.order or self.same_sample)

class Abstract_sample(object):
    def __init__(self, data, target, output_matrix_size):
        self.data = data
        self.input_matrix_size = data[0].shape[-1]
        self.target = target
        self.output_matrix_size = output_matrix_size
        #self.sample_size = target.size / target[0].size

if __name__ == '__main__':
    dm = data_manager(util.DATA_DIR, 300, split_mode='pp', attenate_flag=True, save_as_png=False)
    #dm.plot()
    dm.make_sample()
