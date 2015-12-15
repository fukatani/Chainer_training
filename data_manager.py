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

    def data_slide_split(self):
        if not os.path.exists(self.split_result_dir):
            os.mkdir(self.split_result_dir)
        file_list = []
        for (root, dirs, files) in os.walk(self.directory):
            for file in files:
                if '_split' in file: continue
                file_index = 0
                for offset in range(0, self.data_size, self.offset_width):
                    read_file = open(os.path.join(root, file), 'r')
                    write_line_index = 0
                    write_file = open(self.get_split_file_name(file, file_index), 'w')
                    for read_line_index, line in enumerate(read_file):
                        if read_line_index < offset: continue
                        write_file.write(line)
                        write_line_index += 1
                        if write_line_index == self.data_size:
                            write_file.close()
                            file_index += 1
                            write_file = open(self.get_split_file_name(file, file_index), 'w')
                            write_line_index = 0
                    else:# delete last file
                        write_file.close()
                        os.remove(self.get_split_file_name(file, file_index))
                    read_file.close()

    def plot(self):
        self.clean_split_dir()
        plt.title('Title')
        plt.xlabel('Time(msec)')
        plt.ylabel('Amplitude')

        if self.split_mode == 'non_overlap':
            self.data_split()
            if self.attenate_flag:
                self.attenate()
        elif self.split_mode == 'overlap':
            self.data_slide_split()
            if self.attenate_flag:
                self.attenate()

        for x, y, filename in self.get_xy():
            if self.split_mode and 'split_' not in filename:
                continue
            plt.plot(x, y, label=filename)
        plt.legend()
        if self.save_as_png:
            plt.savefig('./Image/data.png')
        else:
            plt.show()

    def attenate(self):
        for (root, dirs, files) in os.walk(self.split_result_dir):
            for file_name in files:
                if '_split' not in file_name: continue
                for a_index, a_coef in enumerate(self.a_coefs):
                    read_file_name = os.path.join(self.split_result_dir, file_name)
                    read_file = open(read_file_name, 'r')
                    write_file = open(read_file_name.replace('_split_', '_split_a' + str(a_index) + '_'), 'w')
                    for line in read_file:
                        #write_file.write(str(int(int(line) * a_coef)) + '\n')
                        write_file.write(str(int(int(line) + a_coef)) + '\n')
                    write_file.close()
                    read_file.close()

    def clean_split_dir(self):
        for (root, dirs, files) in os.walk(self.split_result_dir):
            for file_name in files:
                os.remove(os.path.join(root, file_name))

    def make_sample(self):
        """ [Functions]
            Make sample for analysis by chainer.
        """
        group_cnts = []
        group_files_dict = {}
        for (root, dirs, files) in os.walk(self.split_result_dir):
            for i, suffix in enumerate(self.group_suffixes):
                group_file = []
                for file_name in files:
                    if file_name[0:2] == suffix:
                        group_file.append(file_name)
                group_files_dict[i] = group_file

        min_item_number = min([len(files) for files in group_files_dict.values()])
        for key, value in group_files_dict.items():
            group_files_dict[key] = value[0:min_item_number]

        sample_size = min_item_number * len(self.group_suffixes)
        data = np.zeros([sample_size, self.data_size], dtype=np.float32)
        target = np.zeros(sample_size)
        sample_index = 0
        for key, value in group_files_dict.items():
            for file_name in value:
                with open(os.path.join(self.split_result_dir, file_name), 'r') as rf:
                    new_data = np.zeros(self.data_size, dtype=np.float32)
                    for line_index, line in enumerate(rf):
                        new_data[line_index] = int(line)
                target[sample_index] = key
                data[sample_index] = new_data
                sample_index += 1

        return Abstract_sample(self.data_size, data, target)

    def __init__(self, directory, data_size=10000, split_mode='', attenate_flag=False, save_as_png=True):
        self.directory = directory
        self.data_size= data_size
        self.offset_width = self.data_size / 4
        self.split_mode = split_mode
        self.attenate_flag = attenate_flag
        self.save_as_png = save_as_png

        self.a_coefs = (-100, 100)
        self.group_suffixes = ('fu', 'in')
        self.split_result_dir = self.directory + '/split_result/'

class Abstract_sample(object):
    def __init__(self, matrix_size, data, target):
        self.matrix_size = matrix_size
        self.data = data
        self.target = target

if __name__ == '__main__':
    dm = data_manager('./numbers', 1000, 'overlap', save_as_png=False)
    dm.plot()
    dm.make_sample()
