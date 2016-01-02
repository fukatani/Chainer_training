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
import matplotlib.pyplot as plt

DUMP_DIR = '../dump_files'
IMAGE_DIR = '../Image'
DATA_DIR = '../numbers'

def get_sum_line(filename):
    """
    Count all line in file.
    """
    return sum(1 for line in open(filename, 'r'))

def simple_plot(data, start=0, end=None, legend=None, style=''):
    """
    Display time series plot.
    Mainly used for debug.
    """
    if isinstance(data, np.ndarray):
        simple_plot_single(data, start, end, style)
    else:
        for each_data in data:
            simple_plot_single(each_data, start, end, style)
    if legend is not None:
        plt.legend(legend)
    plt.show()

def simple_plot_single(data, start, end, style):
    if end is None:
        end = data.size
    data = data[start:end]
    if style == 'scatter':
        plt.scatter(np.arange(0, data.size, 1), data)
    else:
        plt.plot(np.arange(0, data.size, 1), data)

def calc_moving_average(data, width):
    ref = np.ones(width) / width
    return np.convolve(data, ref, 'valid')

def filter_signal(data, cutoff=0.1):
    from scipy import signal
    # filter configuration
    N = 2 # filter order

    b1, a1 = signal.butter(N, cutoff, "low")
    return signal.filtfilt(b1, a1, data)

def time_record(func):
    import datetime
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.datetime.today()
        result = func(*args, **kwargs)
        end = datetime.datetime.today()
        print('time elapsed ' + str(end-start))
    return wrapper

def disp_all_dump_data():
    data_dict = read_all_data(DUMP_DIR, '.dump')
    simple_plot(data=data_dict.values(),
                legend=data_dict.keys())

def read_all_data(directory, extension='.dat'):
    """
    Get numpy array (raw_data) from *.dat file.
    """
    import os
    from collections import OrderedDict
    raw_data_dict = OrderedDict()
    array_size = min([get_sum_line(os.path.join(directory, name)) for name in os.listdir(directory)])
    for file_name in os.listdir(directory):
        if file_name[-len(extension):] != extension: continue
        read_file = open(os.path.join(directory, file_name), 'r')
        i = 0
        line = read_file.readline()
        new_array = np.array([float(num_str) for num_str in (line.split(' '))])
        read_file.close()
        raw_data_dict[file_name] = new_array
    return raw_data_dict

if __name__ == '__main__':
    #simple_plot(np.array([1,2,3]))
    #a = np.array([1,2,3,4,3,6,9,8,10,7,6,5,3,1])
    #b = calc_moving_average(a, 5)
    #simple_plot([a, b])
    disp_all_dump_data()