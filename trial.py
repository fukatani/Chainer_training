#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      rf
#
# Created:     11/08/2015
# Copyright:   (c) rf 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import sys
import pickle

#constructing newral network
def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y  = model.l3(h2)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


def learning(batchsize, n_epoch, optimizer):
    N = 60000
    x_train, x_test = np.split(mnist.data,   [N])
    y_train, y_test = np.split(mnist.target, [N])
    N_test = y_test.size

    train_loss = []
    train_acc  = []
    test_loss = []
    test_acc  = []

    # Learning loop
    for epoch in xrange(1, n_epoch+1):
        print 'epoch', epoch

        # training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0

        for i in xrange(0, N, batchsize):
            x_batch = x_train[perm[i:i+batchsize]]
            y_batch = y_train[perm[i:i+batchsize]]

            # initialize gradient
            optimizer.zero_grads()
            # forward and calculate error
            loss, acc = forward(x_batch, y_batch)
            # calc grad by back prop
            loss.backward()
            optimizer.update()

            train_loss.append(loss.data)
            train_acc.append(acc.data)
            sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
            sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        # display accuracy for training
        print 'train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N)

        # evaluation
        sum_accuracy = 0
        sum_loss     = 0
        for i in xrange(0, N_test, batchsize):
            x_batch = x_test[i:i+batchsize]
            y_batch = y_test[i:i+batchsize]

            # calc accuracy for test
            loss, acc = forward(x_batch, y_batch, train=False)
            test_loss.append(loss.data)
            test_acc.append(acc.data)
            sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
            sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        # display accuracy for test
        print('test  mean loss={}, accuracy={}'.format(sum_loss / N_test, sum_accuracy / N_test))

    # display accuracy for test as graph
    plt.figure(figsize=(8,6))
    plt.plot(range(len(train_acc)), train_acc)
    plt.plot(range(len(test_acc)), test_acc)
    plt.legend(["train_acc","test_acc"],loc=4)
    plt.title("Accuracy of digit recognition.")
    plt.plot()
    plt.show()

def test(testsize=10):
    #under constructing
    N = 60000
    x_train, x_test = np.split(mnist.data,   [N])
    y_train, y_test = np.split(mnist.target, [N])

    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0

    for i in xrange(0, N, testsize):
        x_batch = x_train[perm[i:i+testsize]]
        y_batch = y_train[perm[i:i+testsize]]

if __name__ == '__main__':
    plt.style.use('ggplot')
    print('fetch MNIST dataset')
    mnist = fetch_mldata('MNIST original')
    print('data fetch success')
    mnist.data   = mnist.data.astype(np.float32)
    mnist.data  /= 255
    mnist.target = mnist.target.astype(np.int32)

    # Prepare multi-layer perceptron model
    try:
        with open("model", "r") as f:
            print('load from pickled data.')
            model = pickle.load(f)
    except (IOError, EOFError):
        input_matrix_size = 784
        output_matrix_size = 10
        n_units   = 100

        model = FunctionSet(l1=F.Linear(input_matrix_size, n_units),
                        l2=F.Linear(n_units, n_units),
                        l3=F.Linear(n_units, output_matrix_size))

    # Setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model.collect_parameters())
    learning(batchsize=100, n_epoch=5, optimizer=optimizer)

    # Save final model
    pickle.dump(model, open('model', 'w'), -1)