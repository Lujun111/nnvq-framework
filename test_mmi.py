#!/home/ga96yar/tensorflow_py3/bin/python

import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten, vq
from create_tfrecords import read_single_data, get_transformation_vec, trans_vec_to_phones
from MiscHelper import Misc



def activations(features, weights_nn):
    """

    :param features: input feature
    :param weights_nn: weights of NN
    :return: activations of NN
    """

    # subtract the features of each row (via broadcast)
    weights_nn = np.repeat(weights_nn[np.newaxis, :], features.shape[0], axis=0)

    weights_nn -= features[:, np.newaxis, :]
    weights_nn = np.power(weights_nn, 2)

    # activations
    active = np.sum(weights_nn, axis=2)
    return active


def sort_label_stream(stream, cb_size):
    """

    :param stream: input label stream
    :param cb_size: size of codebook
    :return: return occupancy of label stream
    """

    # define counter array for labels
    count_array = np.zeros(cb_size)

    # count occupancy
    for element in stream:
        count_array[element] += 1.0

    # sort of accupancy and reverse result (from big to small)
    occupany = np.argsort(count_array)[::-1]

    # some debugging
    # print(count_array)
    # print(occupany)

    return occupany


def modify_weight(weights_nn, index, delta):
    """
    :param weights_nn: weights which we want to change
    :param index: index tuple
    :param delta: delta parameter
    :return: return modified weight
    """

    weights_nn[index[0], index[1]] += delta

    return weights_nn


def delta_activations(features, label_single, weights_nn, delta):

    # (w_ij* - x_i)
    print(label_single)
    print(features[label_single[0][:]])
    d1 = 2 * (weights_nn[label_single[0], label_single[1]] - features[label_single[1], :])
    print(np.shape(d1))
    d2 = delta * (delta + d1)


    print(d2)


def update_line(hl, new_data):
    hl.set_xdata(np.append(hl.get_xdata(), new_data[0]))
    hl.set_ydata(np.append(hl.get_ydata(), new_data[1]))
    plt.draw()



# def create_label_stream(feats, weights):


# create some datapoints
# np.random.seed(42)
# data = np.random.uniform(low=-10, high=10, size=[1000, 2])
#
# # cluster data with kmeans
# center, label = kmeans2(data=data, k=10)
# colors = plt.cm.rainbow(np.linspace(0, 1, 10))
#
# # plot data
# if False:
#     plt.scatter(x=data[:, 0], y=data[:, 1], color=colors[label])
#     plt.scatter(x=center[:, 0], y=center[:, 1], color='g', marker='*')
#     plt.show()

misc = Misc()

# read data
if False:
    path = 'data_simple/final_dataset'
    data = read_single_data(path)
    data = data.sample(20000)
    data.to_csv(path + '_new', index=False)
data = pd.read_csv('data_simple/final_dataset_new')
data_numpy = data.values
data_numpy[:, :39] = whiten(data_numpy[:, :39])
# import labels
weights = read_single_data('data_simple/weights')
weights = weights.values
# testing vq

# set uniform weights
# weights = np.random.rand(400, 39)
# load weights
# weights = pd.read_csv('data_simple/weights_calc', index_col=False)
# weights = weights.values[:, 1:]

label, _ = vq(data_numpy[:, :39], weights)

# prepare data
# print(get_transformation_vec())
data_numpy[:, 39] = trans_vec_to_phones(get_transformation_vec(), data_numpy[:, 39])

activations(data_numpy[:, :39], weights)


# calc mutual information
mi_old = misc.calculate_mi(label, data_numpy[:, 39].astype(int))


codebook_size = 400

# for i in xrange(100):


# get the occupancy
# step 1
sorted_labels = sort_label_stream(label, codebook_size)
# calc mutual information
mi_old = misc.calculate_mi(label, data_numpy[:, 39].astype(int))
print(mi_old)
exit()
delta_change = 0.25
# weights = np.random.rand(400, 39)

# step 2
i = 0  # sorted index
j = 0  # activation index
iter = 0
print('Iteration: ' + str(iter))

plt.ion()
fig, ax = plt.subplots()
x, y = [],[]
sc = ax.scatter(x, y)
plt.xlim(0,22)
plt.ylim(0,9)

plt.draw()

# add first iteration
x.append(iter)
y.append(mi_old)
sc.set_offsets(np.c_[x, y])
fig.canvas.draw_idle()
plt.pause(0.1)

while j < 39:
    # print('Activation: ' + str(j))
    new_weights = modify_weight(weights, (sorted_labels[i], j), delta_change)
    new_labels, _ = vq(data_numpy[:, :39], new_weights)

    mi = misc.calculate_mi(new_labels, data_numpy[:, 39].astype(int))

    if mi - mi_old > 0:
        weights = new_weights
        mi_old = mi
        print(mi_old)
        j += 1
        delta_change = np.abs(delta_change)
    else:
        if delta_change < 0:
            delta_change = np.abs(delta_change)
            j += 1

        else:
            delta_change = - delta_change

    if j == 39:
        j = 0
        i += 1
        print('Label number: ' + str(i))

    if i == 400:
        i = 0
        iter += 1
        x.append(iter)
        y.append(mi_old)
        sc.set_offsets(np.c_[x, y])
        fig.canvas.draw_idle()
        plt.pause(0.1)
        print('Iteration: ' + str(iter))
        if iter > 20:
            break
        print('Creating new sorted labels...')
        sorted_labels = sort_label_stream(new_labels, codebook_size)

df_weights = pd.DataFrame(weights)
df_weights.to_csv('data_simple/weights_calc')


# new_labels = activations(data_numpy[:, :39], weights)
# print(np.argmin(new_labels, axis=1))
# # calc mi
# pw_tmp, py_tmp, pyw_tmp = helper_mi(np.argmin(new_labels, axis=1), data_numpy[:, 39].astype(int))
# mi = calculate_mi(pw_tmp, py_tmp, pyw_tmp, 2000)
#
#
# new_labels = np.argmin(new_labels, axis=1)
# delta_activations(data_numpy[:, :39], (np.argmin(new_labels, axis=1), 0), new_weights, 0.1)




