#!/home/ga96yar/tensorflow_env/bin/python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd
import os


def read_single_data(path_to_data):

    # init data matrix
    data_all = pd.read_csv(path_to_data, sep='\s+', header=None)
    return data_all


def get_transformation_vec():
    # define convert vector to get the phones
    trans_vec = np.zeros(166)
    for i in range(10):
        if i < 5:
            trans_vec[i] = 0;
        else:
            trans_vec[i] = 1;

    index = 2
    count = 10

    for i in range(10, trans_vec.shape[0]):
        if (i - count) == 4:
            index += 1
            count = i
        trans_vec[i] = index

    return trans_vec


def trans_vec_to_phones(trans_vec, label_vec):

    phone_vec = np.zeros(label_vec.shape)
    for i in range(label_vec.shape[0]):
        phone_vec[i] = trans_vec[int(label_vec[i])]

    return phone_vec


def normalize_data(data_array, g_mean, g_std):
    return (data_array - g_mean) / g_std

def convert_npy_to_tfrecords(npy_array, path_tfrecords):
    with tf.python_io.TFRecordWriter(path_tfrecords) as tf_writer:
        for row in npy_array:
            features, label = row[:39], row[39]
            example = tf.train.Example(features = tf.train.Features(feature={
                'x': tf.train.Feature(float_list=tf.train.FloatList(value=features.flatten())),
                'y': tf.train.Feature(float_list=tf.train.FloatList(value=label.flatten()))}))
            tf_writer.write(example.SerializeToString())


if __name__ == "__main__":
    path = '../plain_feats_20k/test'
    global_stats = read_single_data('stats.txt')
    global_std, global_mean = global_stats.values[0, :], global_stats.values[1, :]
    for item in os.listdir(path):
        print('Converting ' + item)
        data = read_single_data(path + '/' + item)
        data_norm = np.zeros(data.values.shape)
        data_numpy = data.values
        # convert states to phones
        data_norm[:, 39] = trans_vec_to_phones(get_transformation_vec(), data_numpy[:, 39])
        # normalize data
        data_norm[:, :39] = normalize_data(data_numpy[:, :39], global_mean, global_std)
        # get number of item
        num_item = str.split(item, '_')[2]
        convert_npy_to_tfrecords(data_norm, path + '/data_' + str(num_item) + '.tfrecords')
        print(item + ' converted')
    print("Data converted to TFRecords")