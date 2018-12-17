#!/home/ga96yar/tensorflow_py3/bin/python
import tensorflow as tf
import re
import sys
import random
import argparse
import pandas as pd
import numpy as np
from kaldi_io import kaldi_io
from KaldiHelper.IteratorHelper import DataIterator, AlignmentIterator


class Misc(object):
    """
    Misc class contains auxiliary functions
    """
    def __init__(self, nj, state_based, splice, cmvn, dim=39):
        # TODO can we determine the dim on the fly?
        self._trans_vec = None
        self.global_mean = None
        self.global_std = None

        self._nj = nj
        self._state_based = state_based
        self._splice = splice
        self._dim = dim
        self._cmvn = cmvn

        # set dims for calc mi
        self._dim_seq1 = 512
        self._dim_seq2 = 512

        # set transformation vector
        # here some infos:
        # - there are 40 phonemes for training (with SIL)
        # - for triphone system there are 166 phonemes (left/right context, variants of the
        # same phoneme)
        # - to train a monophone system (40 different labels), we have to map
        # the 166 phonemes to 40 (in case the labels are coming from a triphone model,
        # what is usually the case)
        # - we create an transformation vector which maps the 166 phonemes to 40
        self._get_transformation_vec()

    def calculate_mi(self, stats, floor_val=1e-15):
        # TODO use dict for y_nn and labels is the best way?
        """
        Calculate the mutual information between two sets (here labels and phonemes)

        :param labels:      labels coming e.g. out of a neural network
        :param phonemes:    phonemes from e.g. alignments
        :param nn_output:   flag if neural network output or just a label stream
        :return:            mutual information between labels and phonemes
        """
        alpha = 1.0
        beta = -1.0

        # check type of labels

        assert(isinstance(stats, dict))

        # floor
        stats['p_w'] += floor_val
        stats['p_y'] += floor_val
        stats['p_w_y'] += floor_val

        # normalize
        stats['p_y'] /= np.sum(stats['p_y'])
        stats['p_w'] /= np.sum(stats['p_w'])
        stats['p_w_y'] /= np.sum(stats['p_w_y'], axis=0, keepdims=True)

        # H(Y) on log2 base
        stats['h_y'] = -np.dot(stats['p_y'], np.log2(stats['p_y']))
        stats['h_w'] = -np.dot(stats['p_w'], np.log2(stats['p_w']))

        # H(Y|W) on log2 base
        stats['h_w_y'] = stats['p_w_y'] * np.log2(stats['p_w_y'])  # log2 base
        stats['h_w_y'] = np.sum(stats['h_w_y'], axis=0)  # reduce sum
        stats['h_w_y'] = -np.dot(stats['p_y'], stats['h_w_y'])

        # create a dict for returning values
        result = {
            'MI': alpha * stats['h_w'] + beta * stats['h_w_y'],
            'H(W)': stats['h_w'],
            'H(Y)': stats['h_y'],
            'H(W|Y)': stats['h_w_y']
        }

        return result

    def calculate_mi_tf(self, y_nn, phonemes, codebook_size):
        # TODO quite old, does it work properly
        """
        Calculate the mutual information in tensorflow between two sets (here labels and phonemes)

        :param y_nn:            labels coming e.g. out of a neural network
        :param phonemes:        phonemes from e.g. alignments
        :param codebook_size:   codebook size of the neural network (output dim)
        :return:                mutual information between labels and phonemes
        """
        alpha = 1.0
        beta = -1.0

        # get p_w, p_y and p_w_y from helper
        p_w, p_y, p_w_y = self._helper_mi_tf(y_nn, phonemes, codebook_size)

        # normalize
        p_w /= tf.reduce_sum(p_w)
        p_y /= tf.reduce_sum(p_y)
        p_w_y = tf.divide(p_w_y,
                          tf.expand_dims(tf.clip_by_value(tf.reduce_sum(p_w_y, axis=1),
                                                          1e-8, 1e6), 1))
        # # H(Y) on log2 base
        h_y = tf.multiply(p_y, tf.log(tf.clip_by_value(p_y, 1e-8, 1e6)) / tf.log(2.0))
        h_y = tf.reduce_sum(h_y)

        # H(W) on log2 base
        h_w = tf.multiply(p_w, tf.log(tf.clip_by_value(p_w, 1e-8, 1e6)) / tf.log(2.0))
        h_w = tf.reduce_sum(h_w)

        # H(W|Y) on log2 base
        h_w_y = p_w_y * tf.log(tf.clip_by_value(p_w_y, 1e-12, 1e6)) / tf.log(2.0)  # log2 base
        h_w_y = tf.reduce_sum(h_w_y, axis=0)  # reduce sum
        h_w_y = tf.multiply(p_y, h_w_y)
        h_w_y = tf.reduce_sum(h_w_y)

        return -alpha * h_w - beta * h_w_y, -h_w, -h_y, -h_w_y

    def helper_mi(self, y_nn, labels, stats, nn_output=False):
        """
        Helper functions to get P(w), P(y) and P(w|y)

        :param y_nn:        output nn (dict)
        :param labels:      labels (dict)
        :return:            P(w), P(y) and P(w|y)
        """

        assert (isinstance(stats, dict))

        # if nn output, do argmax
        if nn_output:
            try:
                y_nn = np.argmax(y_nn, axis=1)
            except KeyError:
                print('Key not found!')

        # convert shared phonems to single
        if not self._state_based:
            try:
                labels = self.trans_vec_to_phones(labels)
            except KeyError:
                print('Key not found!')

        # use input array as indexing array
        stats['p_w'][y_nn] += 1.0
        stats['p_y'][labels] += 1.0
        stats['p_w_y'][y_nn, labels] += 1.0

        # test = pd.DataFrame(py_tmp)
        # test.to_csv('test_inference.txt', header=False, index=False)
        # print(np.sum(test.values))

        return stats

    def _helper_mi_tf(self, labels, alignments, cb_len):
        """
        Helper functions in tensorflow to get P(w), P(y) and P(w|y)

        :param labels:      labels coming e.g. out of a neural network
        :param alignments:  phonemes from e.g. alignments
        :param cb_len:      codebook size of the neural network (output dim)
        :return:            P(w), P(y) and P(w|y)
        """
        p = 41

        pwtmp = tf.Variable(tf.zeros(p), trainable=False, dtype=tf.float32)
        pytmp = tf.Variable(tf.zeros(cb_len), trainable=False, dtype=tf.float32)
        pw_y_tmp = tf.Variable(tf.zeros([p, cb_len]), trainable=False, dtype=tf.float32)

        # use input array as indexing array
        pwtmp = pwtmp.assign(tf.fill([p], 0.0))  # reset Variable/floor
        pwtmp = tf.scatter_add(pwtmp, alignments, tf.ones(tf.shape(alignments)))

        pytmp = pytmp.assign(tf.fill([cb_len], 0.0))  # reset Variable/floor
        pytmp = tf.scatter_add(pytmp, labels, tf.ones(tf.shape(labels)))

        pw_y_tmp = pw_y_tmp.assign(tf.fill([p, cb_len], 0.0))  # reset Variable/floor
        pw_y_tmp = tf.scatter_nd_add(pw_y_tmp,
                                     tf.concat([tf.cast(alignments, dtype=tf.int64), tf.expand_dims(labels, 1)],
                                               axis=1), tf.ones(tf.shape(labels)))
        return pwtmp, pytmp, pw_y_tmp

    def _get_transformation_vec(self):
        """
        Create transformation vector to do the mapping from 166 phonemes to 40 phonemes

        trans_vec[0:10]     = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        trans_vec[10:165]   = [2, 2, 2, 3, 3, 3, ... , 40, 40, 40]

        ATTENTION: Yes, we are using 41 phonemes, however, 1 "phoneme" equals to "NSN" (no-
        spoken noise) which is not used. But is easier to extract without changing a lot
        """
        # define convert vector to get the phones
        trans_vec = np.zeros(166, dtype=int)
        for i in range(10):
            if i < 5:
                trans_vec[i] = 0
            else:
                trans_vec[i] = 1

        index = 2
        count = 10

        for i in range(10, trans_vec.shape[0]):
            if (i - count) == 4:
                index += 1
                count = i
            trans_vec[i] = index

        self._trans_vec = trans_vec

    def trans_vec_to_phones(self, label_vec):
        """
        Do the mapping from 166 phonemes to 41 phonemes

        :param label_vec:   vector with labels for transformation
        :return:            return the transformed vector
        """
        return self._trans_vec[label_vec.astype(int)]

    def _normalize_data(self, data_array):
        """
        Normalize the matrix with global mean and glabal variance (row wise)
        :param data_array:  input data matrix
        :return:            return the normalized matrix
        """
        return (data_array - self.global_mean) / self.global_std

    def _convert_npy_to_tfrecords(self, npy_array, path_tfrecords):
        """
        Auxiliary function for the actual creation of the TFRecord files

        :param transformation:      do the mapping from 166 phonemes to 41 phonemes
        :param npy_array:           source data for creating the TFRecord file
        :param path_tfrecords:      path to save the TFRecord files
        :return:
        """

        # set dim for splice
        # set dim
        dim = self._dim * (2 * self._splice + 1)

        with tf.python_io.TFRecordWriter(path_tfrecords) as tf_writer:
            for row in npy_array:
                # print(row[:39])
                if self._state_based:
                    label = np.expand_dims(row[dim], 1)
                else:
                    label = self.trans_vec_to_phones(np.expand_dims(row[dim], 1))

                if not self._cmvn:
                    features = self._normalize_data(row[:dim])  # global normalization
                else:
                    features = row[:dim]

                if np.nan in label:
                    print('Error, check for nan!')
                # print(features)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'x': tf.train.Feature(float_list=tf.train.FloatList(value=features.flatten())),
                    'y': tf.train.Feature(float_list=tf.train.FloatList(value=label.flatten()))}))
                tf_writer.write(example.SerializeToString())

    def create_tfrecords(self, stats, path_input, path_output):
        # TODO refactor to KaldiMiscHelper ???
        """
        Create the TFRecord files

        :param nj: number of jobs split into in data folder
        :param trans_phoneme: transform to single phoneme (41) or multi (166)
        :param splice: splice feats (1 left and 1 right context)
        :param stats: stats-file to normalize data
        :param path_input: path to the folder where the features + phonemes are
        :param path_output: output path to save the tfrecords
        :return:
        """
        assert type(path_input) == str and type(path_output) == str

        # TODO hard-coded
        for key, mat in kaldi_io.read_mat_ark(stats):
            if key == 'mean':
                print('Setting mean')
                self.global_mean = np.transpose(mat)[0, :]
                # print(self.global_mean.shape)
            elif key == 'std':
                print('Setting std')
                self.global_std = np.transpose(mat)[0, :]
            else:
                print('No mean or var set!!!')

        dataset = DataIterator(self._nj, path_input, splice=self._splice, cmvn=self._cmvn)

        tmp_df = pd.DataFrame()
        count = 1
        while True:
            try:
                for _, mat in kaldi_io.read_mat_ark(dataset.next_file()):
                    tmp_df = pd.concat([tmp_df, pd.DataFrame(mat)])

                self._convert_npy_to_tfrecords(tmp_df.values, path_output + '/data_' +
                                               str(count) + '.tfrecords')
                print('/data_' + str(count) + '.tfrecords created')

                count += 1
                tmp_df = pd.DataFrame()

            except StopIteration:
                break

    def test_mmi(self, folder_gmm, folder_nn):

        # create iterator
        ali_gmm = AlignmentIterator(self._nj, folder_gmm, state_based=self._state_based)
        ali_nn = AlignmentIterator(self._nj, folder_nn, convert=True, state_based=self._state_based)

        # print dim
        print('dim seq1: ' + str(ali_gmm.dim))
        print('dim seq2: ' + str(ali_nn.dim))

        # create base arrays for collecting data
        stats = {
            'p_w': np.zeros(ali_gmm.dim),
            'p_y': np.zeros(ali_nn.dim),
            'p_w_y': np.zeros([ali_gmm.dim, ali_nn.dim])
        }

        while True:
            try:
                # get dicts
                dict_w = dict(kaldi_io.read_ali_ark(ali_gmm.next_file()))
                dict_y = dict(kaldi_io.read_ali_ark(ali_nn.next_file()))

                # find mutual keys in both dicts
                keys = [key for key in dict_w if key in dict_y]

                # gather counts for all alignments
                for key in keys:
                    if self._state_based:
                        stats['p_w'][dict_w[key]] += 1.0
                        stats['p_y'][dict_y[key]] += 1.0
                        stats['p_w_y'][dict_w[key], dict_y[key]] += 1.0
                    else:
                        stats['p_w'][self._trans_vec[dict_w[key]]] += 1.0
                        stats['p_y'][self._trans_vec[dict_y[key]]] += 1.0
                        stats['p_w_y'][self._trans_vec[dict_y[key]],
                                       self._trans_vec[dict_w[key]]] += 1.0

            except StopIteration:
                # calculate MI
                print(self.calculate_mi(stats))

                break


def str2bool(v):
    """
    Converts string argument to bool
    :param v:
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(arguments):
    """
    Create argument parser to execute python file from console
    """
    print(arguments)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # add number of jobs / how the data is split
    parser.add_argument('--nj', type=int, help='number of jobs', default=20)
    # flag for state-based or phoneme-based labels
    parser.add_argument('--state-based', type=str2bool, help='flag for state-based or phoneme-based labels',
                        default=True)
    # splice features with context range
    parser.add_argument('--splice', type=int, help='flag for spliced features with context width',
                        default=0)
    # cmvn or global normalization
    parser.add_argument('--cmvn', type=str2bool, help='flag for cmvn or global normalization',
                        default=True)
    # define the path to the stats file
    parser.add_argument('stats', type=str, help='path to stats file')
    # define the folder which should be converted to TFRecords
    parser.add_argument('in_folder', type=str, help='alignment folder which contains the labels of the data')
    # define the output folder where to save the TFRecords files
    parser.add_argument('out', type=str, help='output folder to save the concat data')

    # parse all arguments to parser
    args = parser.parse_args(arguments)

    # print the arguments which we fed into
    for arg in vars(args):
        print("Argument {:14}: {}".format(arg, getattr(args, arg)))

    # create object and perform task
    misc = Misc(args.nj, args.state_based, args.splice, args.cmvn)
    misc.create_tfrecords(args.stats, args.in_folder, args.out)
    print('Created TFRecords')

if __name__ == "__main__":
    misc = Misc(35, False, 0, False)
    misc.test_mmi('mono_ali', 'tri1')
    # sys.exit(main(sys.argv[1:]))