import tensorflow as tf
import sys
import argparse
import pandas as pd
import numpy as np
from kaldi_io import kaldi_io
from KaldiHelper.IteratorHelper import DataIterator, AlignmentIterator


class Misc(object):
    """
    Misc class contains auxiliary functions
    """
    def __init__(self):
        # print('Init Misc')
        self._trans_vec = None
        self.global_mean = None
        self.global_var = None

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

    def calculate_mi(self, labels, phonemes):
        # TODO quite old, does it work properly
        """
        Calculate the mutual information between two sets (here labels and phonemes)

        :param labels:      labels coming e.g. out of a neural network
        :param phonemes:    phonemes from e.g. alignments
        :return:            mutual information between labels and phonemes
        """
        alpha = 1.0
        beta = -1.0

        # convert shared phonems to single
        phonemes = self.trans_vec_to_phones(phonemes)

        p_w, p_y, p_w_y = self._helper_mi(labels.astype(np.int32), phonemes.astype(np.int32))
        # clipping
        p_w[p_w <= 1e-15] = 1e-15
        p_y[p_y <= 1e-15] = 1e-15
        p_w_y[p_w_y <= 1e-15] = 1e-15

        # normalize
        p_w /= np.sum(p_w)
        p_y /= np.sum(p_y)
        p_w_y_scale = np.sum(p_w_y, axis=1)
        p_w_y /= np.expand_dims(p_w_y_scale, 1)

        # H(Y) on log2 base
        h_y = np.dot(p_y, np.log2(p_y))
        h_w = np.dot(p_w, np.log2(p_w))

        # H(Y|W) on log2 base
        h_w_y = p_w_y * np.log2(p_w_y) # log2 base
        h_w_y = np.sum(h_w_y, axis=0)  # reduce sum
        h_w_y = np.dot(p_y, h_w_y)
        return -alpha * h_w - beta * h_w_y, -h_w, -h_y, -h_w_y

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

    def _helper_mi(self, labels, alignments):
        """
        Helper functions to get P(w), P(y) and P(w|y)

        :param labels:      labels coming e.g. out of a neural network
        :param alignments:  phonemes from e.g. alignments
        :return:            P(w), P(y) and P(w|y)
        """
        pw_tmp = np.zeros(41)
        py_tmp = np.zeros(400)
        pw_y_tmp = np.zeros([41, 400])

        # use input array as indexing array
        for element in alignments:
            pw_tmp[element] += 1.0
        for element in labels:
            py_tmp[element] += 1.0
        for element in zip(alignments, labels):
            pw_y_tmp[element[0], element[1]] += 1.0

        test = pd.DataFrame(py_tmp)
        test.to_csv('test_inference.txt', header=False, index=False)
        # print(np.sum(test.values))

        return pw_tmp, py_tmp, pw_y_tmp

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
        trans_vec = np.zeros(166)
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
        label_vec = label_vec.astype(int)
        phone_vec = np.zeros(label_vec.shape)
        for i in range(label_vec.shape[0]):
            phone_vec[i] = self._trans_vec[label_vec[i]]

        return phone_vec

    def _normalize_data(self, data_array):
        """
        Normalize the matrix with global mean and glabal variance (row wise)
        :param data_array:  input data matrix
        :return:            return the normalized matrix
        """
        return (data_array - self.global_mean) / self.global_var

    def _convert_npy_to_tfrecords(self, transformation, splice, npy_array, path_tfrecords):
        """
        Auxiliary function for the actual creation of the TFRecord files

        :param transformation:      do the mapping from 166 phonemes to 41 phonemes
        :param npy_array:           source data for creating the TFRecord file
        :param path_tfrecords:      path to save the TFRecord files
        :return:
        """

        # set dim for splice
        # TODO could be solved in a better way
        if splice:
            dim = 117
        else:
            dim = 39

        with tf.python_io.TFRecordWriter(path_tfrecords) as tf_writer:
            for row in npy_array:
                # print(row[:39])
                if transformation:
                    features, label = self._normalize_data(row[:dim]), np.expand_dims(row[dim], 1)
                else:
                    features, label = self._normalize_data(row[:dim]), self.trans_vec_to_phones(np.expand_dims(row[dim], 1))
                if np.nan in label:
                    print('Error, check for nan!')
                # print(features)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'x': tf.train.Feature(float_list=tf.train.FloatList(value=features.flatten())),
                    'y': tf.train.Feature(float_list=tf.train.FloatList(value=label.flatten()))}))
                tf_writer.write(example.SerializeToString())

    def create_tfrecords(self, nj, trans_phoneme, splice, stats, path_input, path_output):
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
                print(self.global_mean.shape)
            elif key == 'std':
                print('Setting std')
                self.global_var = np.transpose(mat)[0, :]
            else:
                print('No mean or var set!!!')

        dataset = DataIterator(nj, splice, path_input)

        tmp_df = pd.DataFrame()
        count = 1
        while True:
            try:
                for _, mat in kaldi_io.read_mat_ark(dataset.next_file()):
                    tmp_df = pd.concat([tmp_df, pd.DataFrame(mat)])

                self._convert_npy_to_tfrecords(trans_phoneme, splice, tmp_df.values, path_output + '/data_' + str(count) + '.tfrecords')
                print('/data_' + str(count) + '.tfrecords created')

                count += 1
                tmp_df = pd.DataFrame()

            except StopIteration:
                break

    def calculate_mi_merged(self, nj, labels_folder, phoneme_folder):
        # TODO quite old, does it work properly
        """

        :param nj:
        :param labels_folder:
        :param phoneme_folder:
        :return:
        """
        labels_iter = DataIterator(nj, labels_folder)
        phoneme_iter = AlignmentIterator(nj, phoneme_folder)

        tmp_df = pd.DataFrame()
        while True:
            try:
                for (key_lab, mat_lab), (key_pho, mat_pho) in zip(kaldi_io.read_mat_ark(labels_iter.next_file()),
                                                                  kaldi_io.read_ali_ark(phoneme_iter.next_file())):

                    if key_lab == key_pho:
                        tmp_df = pd.concat([tmp_df, pd.concat([pd.DataFrame(mat_lab), pd.DataFrame(mat_pho)], axis=1)])

            except StopIteration:
                print(self.calculate_mi(tmp_df.values[:, 0], tmp_df.values[:, 1]))
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
    # define the path to the stats file
    parser.add_argument('stats', type=str, help='path to stats file')
    # define the folder which should be converted to TFRecords
    parser.add_argument('in_folder', type=str, help='alignment folder which contains the labels of the data')
    # define the output folder where to save the TFRecords files
    parser.add_argument('out', type=str, help='output folder to save the concat data')
    # define splice-feats or not
    parser.add_argument('--splice', type=str2bool, help='flag for spliced features', default=False)
    # parse all arguments to parser
    args = parser.parse_args(arguments)

    # print the arguments which we fed into
    for arg in vars(args):
        print("Argument {:14}: {}".format(arg, getattr(args, arg)))

    # create object and perform task
    misc = Misc()
    misc.create_tfrecords(args.nj, args.state_based, args.splice, args.stats, args.in_folder, args.out)
    print('Created TFRecords')

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
    # test.calculate_mi_merged(30, '../plain_feats/backup_20k_vq/400/vq_test', '../alignments/train')
    # test.create_tfrecords(20, False, '../stats_20k.mat', '../tmp/feat_20k_state', '../tmp')

    # test.create_tfrecords(30, False, 'stats_20k.mat', '../features/dev', '../tf_data/dev')

    # test = Misc()
    # test.create_tfrecords(30, False, '../stats_20k.mat', '../tmp/feat_test_state', '../tmp')