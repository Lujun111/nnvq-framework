#!/home/ga96yar/tensorflow_env/bin/python
# coding: utf-8
import tensorflow as tf
import os
import pandas as pd
import re
import numpy as np
from kaldi_io import kaldi_io
from KaldiHelper.IteratorHelper import DataIterator
from KaldiHelper.MiscHelper import Misc


# TODO no path and file checking so far
# TODO pytorch inference
class TrainedModel(object):
    """
        TrainedModel for performing the inference after training
    """
    def __init__(self, meta_file, stats_file, cond_prob_file, transform_prob=True, log_output=True):
        """
        :param meta_file:       path to meta file (has to be created during training)
        :param stats_file:      path to stats file for normalization (kaldi-format)
        :param cond_prob_file:  path to P(s_k|m_j) file (kaldi-format)
        :param transform_prob:  flag cond_prob_file to create continuous probability
        :param log_output:      flag for setting log-output
        """
        # define some fields
        self.transform = transform_prob     # transform prob to continuous probability (default=True)
        self.log_ouput = log_output         # do log on output (default=True)
        self.list_path = None
        self.global_mean = None
        self.global_var = None
        self.cond_prob = None
        self.prior = None
        self._session = None
        self._graph = None
        self._meta_file = None
        self._checkpoint_folder = None

        # execute some init methods
        self._get_checkpoint_data(meta_file)
        self._create_session()
        self._create_graph()
        self._set_global_stats(stats_file)
        # setting transform matrix
        if self.transform:
            self._set_probabilities(cond_prob_file)

        # hint to produce a discrete output:
        # set transform_prob=False and log_output=False, then you get discrete labels out
        # of the inference model

    def do_inference(self, nj, input_folder, output_folder):
        """
        Does the inference of the model

        :param nj:              number of jobs (how the dataset is split in kaldi)
        :param input_folder:    path to the data folder to do the inference
        :param output_folder:   path to save the output of the inference
        """

        # create DataIterator for iterate through the split folder created by kaldi
        dataset = DataIterator(nj, input_folder)

        # number iterator for counting, necessary for writing the matrices later
        iterator = iter([i for i in range(1, dataset.get_size() + 1)])

        features_all = []
        phoneme_all = []
        inferenced_data = {}  # storing the inferenced data

        while True:
            try:
                data_path = dataset.next_file()  # get path to data
                print(data_path)
                # iterate through data
                for key, mat in kaldi_io.read_mat_ark(data_path):
                    inferenced_data[key] = self._do_single_inference(mat[:, :39])  # do inference for one batch
                    if np.shape(mat)[1] > 39:   # get statistics for mi (only if we input data + labels), for debugging
                        phoneme_all.append(mat[:, 39])
                    # add for debugging, see below
                    features_all.append(self._normalize_data(mat[:, :39]))

                # write posteriors (inferenced data) to files
                with open(output_folder + '/feats_vq_' + str(next(iterator)), 'wb') as f:
                    for key, mat in list(inferenced_data.items()):
                        if self.transform:
                            kaldi_io.write_mat(f, mat, key=key)
                        else:
                            kaldi_io.write_mat(f, mat[:, np.newaxis], key=key)
                inferenced_data = {}  # reset dict

            except StopIteration:
                # hardcoded for mi, for debugging
                if False:
                    misc = Misc()
                    features_all = np.concatenate(features_all)
                    phoneme_all = np.expand_dims(np.concatenate(phoneme_all), 1)
                    phoneme_all = misc.trans_vec_to_phones(phoneme_all)
                    # print(misc.calculate_mi(features_all, phoneme_all))
                    mi, test_py, test_pw, test_pyw = self._session.run(["mutual_info:0", "p_y:0", "p_w:0", "p_yw:0"], feed_dict={"is_train:0": False,
                                                                        "ph_features:0": features_all,
                                                                        "ph_labels:0": phoneme_all})
                    print(mi)
                    tmp_pywtest = pd.DataFrame(test_py)
                    tmp_pywtest.to_csv('py_inf.txt', header=False, index=False)
                    tmp_pywtest = pd.DataFrame(test_pw)
                    tmp_pywtest.to_csv('pw_inf.txt', header=False, index=False)
                    tmp_pywtest = pd.DataFrame(test_pyw)
                    tmp_pywtest.to_csv('pwy_inf.txt', header=False, index=False)

                break

    def _do_single_inference(self, np_mat):
        """
        Does a single inference for a batch (size can be chosen as prefered)

        :param np_mat:  batch of data (e.g. data of size [N, dim_features])
        :return:        output of the network
        """
        variables_names = [v.name for v in tf.trainable_variables()]
        values = self._session.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            print(v)

        # TODO check for None; normalize data (YES/NO)
        # normalize data
        np_mat = self._normalize_data(np_mat)

        # inference
        output = self._session.run("nn_output:0", feed_dict={"ph_features:0": np_mat, "is_train:0": False})

        # transform data with "continuous trick"
        # here same theory:
        # P(m_j) = output
        # P(o_k) = sum_j [ P(m_j) * P(s_k|m_j) ]
        if self.transform:
            if self.cond_prob is not None:
                output = np.dot(output, self.cond_prob)
            else:
                raise ValueError("cond_prob is None, please check!")
        # if we don't do the "continuous trick" we output discrete labels
        # therefore, we use argmax of the output
        else:
            output = np.argmax(output, axis=1)
            output = output.astype(np.float64, copy=False)

        # flag for setting log-output or normal output
        if self.log_ouput:
            output /= self.prior    # divide through prior to get pseudo-likelihood
            output = np.log(output)
        return output

    def _create_session(self):
        """
        Create interactive session to save space on gpu
        """
        self._session = tf.InteractiveSession()

    def _create_graph(self):
        """
        Create graph and load model (file comes out of the training)
        """
        saver = tf.train.import_meta_graph(self._checkpoint_folder + '/' + self._meta_file)
        saver.restore(self._session, tf.train.latest_checkpoint(self._checkpoint_folder))
        self._graph = tf.get_default_graph()

    def _create_list(self, folder_name):
        """
        Takes a folder and create a list of all the files in the folder
        The list is sorted in natural sort order

        :param folder_name: path to folder, to get list of
        """
        self.list_path = [folder_name + s for s in os.listdir(folder_name)]

        # sort list for later processing
        convert = lambda text: int(text) if text.isdigit() else text
        self.list_path.sort(key=lambda key: [convert(c) for c in re.split('([0-9]+)', key)])

    def _set_global_stats(self, file):
        """
        Set the mean and the variance in the class

        :param file: path to stats file (kaldi-format, must contain the keys 'mean' and 'var')
        :return:
        """
        for key, mat in kaldi_io.read_mat_ark(file):
            if key == 'mean':
                print('Setting mean')
                self.global_mean = np.transpose(mat)
            elif key == 'std':
                print('Setting var')
                self.global_var = np.transpose(mat)
            else:
                print('No mean or var set!!!')

    def _set_probabilities(self, file):
        # TODO hard coded for getting class counts --> make sure that file class.counts exists
        # TODO and contains the key class_counts
        """
        Set P(s_k|m_j) and prior P(s_k) from training

        :param file: path to P(s_k|m_j) file (kaldi-format, must contain the key 'p_s_m')
        """
        # Set P(s_k|m_j)
        for key, mat in kaldi_io.read_mat_ark(file):
            if key == 'p_s_m':
                print('Setting P(s_k|m_j)')
                self.cond_prob = np.transpose(mat)  # we transpose for later dot product
            else:
                print('No probability found')

        # Set prior P(s_k)
        for key, mat in kaldi_io.read_mat_ark('../class.counts'):
            if key == 'class_counts':
                print('Setting Prior')
                self.prior = mat / np.sum(mat)
            else:
                print('No Prior found')

    def _normalize_data(self, data_array):
        """
        Normalize data using global mean and variance

        :param data_array: numpy data matrix
        """
        return (data_array - self.global_mean) / self.global_var

    def _get_checkpoint_data(self, checkpoint):
        """
        Find the last checkpoint for loading model

        :param path to checkpoint file
        """
        assert type(checkpoint) == str

        # split string
        _, self._checkpoint_folder, self._meta_file = str.split(checkpoint, '/')
        self._checkpoint_folder = '../' + self._checkpoint_folder


if __name__ == "__main__":
    # set transform_prob=False if you don't want to get a continuous output (default is True)

    # flag for model type
    discrete = False

    if discrete:
        # discrete model
        model_discrete = TrainedModel('../model_checkpoint/saved_model-99.meta', '../stats_20k.mat',
                                      '../p_s_m.mat', log_output=False, transform_prob=False)

        model_discrete.do_inference(20, '/features/train_20k/feats', '/home/ga96yar/kaldi/egs/tedlium/s5_r2/'
                                                                     '/exp/test_400_0/vq_train')
        model_discrete.do_inference(30, 'test', '/home/ga96yar/kaldi/egs/tedlium/s5_r2/'
                                                'exp/test_400_0/vq_test')
    else:
        # continuous model
        model_continuous = TrainedModel('../model_checkpoint/saved_model-99.meta', '../stats_20k.mat',
                                        '../p_s_m.mat',)
        # model_continuous.do_inference(20, 'features/train_20k/feats', '../tmp/tmp_testing')
        model_continuous.do_inference(30, 'test', '../tmp/tmp_testing')







