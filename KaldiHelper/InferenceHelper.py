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
from NeuralNetHelper.MiscNNHelper import MiscNN
import matplotlib.pyplot as plt


# TODO no path and file checking so far
# TODO pytorch inference
class InferenceModel(object):
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
        self._session1 = None
        self._graph = tf.Graph()
        self._graph1 = tf.Graph()
        self._meta_file = None
        self._checkpoint_folder = None
        self._dev_alignments = {}

        # execute some init methods
        # self._get_checkpoint_data(meta_file)
        self._create_session()
        self._create_graph(meta_file, identifier=None)
        # self.init_object(session)
        self._set_global_stats(stats_file)
        # setting transform matrix
        if self.transform:
            self._set_probabilities(cond_prob_file)
        self._load_dev_alignemnts()
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

        features_all = {}
        phoneme_all = {}
        inferenced_data = {}  # storing the inferenced data
        check_data = {}
        output_all = {}

        while True:
            try:
                data_path = dataset.next_file()  # get path to data
                print(data_path)
                # iterate through data
                for key, mat in kaldi_io.read_mat_ark(data_path):
                    inferenced_data[key] = self._do_single_inference(mat[:, :39])  # do inference for one batch
                    tmp = self._do_single_inference(mat[:, :39])
                    # check_data[key] = [np.argmax(tmp[0], axis=1), np.argmax(tmp[1], axis=1),
                    #                    np.argmax(tmp[2], axis=1), self._dev_alignments[key]]
                    if np.shape(mat)[1] > 39:   # get statistics for mi (only if we input data + labels), for debugging
                        phoneme_all[key] = mat[:, 39]
                    # add for debugging, see below
                    output_all[key] = tmp

                # write posteriors (inferenced data) to files
                with open(output_folder + '/feats_vq_' + str(next(iterator)), 'wb') as f:
                    for key, mat in list(inferenced_data.items()):
                        if self.transform:
                            kaldi_io.write_mat(f, mat, key=key)
                        else:
                            kaldi_io.write_mat(f, mat[:, np.newaxis], key=key)
                inferenced_data = {}  # reset dict

            except StopIteration:
                # debugging
                # gather_right = np.zeros(127)
                # gather_right.fill(1e-5)
                # gather_wrong = np.zeros(127)
                # gather_wrong.fill(1e-5)
                # gather_vq = np.zeros(127)
                # gather_vq.fill(1e-5)
                # gather_comb = np.zeros(127)
                # gather_comb.fill(1e-5)
                #
                # for key, entry in check_data.items():
                #     tmp_van = entry[0] == entry[3]  # right pred of vanilla
                #     tmp_vq = entry[1] == entry[3]  # right pred of vanilla
                #     tmp_comb = entry[2] == entry[3]  # right pred of vanilla
                #
                #     # np.max(np.expand_dims(~tmp_vq, 1) * output_all[key], axis=1)
                #
                #     comb_right = [t for t, x in enumerate(tmp_comb) if x]
                #     comb_wrong = [t for t, x in enumerate(~tmp_comb) if x]
                #     vq_right = [t for t, x in enumerate(tmp_vq) if x]
                #     vq_wrong = [t for t, x in enumerate(~tmp_vq) if x]
                #     van_right = [t for t, x in enumerate(tmp_van) if x]
                #     van_wrong = [t for t, x in enumerate(~tmp_van) if x]
                #
                #     list_vq = ~(entry[0] == entry[3]) == (entry[1] == entry[3])
                #     list_comb = (entry[0] == entry[3]) == ~(entry[2] == entry[3])
                #     ind_vq_true = [t for t, x in enumerate(list_vq) if x]
                #     ind_comb_true = [t for t, x in enumerate(list_comb) if x]
                #     ind_vq_false = [t for t, x in enumerate(list_vq) if not x]
                #     # est = output_all[key][1]
                #
                #
                #     # plt.subplot(2, 1, 1)
                #     # # plt.hist(np.ndarray.flatten(np.expand_dims(list_vq, 1) * output_all[key]), bins=100, range=[1e-15, 1])
                #     # plt.hist(-np.sum(np.log2(output_all[key][0]) * output_all[key][0], axis=1), bins=10)
                #     # plt.subplot(2, 1, 2)
                #     # # plt.hist(np.ndarray.flatten(np.expand_dims(~list_vq, 1) * output_all[key]), bins=100, range=[1e-15, 1])
                #     # plt.hist(-np.sum(np.log2(output_all[key][1]) * output_all[key][1], axis=1), bins=10)
                #     # plt.show()
                #
                #     print('right comb: ' + str(len(comb_right)))
                #     print('wrong comb: ' + str(len(comb_wrong)))
                #     print('right vq: ' + str(len(vq_right)))
                #     print('wrong vq: ' + str(len(vq_wrong)))
                #     print('right van: ' + str(len(van_right)))
                #     print('wrong van: ' + str(len(van_wrong)))
                #     # print(len(van_right) + len(van_wrong))
                #     # print(entry[2][van_wrong])
                #     gather_right[entry[3][comb_right]] += 1.0
                #     gather_wrong[entry[3][comb_wrong]] += 1.0
                #     gather_vq[entry[3][ind_vq_true]] += 1.0
                #     gather_comb[entry[3][ind_comb_true]] += 1.0
                #     # print(len(van_right) + len(van_wrong))
                #     # print(len(entry[2]))
                #     print(sum(list_comb) / len(entry[3]))
                #     print(sum(list_vq) / len(entry[3]))

                # plt.subplot(3, 1, 1)
                # plt.bar(range(0, 127), gather_right)
                # plt.subplot(3, 1, 2)
                # plt.bar(range(0, 127), gather_wrong)
                # plt.subplot(3, 1, 3)
                # plt.bar(range(0, 127), gather_vq)
                # plt.show()
                # print(check_data[0] == check_data[1])
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
        # variables_names = [v.name for v in tf.trainable_variables()]
        # values = self._session.run(variables_names)
        # for k, v in zip(variables_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
        #     print(v)
        # exit()

        # TODO check for None; normalize data (YES/NO)
        # normalize data
        np_mat = self._normalize_data(np_mat)
        # inference
        # with self._graph.as_default() as g:
        #     logits = g.get_tensor_by_name("vanilla_network/nn_output:0")
        #     sess = tf.InteractiveSession()
        #     sess.run(tf.global_variables_initializer())
        #     output = sess.run(logits,
        #                            feed_dict={"ph_features:0": np_mat, "is_train:0": False})
        # logits = self._graph.get_tensor_by_name("combination_network/nn_output:0")
        # # print(logits)
        # output = self._session.run(logits, feed_dict={"ph_features:0": np_mat, "is_train:0": False})
        # output += 1e-30
        output_van = self._session.run("vanilla_network/nn_output:0", feed_dict={"ph_features:0": np_mat, "is_train:0": False, "train_output:0": False})
        output_vq = self._session.run("base_network/nn_output:0", feed_dict={"ph_features:0": np_mat, "is_train:0": False, "train_output:0": False})
        output_comb = self._session.run("combination_network/nn_output:0", feed_dict={"ph_features:0": np_mat, "is_train:0": False, "train_output:0": False})

        # print(np.max(output, axis=1))
        # transform data with "continuous trick"
        # here same theory:
        # P(m_j) = output
        # P(o_k) = sum_j [ P(m_j) * P(s_k|m_j) ]
        # TODO what to do if we have no transform
        if self.transform:
            if self.cond_prob is not None:
                # print("Transform output to 127 pdf...")
                output_vq = np.dot(output_vq, self.cond_prob)
                # pass
            else:
                raise ValueError("cond_prob is None, please check!")
        # if we don't do the "continuous trick" we output discrete labels
        # therefore, we use argmax of the output
        else:
            output_vq = np.argmax(output, axis=1)
            output_vq = output_vq.astype(np.float64, copy=False)

        # print(np.argmax(output_van, axis=1))
        # print(np.argmax(output_vq, axis=1))

        # tmp = np.sum(np.argmax(output_van, axis=1) == np.argmax(output_vq, axis=1)) / np.shape(output_van)[0]
        # print(tmp)
        # output = self.posterior_combination(np.log(output_van), np.log(output_vq), 0.25)

        # output = np.log(self.min_max_combination(output_van, output_vq))
        # output -= np.log(self.prior)
        output = output_comb
        output += 1e-30
        # # flag for setting log-output or normal output
        if self.log_ouput:
            # print(np.min(output))
            output /= self.prior    # divide through prior to get pseudo-likelihood
            output = np.log(output)
        return output

    @staticmethod
    def min_max_combination(posterior_1, posterior_2):
        """
        Taking the maximum posterior

        :param log_posterior_1:
        :param log_posterior_2:
        :return:
        """
        max_post_1 = np.max(posterior_1, axis=1)
        max_post_2 = np.max(posterior_2, axis=1)

        # max mask reference to post 1
        mask_max = max_post_1 > max_post_2

        combined_post = np.expand_dims(mask_max, 1) * posterior_1 + np.expand_dims(~mask_max, 1) * posterior_2

        return combined_post





    @staticmethod
    def posterior_combination(log_posterior_1, log_posterior_2, alpha):
        """
        Doing the posterior combination with log posteriors

        :param log_posterior_1:
        :param log_posterior_2:
        :param alpha:
        :return:
        """
        return alpha * log_posterior_1 + (1 - alpha) * log_posterior_2

    def _create_session(self):
        """
        Create interactive session to save space on gpu
        """
        self._session = tf.Session(graph=self._graph)
        # self._session.run(tf.global_variables_initializer())
        self._session1 = tf.Session(graph=self._graph1)
        # self._session1.run(tf.global_variables_initializer())

    def _create_graph(self, meta_file, identifier=None):
        """
        Create graph and load model (file comes out of the training)
        """
        # print(os.path.dirname(meta_file))
        if identifier is not None:
            path_van = os.path.dirname(meta_file) + '/van_graph'
            path_vq = os.path.dirname(meta_file) + '/vq_graph'
            with self._graph.as_default():
                saver = tf.train.import_meta_graph(path_van + '/saved_model.meta')
                saver.restore(self._session, tf.train.latest_checkpoint(path_van))
            # tf.reset_default_graph()
            # self._create_session()
            #
            # self._graph1 = tf.get_default_graph()
            with self._graph1.as_default():
                saver = tf.train.import_meta_graph(path_vq + '/saved_model.meta')
                saver.restore(self._session1, tf.train.latest_checkpoint(path_vq))
            # tf.reset_default_graph()

        else:
            path = os.path.dirname(meta_file)
            with self._graph.as_default():
                saver = tf.train.import_meta_graph(path + '/saved_model.meta')
                saver.restore(self._session, tf.train.latest_checkpoint(path))


        # saver = tf.train.import_meta_graph(meta_file)
        # saver.restore(self._session, tf.train.latest_checkpoint(os.path.dirname(meta_file)))
        # self._graph = tf.get_default_graph()

        #  with self._graph.as_default():
        #     saver = tf.train.import_meta_graph(path_van + '/saved_model.meta')
        #     saver.restore(self._session, tf.train.latest_checkpoint(path_van))
        # # tf.reset_default_graph()
        # # self._create_session()
        # #
        # # self._graph1 = tf.get_default_graph()
        # with self._graph1.as_default():
        #     saver = tf.train.import_meta_graph(path_vq + '/saved_model.meta')
        #     saver.restore(self._session1, tf.train.latest_checkpoint(path_vq))
        # tf.reset_default_graph()


        # list_restore = [v for v in tf.trainable_variables()]
        # print(list_restore)
        # self._session.run(tf.global_variables_initializer())
        # saver.restore(self._session, tf.train.latest_checkpoint(path_van))
        # saver1.restore(self._session, tf.train.latest_checkpoint(path_vq))

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
                # print(np.sum(self.cond_prob, axis=1))
                # print(np.shape(np.sum(self.cond_prob, axis=1)))
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

    def init_object(self, session):
        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())

    def _load_dev_alignemnts(self):
        for key, mat in kaldi_io.read_ali_ark('../tmp/state_labels/all_ali_dev'):
            self._dev_alignments[key] = mat
        print('Finished loading dev alignments')


if __name__ == "__main__":
    # set transform_prob=False if you don't want to get a continuous output (default is True)

    # flag for model type
    discrete = False

    if discrete:
        # discrete model
        model_discrete = InferenceModel('../model_checkpoint/saved_model-99.meta', '../stats_20k.mat',
                                      '../p_s_m.mat', log_output=False, transform_prob=False)

        model_discrete.do_inference(20, '/features/train_20k/feats', '/home/ga96yar/kaldi/egs/tedlium/s5_r2/'
                                                                     '/exp/test_400_0/vq_train')
        model_discrete.do_inference(30, 'test', '/home/ga96yar/kaldi/egs/tedlium/s5_r2/'
                                                'exp/test_400_0/vq_test')
    else:
        # continuous model
        model_continuous = InferenceModel('../model_checkpoint/saved_model.meta', '../stats_20k.mat',
                                        '../p_s_m.mat',)
        # model_continuous.do_inference(20, 'features/train_20k/feats', '../tmp/tmp_testing')
        model_continuous.do_inference(30, 'test', '../tmp/tmp_testing')







