import tensorflow as tf
import pandas as pd
from tensorflow.python import debug as tf_debug
import os
import time
import numpy as np
from kaldi_io import kaldi_io
from NeuralNetHelper.MiscNNHelper import MiscNN
from NeuralNetHelper.ModelHelper import Model
from NeuralNetHelper import Settings
from NeuralNetHelper.DataFeedingHelper import DataFeeder
from NeuralNetHelper.LossHelper import Loss
from NeuralNetHelper.OptimizerHelper import Optimizer
from KaldiHelper.InferenceHelper import InferenceModel


class Management(object):
    """
    This class manages everything
    """
    # TODO interface for Management object?
    def __init__(self):
        # define some fields
        self._session = None
        self._graph = None
        self._saver = None
        self._meta_file = None
        self._feeder = None
        self._model = None
        self._misc = None
        self._loss = None
        self._optimizer = None
        self._merged = None
        self._train_writer = None
        self._output_model = None

        # misc
        self._mutual_information = None
        self._joint_probability = None
        self._conditioned_probability = None
        self._conditioned_entropy = None
        self._data_vqed = None

        # job to do
        self._job_name = None
        self._current_mi = -10.0
        self._count = 0.0

        # feeding input (tuple)
        self._input_train = None
        self._input_test = None
        self._input_dev = None

        # placeholders
        self._ph_train = None
        self._ph_labels = None
        self._ph_features = None
        self._ph_lr = None
        self._ph_conditioned_probability = None

        # vars
        self._global_step = None
        self._nominator = None
        self._denominator = None

        # init
        self._global_init()

    def _init_session(self):
        """
        Create interactive session to save space on gpu
        """
        self._session = tf.Session()
        # self._session.run(tf.global_variables_initializer())

    def _init_graph(self):
        """
        Create graph and load model (file comes out of the training)
        """
        self._graph = tf.get_default_graph()

    def restore_model(self, meta_data=None):
        # if we call this function, set restore boolean
        Settings.restore = True
        self._session.run(tf.global_variables_initializer())

        # decide what path to use
        if meta_data is not None:
            path_checkpoint = meta_data
        else:
            path_checkpoint = Settings.path_restore

        # check if there is a checkpoint
        if tf.train.latest_checkpoint(meta_data) is not None:
            print('Restore old model and train it further..')
            self._saver.restore(self._session, tf.train.latest_checkpoint(path_checkpoint))
        else:
            print('Cannot find a checkpoint with a model, starting to train a new model...')

    # def _init_vars(self):
    #     if self._session is not None:
    #         self._session = tf.Session()
    #     else:
    #         raise BaseException('Session is None, init session!')

    def _get_job(self, job_name):
        assert type(job_name) is str
        # convert to upper case
        job_name = job_name.upper()
        # define list to find the job to do
        job_list = ['TRAIN', 'RESTORE', 'INFERENCE']

        if job_name in job_list:
            self._job = job_name
        else:
            raise SyntaxError("Please define your job!"
                              "Options are: TRAIN, RESTORE and INFERENCE")

    def _init_placeholder(self):
        # define your placeholders here
        self._ph_train = tf.placeholder(tf.bool, name="is_train")
        self._ph_labels = tf.placeholder(tf.float32, shape=[None, Settings.dim_labels], name='ph_labels')
        self._ph_lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self._ph_features = tf.placeholder(tf.float32, shape=[None, Settings.dim_features], name='ph_features')
        self._ph_conditioned_probability = tf.placeholder(tf.float32, shape=None, name='ph_conditioned_probability')

    def _init_model(self):
        self._model = Model(self._ph_train, self._ph_features, Settings)

    def _init_saver(self):
        # list_restore = [v for v in tf.trainable_variables()]
        # print(list_restore[:6])
        self._saver = tf.train.Saver()

    def _init_misc(self):
        self._misc = MiscNN(Settings)

    def _init_data_feeder(self):
        dict_lists_tfrecords = {
            'train': [Settings.path_train + '/' + s for s in os.listdir(Settings.path_train)],
            'test': [Settings.path_test + '/' + s for s in os.listdir(Settings.path_test)],
            'dev': [Settings.path_dev + '/' + s for s in os.listdir(Settings.path_dev)]}

        # define feeder
        self._feeder = DataFeeder(dict_lists_tfrecords, Settings, self._session)

        # create 3 pipelines for features
        self._input_train = self._feeder.train.get_next()
        self._input_test = self._feeder.test.get_next()
        self._input_dev = self._feeder.dev.get_next()

    def _init_variables(self):
        self._global_step = tf.Variable(0, trainable=False)
        self._nominator = tf.Variable(tf.zeros([Settings.num_labels, Settings.codebook_size]), trainable=False,
                                      dtype=tf.float32)
        self._denominator = tf.Variable(tf.zeros([Settings.codebook_size]), trainable=False, dtype=tf.float32)

    def _global_init(self):
        self._init_session()
        self._init_graph()
        self._init_data_feeder()
        self._init_placeholder()
        self._init_variables()
        self._init_model()
        self._init_misc()
        self._init_saver()
        self._init_before_train()

    def _init_before_train(self):

        self._mutual_information = self._misc.calculate_mi_tf(self._model.inference, self._ph_labels)
        self._joint_probability = self._misc.joint_probability(self._model.inference, self._ph_labels)
        self._conditioned_probability = self._misc.conditioned_probability(self._model.inference,
                                                                           self._ph_labels,
                                                                           discrete=Settings.sampling_discrete)

        self._conditioned_entropy = -tf.reduce_sum(self._joint_probability * tf.log(self._conditioned_probability))
        self._data_vqed = self._misc.vq_data(self._model.inference, self._ph_labels, self._nominator,
                                             self._denominator)

        # init variables if we don't restore a model
        # if not Settings.restore:
        #     self._session.run(tf.global_variables_initializer())
        if Settings.train_prob:
            # train last layer (to produce P(s_k|m_j)) or use hand-made P(s_k|m_j)
            self._loss = Loss(self._model.logits_new, self._ph_labels, cond_prob=None)
        else:
            self._loss = Loss(self._model.inference, self._ph_labels, cond_prob=self._conditioned_probability)

        self._optimizer = Optimizer(Settings, self._loss.loss, self._global_step, var_list=self._var_list_training())

        tf.summary.scalar('train/loss', self._loss.loss)
        tf.summary.scalar('misc/conditioned_entropy', self._conditioned_entropy)

        # logging
        self._merged = tf.summary.merge_all()

        time_string = time.strftime('%d.%m.%Y - %H:%M:%S')
        self._train_writer = tf.summary.FileWriter(Settings.path_tensorboard + '/training_' + time_string, self._graph)

    @staticmethod
    def _var_list_training(number=None):

        if number is None:
            print('No identify string, train all parameters')
            return None
        elif number.upper() == 'BASE':
            print('Optimizing the base parameters')
            return [v for v in tf.trainable_variables('base_network')]
        elif number.upper() == 'ADDED':
            return [v for v in tf.trainable_variables('added_network')]
        elif number.upper() == 'ALL':
            print('Optimizing all trainable parameters')
            return [v for v in tf.trainable_variables()]
        else:
            print('Cannot identify string, returning full list')
            return [v for v in tf.trainable_variables()]

    def train_single_epoch(self):
        """
        Train a single epoch
        """

        self._feeder.init_train()

        while True:
            try:
                feat, labs = self._session.run([self._input_train[0], self._input_train[1]])

                # check for exponential decayed learning rate and set it
                _, loss_value, summary, self._count, mi, y_print = self._session.run(
                    [self._optimizer.train_op, self._loss.loss, self._merged, self._global_step,
                     self._mutual_information, self._model.inference],
                    feed_dict={self._ph_train: True, self._ph_features: feat, self._ph_labels: labs,
                               self._ph_lr: Settings.learning_rate})

                if self._count % 100:
                    self._train_writer.add_summary(summary, self._count)
                    summary_tmp = tf.Summary()
                    summary_tmp.value.add(tag='train/mutual_information', simple_value=mi[0])
                    summary_tmp.value.add(tag='train/H(w)', simple_value=mi[1])
                    summary_tmp.value.add(tag='train/H(y)', simple_value=mi[2])
                    summary_tmp.value.add(tag='train/H(w|y)', simple_value=mi[3])
                    summary_tmp.value.add(tag='misc/learning_rate', simple_value=Settings.learning_rate)
                    self._train_writer.add_summary(summary_tmp, self._count)
                    self._train_writer.flush()

            except tf.errors.OutOfRangeError:
                # print(nom_vq/den_vq)
                print('loss: ' + str(loss_value))
                print('max: ' + str(np.max(y_print)))
                print('min: ' + str(np.min(y_print)))
                self._train_writer.add_summary(summary, self._count)
                summary_tmp = tf.Summary()
                self._train_writer.add_summary(summary_tmp, self._count)
                self._train_writer.flush()
                break

    def do_validation(self):
        """
        Perform validation on the current model
        """

        self._feeder.init_dev()
        features_all = []
        labels_all = []

        while True:
            try:
                feat, labs = self._session.run([self._input_dev[0], self._input_dev[1]])
                features_all.append(feat)
                labels_all.append(labs)

            except tf.errors.OutOfRangeError:
                # reshape data
                features_all = np.concatenate(features_all)
                labels_all = np.concatenate(labels_all)

                # mi_test = sum_mi / self._count_mi
                mi_vald = self._session.run(self._mutual_information, feed_dict={self._ph_train: False, self._ph_features:
                    features_all, self._ph_labels: labels_all})
                print(mi_vald)
                summary_tmp = tf.Summary()
                summary_tmp.value.add(tag='validation/mutual_information', simple_value=mi_vald[0])
                self._train_writer.add_summary(summary_tmp, self._count)
                self._train_writer.flush()

                # print(self._count_mi)
                # TODO save current mi
                if mi_vald[0] > self._current_mi:
                    print('Saving better model...')
                    self._saver.save(self._session, Settings.path_checkpoint + '/saved_model')
                    self._current_mi = mi_vald[0]
                break

    def create_p_s_m(self):

        self._feeder.init_train()

        # set model.train to False to avoid training
        # model.train = False
        while True:
            try:
                feat, labs = self._session.run([self._input_train[0], self._input_train[1]])

                nom_vq, den_vq = self._session.run(self._data_vqed, feed_dict={self._ph_train: False, self._ph_features: feat,
                                                                               self._ph_labels: labs})

            except tf.errors.OutOfRangeError:
                nom_vq += Settings.delta
                den_vq += Settings.num_labels * Settings.delta
                prob = nom_vq / den_vq

                # saving matrix with kaldi_io
                save_dict = {'p_s_m': prob}
                print('Saving P(s_k|m_j)')
                with open('p_s_m.mat', 'wb') as f:
                    for key, mat in list(save_dict.items()):
                        kaldi_io.write_mat(f, mat, key=key)

                # reset den and nom
                self._session.run([self._misc.reset_variable(self._nominator),
                                   self._misc.reset_variable(self._denominator)])

                break

    def do_inference(self, stats_file='stats_20k.mat', cond_prob_file='p_s_m.mat', transform_prob=True, log_output=True):
        inference = InferenceModel(self._session, stats_file,
                                   cond_prob_file, transform_prob=transform_prob, log_output=log_output)
        # model_continuous.do_inference(20, 'features/train_20k/feats', '../tmp/tmp_testing')
        inference.do_inference(30, 'test', 'tmp/tmp_testing')
