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


class Management(object):
    """
    This class should manage everything
    """
    # def __init__(self, model, loss, optimizer, logger, datafeeder, settings, job_name='TRAIN'):
    def __init__(self):
        # define some fields
        self._session = None
        self._graph = None
        self._meta_file = None
        self._model = None
        self._misc = None
        self._loss = None
        self._optimizer = None
        self._merged = None

        # misc
        self._mutual_information = None
        self._joint_probability = None
        self._conditioned_entropy = None

        # job to do
        self._job_name = None

        # feeding input (tuple)
        self._input_train = None
        self._input_test = None
        self._input_dev = None

        # placeholders
        self._ph_train = None
        self._ph_labels = None
        self._ph_features = None
        self._ph_lr = None

        # vars
        self._global_step = None

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

    def restore_model(self, meta_data):
        saver = tf.train.import_meta_graph(meta_data)
        saver.restore(self._session, tf.train.latest_checkpoint(os.path.dirname(meta_data)))
        self._graph = tf.get_default_graph()

    def _init_vars(self):
        if self._session is not None:
            self._session(tf.global_variables_initializer())
        else:
            raise BaseException('Session is None, init session!')

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

    def _init_model(self):
        self._model = Model(self._ph_train, self._ph_features, Settings)

    def _init_misc(self):
        self._misc = MiscNN(Settings)

    def _init_data_feeder(self):
        dict_lists_tfrecords = {
            'train': [Settings.path_train + '/' + s for s in os.listdir(Settings.path_train)],
            'test': [Settings.path_test + '/' + s for s in os.listdir(Settings.path_test)],
            'dev': [Settings.path_dev + '/' + s for s in os.listdir(Settings.path_dev)]}

        # define feeder
        feeder = DataFeeder(dict_lists_tfrecords, Settings, self._session)

        # create 3 pipelines for features
        self._input_train = feeder.train.get_next()
        self._input_test = feeder.test.get_next()
        self._input_dev = feeder.dev.get_next()

    def _init_variables(self):
        self._global_step = tf.Variable(0, trainable=False)

    def _global_init(self):
        self._init_session()
        self._init_graph()
        self._init_data_feeder()
        self._init_placeholder()
        self._init_variables()
        self._init_model()
        self._init_pretrain()

    def _init_before_train(self):
        self._loss = Loss(self._model.inference, self._ph_labels)
        self._optimizer = Optimizer(Settings, self._loss, self._global_step)
        self._mutual_information = self._misc.calculate_mi_tf(self._model.inference_learned, self._ph_labels)
        self._joint_probability = self._misc.joint_probability(self._model.inference_learned, self._ph_labels)
        # self._conditioned_entropy = -tf.reduce_sum(self._joint_probability * tf.log(cond_prob))

        self._session.run(tf.global_variables_initializer())

    def _train_single_epoch(self):
        # set something to test

        time_string = time.strftime('%d.%m.%Y - %H:%M:%S')
        train_writer = tf.summary.FileWriter(Settings.path_tensorboard + '/training_' + time_string, self._graph)

        while True:
            try:
                feat, labs = self._session.run([self._input_train[0], self._input_train[1]])

                # check for exponential decayed learning rate and set it

                _, loss_value, summary, count, mi, y_print, y_debug = self._session.run(
                    [self._optimizer.train_op, self._loss.loss, merged, self._global_step, self._mutual_information, self._model.inference_learned, self._model.inference],
                    feed_dict={self._ph_train: True, self._ph_features: feat, self._ph_labels: labs, self._ph_lr: Settings.learning_rate})

                if count % 100:
                    train_writer.add_summary(summary, count)
                    summary_tmp = tf.Summary()
                    summary_tmp.value.add(tag='train/mutual_information', simple_value=mi[0])
                    summary_tmp.value.add(tag='train/H(w)', simple_value=mi[1])
                    summary_tmp.value.add(tag='train/H(y)', simple_value=mi[2])
                    summary_tmp.value.add(tag='train/H(w|y)', simple_value=mi[3])
                    summary_tmp.value.add(tag='misc/learning_rate', simple_value=Settings.learning_rate)
                    train_writer.add_summary(summary_tmp, count)
                    train_writer.flush()

            except tf.errors.OutOfRangeError:
                # print(nom_vq/den_vq)
                print('loss: ' + str(loss_value))
                print('max: ' + str(np.max(y_print)))
                print('min: ' + str(np.min(y_print)))
                train_writer.add_summary(summary, count)
                summary_tmp = tf.Summary()
                train_writer.add_summary(summary_tmp, count)
                train_writer.flush()
                break



class Logger(object):
    def __init__(self):