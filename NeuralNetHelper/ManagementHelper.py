import tensorflow as tf
import pandas as pd
from tensorflow.python import debug as tf_debug
import os
import time
import numpy as np
from kaldi_io import kaldi_io
from NeuralNetHelper.MiscNN import MiscNN
from NeuralNetHelper.ModelHelper import Model
from NeuralNetHelper import Settings
from NeuralNetHelper.DataFeedingHelper import DataFeeder


class Management(object):
    """
    This class should manage everything
    """
    # def __init__(self, model, loss, optimizer, logger, datafeeder, settings, job_name='TRAIN'):
    def __init__(self, model):
        # define some fields
        # self.transform = transform_prob     # transform prob to continuous probability (default=True)
        # self.log_ouput = log_output         # do log on output (default=True)
        self.list_path = None
        self.global_mean = None
        self.global_var = None
        self.cond_prob = None
        self.prior = None
        self._session = None
        self._graph = None
        self._meta_file = None
        self._checkpoint_folder = None
        self.model = model
        self._job = None

        tf.reset_default_graph()
        self._create_session()

    def _create_session(self):
        """
        Create interactive session to save space on gpu
        """
        self._session = tf.Session()
        # self._session.run(tf.global_variables_initializer())

    def _create_graph(self):
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

    def test(self):
        # print('here')
        print(self._session.run("nn_output:0", feed_dict={"ph_features:0": np.random.rand(256, 39), "is_train:0": False}))

        # variables_names = [v.name for v in tf.trainable_variables()]
        # values = self._session.run(variables_names)
        # for k, v in zip(variables_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
        #     print(v)


