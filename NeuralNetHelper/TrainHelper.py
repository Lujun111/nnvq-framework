# -*- coding: utf-8 -*
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


class Train(object):
    """
    This class manages everything
    """
    # TODO interface for Management object?
    def __init__(self, session, data_feeder, misc, model, saver, placeholders, variables):
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
        self._train_dict = None

        # misc
        self._mutual_information = None
        self._joint_probability = None
        self._conditioned_probability = None
        self._conditioned_entropy = None
        self._data_vqed = None
        self._accuracy = None
        self._p_s_m = None

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
        self._ph_last_layer = None

        # vars
        self._global_step = None
        self._nominator = None
        self._denominator = None

        # init
        self._global_init()

    # Decorator Functions:
    def show_timing(text='', verbosity=2):
        """ Decorator function for easy displaying executed function and timing

        inputs:
        - text: Additional Information specific for this execution of the function to be displayed
        - verbosity: 1 -> Displays only the function name | 2 -> Displays function name and timing | 0 -> Displays nothing

        Use it with an @ in front and in the line directly above you function like this:

        ...
        @show_timing(text='additional information to be displayed', verbosity=2)
        a = func(b)
        ...

        """
        def wrapper(func):
            def function_wrapper(*args, **kwargs):
                if verbosity > 0:
                    if verbosity > 1:
                        time_start = time.time()
                        print('Executing: ' + '<' + func.__name__ + '>' + ' (%s)' % text + ' ...  ', end="", flush=True)
                        output = func(*args, **kwargs)
                        print('DONE[(%.2f sec)]' % (time.time() - time_start))
                    else:
                        print('Executing: ' + '<' + func.__name__ + '>' + ' (%s)' % text)
                        output = func(*args, **kwargs)
                else:
                    output = func(*args, **kwargs)
                return output

            return function_wrapper

        return wrapper

    def _init_session(self):
        """
        Create interactive session to save space on gpu
        """
        self._session = tf.Session()
        # self._session.run(tf.local_variables_initializer())
        # self._session.run(tf.global_variables_initializer())

    def _init_graph(self):
        """
        Create graph and load model (file comes out of the training)
        """
        self._graph = tf.get_default_graph()

    def restore_model(self, meta_data=None):
        # if we call this function, set restore boolean
        Settings.restore = True

        # self._session.run(tf.global_variables_initializer())
        # self._session.run(tf.local_variables_initializer())

        # decide what path to use
        # if meta_data is not None:
        #     path_checkpoint = meta_data
        # else:
        #     path_checkpoint = Settings.path_restore
        #
        # # check if there is a checkpoint
        # if tf.train.latest_checkpoint(meta_data) is not None:
        #     print('Restore old model and train it further..')
        #     self._saver.restore(self._session, tf.train.latest_checkpoint(path_checkpoint))
        # else:
        #     print('Cannot find a checkpoint with a model, starting to train a new model...')

        path_van = meta_data + '/van_graph'
        path_vq = meta_data + '/vq_graph'

        # variables_names = [v.name for v in tf.trainable_variables()]
        # print(variables_names)
        # # values = self._session.run(variables_names)
        # # for k, v in zip(variables_names, values):
        # #     print("Variable: ", k)
        # #     print("Shape: ", v.shape)
        # #     print(v)
        # exit()

        van_collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vanilla_network')
        vq_collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='base_network')
        saver = tf.train.Saver(var_list=van_collection)
        saver.restore(self._session, tf.train.latest_checkpoint(path_van))
        saver = tf.train.Saver(var_list=vq_collection)
        saver.restore(self._session, tf.train.latest_checkpoint(path_vq))

        saver = tf.train.import_meta_graph(path_van + '/saved_model.meta')
        saver.restore(self._session, tf.train.latest_checkpoint(path_van))
        with self._graph.as_default():
            saver = tf.train.import_meta_graph(path_vq + '/saved_model.meta')
            saver.restore(self._session, tf.train.latest_checkpoint(path_vq))

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
        self._ph_last_layer = tf.placeholder(tf.bool, name="train_output")

    def _init_model(self):
        # import P(s_k|m_j)
        self._p_s_m = self._misc.set_probabilities('model_checkpoint/vq_graph/p_s_m.mat')
        self._model = Model(self._ph_train, self._ph_features, Settings, input_tensor=self._p_s_m,
                            ph_output=self._ph_last_layer)

    def _init_saver(self):
        # list_restore = [v for v in tf.trainable_variables()]
        # print(list_restore[:6])
        self._saver = tf.train.Saver()
        # pass

    def _init_misc(self):
        self._misc = MiscNN(Settings)

    def _init_data_feeder(self):

        # define feeder
        self._feeder = DataFeeder(Settings, self._session)

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
        self._init_misc()
        self._init_model()
        self._init_saver()
        self._init_before_train(Settings.identifier)

    def _init_before_train(self, identifier=None):
        # we create different dicts depending on the task
        self._train_dict = {}

        if 'nnvq' in identifier:
            # create dict
            self._train_dict['nnvq'] = {'mi': self._misc.calculate_mi_tf(self._model.inference, self._ph_labels),
                         'joint_prob': self._misc.joint_probability(self._model.inference, self._ph_labels),
                         'cond_prob': self._misc.conditioned_probability(self._model.inference, self._ph_labels,
                                                                         discrete=Settings.sampling_discrete),
                         'data_vq': self._misc.vq_data(self._model.inference, self._ph_labels, self._nominator,
                                                       self._denominator),
                         'loss': Loss(self._model.inference, self._ph_labels,
                                      cond_prob=self._misc.conditioned_probability(self._model.inference, self._ph_labels,
                                                                                   discrete=Settings.sampling_discrete),
                                      identifier='nnvq').loss,
                         'train_op': Optimizer(Settings.learning_rate_pre,
                                               loss=Loss(self._model.inference, self._ph_labels,
                                                         cond_prob=self._misc.conditioned_probability(
                                                             self._model.inference, self._ph_labels,
                                                             discrete=Settings.sampling_discrete), identifier='nnvq').loss,
                                               control_dep=tf.get_collection(tf.GraphKeys.UPDATE_OPS)).get_train_op(
                             global_step=self._global_step),
                         'output': self._model.inference,
                         'count': self._global_step}

            # create loggers
            tf.summary.scalar('train/loss', self._train_dict['nnvq']['loss'])
            tf.summary.scalar('train/mutual_information', self._train_dict['nnvq']['mi'][0])
            tf.summary.scalar('train/H(w)', self._train_dict['nnvq']['mi'][1])
            tf.summary.scalar('train/H(y)', self._train_dict['nnvq']['mi'][2])
            tf.summary.scalar('train/H(w|y)', self._train_dict['nnvq']['mi'][3])
            tf.summary.scalar('misc/learning_rate', Settings.learning_rate_pre)
            self._train_dict['nnvq']['merge'] = tf.summary.merge_all()

        elif identifier == 'vanilla':
            # create codebook
            self._train_dict = {
            'vanilla': {'loss': Loss(self._model.logits, self._ph_labels,
                                  identifier='vanilla').loss,
                        'train_op': Optimizer(Settings.learning_rate_pre,
                                           loss=Loss(self._model.logits, self._ph_labels, identifier='vanilla').loss,
                                           control_dep=tf.get_collection(tf.GraphKeys.UPDATE_OPS)).get_train_op(
                         global_step=self._global_step),
                        'accuracy': tf.metrics.accuracy(self._ph_labels,
                                                        tf.argmax(self._model.inference, axis=1)),
                        'output': self._model.inference,
                        'count': self._global_step
                        }
            }

            # create loggers
            tf.summary.scalar('train/loss', self._train_dict[identifier]['loss'])
            tf.summary.scalar('misc/learning_rate', Settings.learning_rate_pre)
            tf.summary.scalar('train/accuracy', self._train_dict[identifier]['accuracy'][0])
            self._train_dict[identifier]['merge'] = tf.summary.merge_all()

        elif identifier == 'nnvq':
            self._train_dict = {
                'nnvq': {'loss': Loss(self._model.logits, self._ph_labels,
                                         identifier='nnvq', cond_prob=self._misc.conditioned_probability(
                        self._model.inference_nnvq, self._ph_labels, discrete=Settings.sampling_discrete)).loss,
                        'train_op': Optimizer(Settings.learning_rate_pre,
                                      loss=Loss(self._model.logits, self._ph_labels,
                                                identifier='nnvq', cond_prob=self._misc.conditioned_probability(
                self._model.inference_nnvq, self._ph_labels, discrete=Settings.sampling_discrete)).loss,
                                                  control_dep=tf.get_collection(tf.GraphKeys.UPDATE_OPS)).get_train_op(
                                global_step=self._global_step),
                        'mi': self._misc.calculate_mi_tf(self._model.inference_nnvq, self._ph_labels),
                        'output': self._model.inference,
                        'count': self._global_step,
                        'data_vq': self._misc.vq_data(self._model.inference_nnvq, self._ph_labels, self._nominator,
                                                       self._denominator),
                            }
            }

            # create loggers
            tf.summary.scalar('train/loss', self._train_dict[identifier]['loss'])
            tf.summary.scalar('misc/learning_rate', Settings.learning_rate_pre)

            self._train_dict[identifier]['merge'] = tf.summary.merge_all()

        elif identifier == 'combination':
            # define loss out of 3 losses
            l1_vanilla = Loss(self._model.logits_vanilla, self._ph_labels, identifier='vanilla').loss
            l2_vq = Loss(self._model.inference_nnvq, self._ph_labels, cond_prob=self._misc.conditioned_probability(
                self._model.inference_nnvq, self._ph_labels, discrete=Settings.sampling_discrete), identifier='nnvq').loss
            l3_combination = Loss(self._model.logits_combination, self._ph_labels, identifier='vanilla').loss

            loss = 0.45 * l1_vanilla + 0.1 * l2_vq + 0.45 * l3_combination
            # loss = l2_vq

            # Loss(self._model.inference_combination, self._ph_labels, identifier='own').loss

            # scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='nnvq_network')
            self._train_dict = {
                'combination': {'mi': self._misc.calculate_mi_tf(self._model.inference_nnvq, self._ph_labels),
                                'loss': loss,

                                'accuracy_vanilla': tf.metrics.accuracy(self._ph_labels,
                                                                        tf.argmax(self._model.inference_vanilla, axis=1)),
                                'accuracy_combination': tf.metrics.accuracy(self._ph_labels,
                                                                        tf.argmax(self._model.inference_combination, axis=1)),
                                'output': self._model.inference_combination,
                                'count': self._global_step,
                                'data_vq': self._misc.vq_data(self._model.inference_nnvq, self._ph_labels, self._nominator,
                                                              self._denominator),
                                'train_op': Optimizer(Settings.learning_rate_pre,
                                                      loss=loss,
                                                      control_dep=tf.get_collection(
                                                          tf.GraphKeys.UPDATE_OPS)).get_train_op(
                                    global_step=self._global_step,
                                    var_list=None)


                }
            }

        if 'restore' in identifier:
            self._train_dict['restore'] = {'loss': Loss(self._model.logits_combination, self._ph_labels,
                                         identifier='vanilla').loss,
                        'train_op': Optimizer(Settings.learning_rate_pre,
                                      loss=Loss(self._model.logits_combination, self._ph_labels,
                                                identifier='vanilla').loss,
                                                  control_dep=tf.get_collection(tf.GraphKeys.UPDATE_OPS)).get_train_op(
                                global_step=self._global_step, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                          scope='combination_network')),
                        'output': self._model.inference_combination,
                        'count': self._global_step,
                        'accuracy_combination': tf.metrics.accuracy(self._ph_labels,
                                                                        tf.argmax(self._model.inference_combination,
                                                                                  axis=1))
                                           }

        if 'front' in identifier:
            print('here')
            # define loss out of 3 losses
            l1_vanilla = Loss(self._model.logits_vanilla, self._ph_labels, identifier='vanilla').loss
            l2_vq = Loss(self._model.inference_nnvq, self._ph_labels, cond_prob=self._misc.conditioned_probability(
                self._model.inference_nnvq, self._ph_labels, discrete=Settings.sampling_discrete), identifier='nnvq').loss

            loss = 0.5 * l1_vanilla + 0.5 * l2_vq

            var_base = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='base_network')
            var_van = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vanilla_network')
            var_group = var_base + var_van
            # loss = l2_vq

            # Loss(self._model.inference_combination, self._ph_labels, identifier='own').loss

            # scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='nnvq_network')
            self._train_dict['front'] = {'mi': self._misc.calculate_mi_tf(self._model.inference_nnvq, self._ph_labels),
                                'loss': loss,

                                'accuracy_vanilla': tf.metrics.accuracy(self._ph_labels,
                                                                        tf.argmax(self._model.inference_vanilla, axis=1)),
                                'output': self._model.inference_vanilla,
                                'count': self._global_step,
                                'data_vq': self._misc.vq_data(self._model.inference_nnvq, self._ph_labels, self._nominator,
                                                              self._denominator),
                                'train_op': Optimizer(Settings.learning_rate_pre,
                                                      loss=loss,
                                                      control_dep=tf.get_collection(
                                                          tf.GraphKeys.UPDATE_OPS)).get_train_op(
                                    global_step=self._global_step,
                                    var_list=var_group)


                                }
        # create loggers
        tf.summary.scalar('train/loss_front', self._train_dict['front']['loss'])
        tf.summary.scalar('misc/learning_rate', Settings.learning_rate_pre)

        self._train_dict['front']['merge'] = tf.summary.merge_all()
        self._session.run(tf.local_variables_initializer())
        self._session.run(tf.global_variables_initializer())

        # 'train_op': Optimizer(Settings.learning_rate_pre,
        #                       loss=loss,
        #                       control_dep=tf.get_collection(
        #                           tf.GraphKeys.UPDATE_OPS)).get_train_op(
        #     global_step=self._global_step,
        #     var_list=None)

        # 'alpha': tf.reduce_max(self._model.alpha)
        # 'train_op': Optimizer(Settings.learning_rate_pre, loss=loss,
        #                       control_dep=tf.get_collection(
        #                           tf.GraphKeys.UPDATE_OPS)).get_train_op(global_step=self._global_step),
        # tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='combination_network')

        # create loggers
        # tf.summary.scalar('train/loss', self._train_dict[identifier]['loss'])
        # tf.summary.scalar('train/mutual_information', self._train_dict[identifier]['mi'][0])
        # tf.summary.scalar('train/H(w)', self._train_dict[identifier]['mi'][1])
        # tf.summary.scalar('train/H(y)', self._train_dict[identifier]['mi'][2])
        # tf.summary.scalar('train/H(w|y)', self._train_dict[identifier]['mi'][3])
        # tf.summary.scalar('misc/learning_rate', Settings.learning_rate_pre)
        # tf.summary.scalar('train/accuracy_vanilla', self._train_dict[identifier]['accuracy_vanilla'][0])
        # tf.summary.scalar('train/accuracy_combination', self._train_dict[identifier]['accuracy_combination'][0])
        # self._train_dict[identifier]['merge'] = tf.summary.merge_all()

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

    @show_timing(text='train', verbosity=2)
    def train_single_epoch(self, identifier=None, train_bn=True):
        """
        Train single epoch

        :param train_last_layer:    Flag for training the last layer only
        :return:
        """

        self._feeder.init_train()

        while True:
            try:
                feat, labs = self._session.run([self._input_train[0], self._input_train[1]])
                # variables_names = [v.name for v in tf.trainable_variables()]
                # values = self._session.run(variables_names)
                # for k, v in zip(variables_names, values):
                #     print("Variable: ", k)
                #     print("Shape: ", v.shape)
                #     print(v)
                # exit()

                return_dict = self._session.run(self._train_dict[Settings.identifier],
                                                feed_dict={self._ph_train: train_bn, self._ph_features: feat,
                                                           self._ph_labels: labs,
                                                           self._ph_lr: Settings.learning_rate_pre,
                                                           self._ph_last_layer: True})

                # print(return_dict['test'][0])
                # check for exponential decayed learning rate and set it
                # _, loss_value, summary, self._count, acc, y_print = self._session.run(
                #     [self._optimizer, self._loss.loss, self._merged, self._global_step,
                #      self._accuracy, self._model.inference],
                #     feed_dict={self._ph_train: is_train, self._ph_features: feat, self._ph_labels: labs,
                #                self._ph_lr: Settings.learning_rate_pre, self._ph_last_layer: train_last_layer})
                #
                if return_dict['count'] % 100:
                    summary_tmp = tf.Summary()
                    if Settings.identifier == 'combination':
                        summary_tmp.value.add(tag='train/mutual_information', simple_value=return_dict['mi'][0])
                        summary_tmp.value.add(tag='train/H(w)', simple_value=return_dict['mi'][1])
                        summary_tmp.value.add(tag='train/H(y)', simple_value=return_dict['mi'][2])
                        summary_tmp.value.add(tag='train/H(w|y)', simple_value=return_dict['mi'][3])
                        summary_tmp.value.add(tag='misc/learning_rate', simple_value=Settings.learning_rate_pre)
                        summary_tmp.value.add(tag='train/acc_vanilla', simple_value=return_dict['accuracy_vanilla'][0])
                        summary_tmp.value.add(tag='train/accuracy',
                                              simple_value=return_dict['accuracy_combination'][0])
                    elif Settings.identifier == 'nnvq':
                        summary_tmp.value.add(tag='train/mutual_information', simple_value=return_dict['mi'][0])
                        summary_tmp.value.add(tag='train/H(w)', simple_value=return_dict['mi'][1])
                        summary_tmp.value.add(tag='train/H(y)', simple_value=return_dict['mi'][2])
                        summary_tmp.value.add(tag='train/H(w|y)', simple_value=return_dict['mi'][3])
                        summary_tmp.value.add(tag='misc/learning_rate', simple_value=Settings.learning_rate_pre)
                    elif Settings.identifier == 'vanilla':
                        summary_tmp.value.add(tag='train/accuracy', simple_value=return_dict['accuracy'][0])
                        summary_tmp.value.add(tag='misc/learning_rate', simple_value=Settings.learning_rate_pre)
                    elif Settings.identifier == 'restore':
                        summary_tmp.value.add(tag='train/accuracy', simple_value=return_dict['accuracy_combination'][0])
                        summary_tmp.value.add(tag='misc/learning_rate', simple_value=Settings.learning_rate_pre)
                    if Settings.identifier == 'front':
                        summary_tmp.value.add(tag='train/mutual_information', simple_value=return_dict['mi'][0])
                        summary_tmp.value.add(tag='train/H(w)', simple_value=return_dict['mi'][1])
                        summary_tmp.value.add(tag='train/H(y)', simple_value=return_dict['mi'][2])
                        summary_tmp.value.add(tag='train/H(w|y)', simple_value=return_dict['mi'][3])
                        summary_tmp.value.add(tag='misc/learning_rate', simple_value=Settings.learning_rate_pre)
                        summary_tmp.value.add(tag='train/acc_vanilla', simple_value=return_dict['accuracy_vanilla'][0])
                    summary_tmp.value.add(tag='train/loss', simple_value=return_dict['loss'])
                    # summary_tmp.value.add(tag='misc/alpha', simple_value=return_dict['alpha'])
                    self._train_writer.add_summary(summary_tmp, return_dict['count'])
                    self._train_writer.flush()

            except tf.errors.OutOfRangeError:
                # print(nom_vq/den_vq)
                # print('loss: ' + str(loss_value))
                # print('max: ' + str(np.max(y_print)))
                # print('min: ' + str(np.min(y_print)))
                # print('max_output: ' + str(np.max(tmp)))
                # print('min_output: ' + str(np.min(tmp)))
                # print('output: ' + str(tmp[0, :]))
                # self._train_writer.add_summary(summary, self._count)
                # summary_tmp = tf.Summary()
                # self._train_writer.add_summary(summary_tmp, self._count)
                # self._train_writer.flush()
                break

    @show_timing(text='validation', verbosity=2)
    def do_validation(self):
        """
        Perform validation on the current model
        """
        # TODO !!!!!

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

                data_all = list(zip(features_all, labels_all))

                # pick subset to fit onto the gpu
                sub_percentage = 0.5
                sub_indices = np.random.choice(len(data_all), int(sub_percentage * len(data_all)), replace=False)
                sub_data = [data_all[i] for i in sub_indices]

                features_all, labels_all = zip(*sub_data)

                # mi_test = sum_mi / self._count_mi
                # mi_vald = self._session.run(self._mutual_information, feed_dict={self._ph_train: False, self._ph_features:
                #     features_all, self._ph_labels: labels_all})
                sub_dict = dict((k, self._train_dict[Settings.identifier][k]) for k in ('count', 'accuracy_combination', 'mi')
                                if k in self._train_dict[Settings.identifier])
                return_dict = self._session.run(sub_dict,
                                                feed_dict={self._ph_train: False, self._ph_features: features_all,
                                                           self._ph_labels: labels_all,
                                                           self._ph_lr: Settings.learning_rate_pre,
                                                           self._ph_last_layer: False})

                summary_tmp = tf.Summary()
                if Settings.identifier == 'combination':
                    summary_tmp.value.add(tag='validation/mutual_information', simple_value=return_dict['mi'][0])
                    summary_tmp.value.add(tag='validation/accuracy', simple_value=return_dict['accuracy_combination'][0])
                elif Settings.identifier == 'vanilla':
                    summary_tmp.value.add(tag='validation/accuracy', simple_value=return_dict['accuracy'][0])
                elif Settings.identifier == 'nnvq':
                    summary_tmp.value.add(tag='validation/mutual_information', simple_value=return_dict['mi'][0])
                elif Settings.identifier == 'restore':
                    summary_tmp.value.add(tag='validation/accuracy', simple_value=return_dict['accuracy_combination'][0])

                self._train_writer.add_summary(summary_tmp, return_dict['count'])
                self._train_writer.flush()

                # TODO save current mi
                if return_dict['accuracy_combination'][0] > self._current_mi:
                    print('Saving better model...')
                    self._saver.save(self._session, Settings.path_checkpoint + '/saved_model')
                    self._current_mi = return_dict['accuracy_combination'][0]


                # return_dict = {}
                break

    def create_p_s_m(self):

        self._feeder.init_train()

        # set model.train to False to avoid training
        # model.train = False
        while True:
            try:
                feat, labs = self._session.run([self._input_train[0], self._input_train[1]])

                nom_vq, den_vq = self._session.run(self._train_dict[Settings.identifier]['data_vq'], feed_dict={self._ph_train: False, self._ph_features: feat,
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
