# -*- coding: utf-8 -*
import tensorflow as tf
import pandas as pd
from tensorflow.python import debug as tf_debug
import os
import time
import numpy as np
from kaldi_io import kaldi_io
from NeuralNetHelper import Settings
from NeuralNetHelper.DataFeedingHelper import DataFeeder
from NeuralNetHelper.SummaryHelper import Summary
from KaldiHelper.InferenceHelper import InferenceModel


class Train(object):
    """
    This class manages everything
    """
    # TODO interface for Management object?
    def __init__(self, session, settings, model, misc, optimizer, loss, data_feeder, saver, placeholders, variables):
        self._session = session
        self._settings = settings
        self._model = model
        self._misc = misc
        self._loss = loss
        self._optimizer = optimizer
        self._placeholders = placeholders
        self._variables = variables
        self._saver = saver
        self._summary = Summary(settings)

        self._feeder = data_feeder
        self._input_train = data_feeder.train.get_next()
        self._input_test = data_feeder.test.get_next()
        self._input_dev = data_feeder.dev.get_next()

        self._current_value = -100.0
        self._count_reduce_lr = 0

        self._increment_epoch_op = tf.assign_add(self._variables['epoch'], 1)
        self._mult_lr_op = tf.multiply(self._variables['learning_rate'], Settings.lr_decay)

        self._create_train_dict()

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

    def _init_saver(self):
        # list_restore = [v for v in tf.trainable_variables()]
        # print(list_restore[:6])
        self._saver = tf.train.Saver()

    def _init_data_feeder(self):

        # define feeder
        self._feeder = DataFeeder(Settings, self._session)

        # create 3 pipelines for features
        self._input_train = self._feeder.train.get_next()
        self._input_test = self._feeder.test.get_next()
        self._input_dev = self._feeder.dev.get_next()

    def _global_init(self):
        self._init_data_feeder()
        self._init_saver()

    def _create_train_dict(self):
        # we create different dicts depending on the task
        with tf.variable_scope('TrainHelper/create_train_dict'):

            if self._settings.identifier == 'nnvq':
                self._train_dict = {
                    'mi': self._misc.calculate_mi_tf(self._model.inference, self._placeholders['ph_labels']),
                    # 'joint_prob': self._misc.joint_probability(self._model.inference, self._placeholders['ph_labels']),
                    'cond_prob': self._variables['conditioned_probability'],
                    # 'data_vq': self._misc.vq_data(self._model.inference, self._placeholders['ph_labels'],
                    #                               self._variables['nominator'], self._variables['denominator']),
                    'loss': self._loss.loss,
                    'train_op': self._optimizer.get_train_op(global_step=self._variables['global_step'], clipping=False),
                    'output': self._model.inference,
                    'count': self._variables['global_step'],
                    'learning_rate': self._variables['learning_rate']}
                self._val_dict = {
                    'stats': self._misc.create_stats_val(tf.argmax(self._model.inference, axis=1),
                                                         self._placeholders['ph_labels'])}

            elif self._settings.identifier == 'nnvq_tri':
                self._train_dict = {
                    'mi': self._misc.calculate_mi_tf(self._model.inference, self._placeholders['ph_labels']),
                    'joint_prob': self._misc.joint_probability(self._model.inference, self._placeholders['ph_labels']),
                    # 'cond_prob': self._misc.conditioned_probability(self._model.inference, self._placeholders['ph_labels'],
                    #                                                 discrete=Settings.sampling_discrete),
                    'data_vq': self._misc.vq_data(self._model.inference, self._placeholders['ph_labels'],
                                                  self._variables['nominator'], self._variables['denominator']),
                    'loss': self._loss.loss,
                    'train_op': self._optimizer.get_train_op(global_step=self._variables['global_step']),
                    'output': self._model.inference,
                    'count': self._variables['global_step']}

                # self._variables['conditioned_probability'.ass

            elif self._settings.identifier == 'vanilla':
                self._train_dict = {
                    'loss': self._loss.loss,
                    'train_op': self._optimizer.get_train_op(global_step=self._variables['global_step']),
                    'accuracy': tf.metrics.accuracy(self._placeholders['ph_labels'],
                                                    tf.argmax(self._model.inference, axis=1)),
                    'output': self._model.inference,
                    'count': self._variables['global_step']}

            elif self._settings.identifier == 'combination':
                self._train_dict = {
                    'mi': self._misc.calculate_mi_tf(self._model.inference_nnvq, self._placeholders['ph_labels']),
                    'loss': self._loss.loss,
                    'accuracy_vanilla': tf.metrics.accuracy(self._placeholders['ph_labels'],
                                                                            tf.argmax(self._model.inference_vanilla, axis=1)),
                    'accuracy_combination': tf.metrics.accuracy(self._placeholders['ph_labels'],
                                                                            tf.argmax(self._model.inference_combination, axis=1)),
                    'output': self._model.inference_combination,
                    'count': self._variables['global_step'],
                    'data_vq': self._misc.vq_data(self._model.inference_nnvq, self._placeholders['ph_labels'], self._variables['nominator'],
                                                  self._variables['denominator']),
                    'train_op': self._optimizer.get_train_op(global_step=self._variables['global_step'])}

        self._session.run(tf.local_variables_initializer())
        self._session.run(tf.global_variables_initializer())

        time_string = time.strftime('%d.%m.%Y - %H:%M:%S')
        self._train_writer = tf.summary.FileWriter(Settings.path_tensorboard + '/training_' + time_string, tf.get_default_graph())

    @show_timing(text='train', verbosity=2)
    def train_single_epoch(self, cond_prob=None):
        """
        Train single epoch


        :return:
        """
        self._feeder.init_train()

        while True:
            try:
                feat, labs = self._session.run([self._input_train[0], self._input_train[1]])
                # print(np.max(labs))
                # variables_names = [v.name for v in tf.trainable_variables()]
                # values = self._session.run(variables_names)
                # for k, v in zip(variables_names, values):
                #     print("Variable: ", k)
                #     print("Shape: ", v.shape)
                #     print(v)
                # exit()

                return_dict = self._session.run(self._train_dict,
                                                feed_dict={self._placeholders['ph_train']: True,
                                                           self._placeholders['ph_features']: feat,
                                                           self._placeholders['ph_labels']: labs,
                                                           self._placeholders['ph_last_layer']: True})
                # self._settings.current_lr = self._session.run(self._variables['learning_rate'])
                # print(np.max(return_dict['output']))
                # print(feat)
                # print(np.sum(return_dict['cond_prob'], axis=0))
                if return_dict['count'] % 10 == 0:
                    # summary_tmp.value.add(tag='misc/alpha', simple_value=return_dict['alpha'])
                    self._train_writer.add_summary(self._summary.train_logs(return_dict), return_dict['count'])
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
        output_all = []
        labels_all = []
        p_w = np.zeros([Settings.num_labels])
        p_y = np.zeros([Settings.codebook_size])
        p_w_y = np.zeros([Settings.num_labels, Settings.codebook_size])
        p_w.fill(1e-10)
        p_y.fill(1e-10)
        p_w_y.fill(1e-10)

        while True:
            try:
                feat, labs = self._session.run([self._input_dev[0], self._input_dev[1]])

                # output_nn = self._session.run(self._train_dict['output'],
                #                               feed_dict={self._placeholders['ph_train']: False,
                #                               self._placeholders['ph_features']: feat,
                #                               self._placeholders['ph_labels']: labs,
                #                               self._placeholders['ph_last_layer']: False})
                stats = self._session.run(self._val_dict['stats'],
                                          feed_dict={self._placeholders['ph_train']: False,
                                          self._placeholders['ph_features']: feat,
                                          self._placeholders['ph_labels']: labs,
                                          self._placeholders['ph_last_layer']: False})

                # y_tmp = np.argmax(output_nn, axis=1).astype(np.int32)
                # p_w[labs.astype(np.int32)] += 1.0
                # p_y[y_tmp] += 1.0
                # p_w_y[labs.astype(np.int32), y_tmp] += 1.0

                p_w += stats[0]
                p_y += stats[1]
                p_w_y += stats[2]

                # output_all.append(np.argmax(output_nn, axis=1))
                # output_all.append(feat)
                # labels_all.append(labs)

            except tf.errors.OutOfRangeError:
                # reshape data
                # output_all = np.concatenate(output_all)
                # labels_all = np.concatenate(labels_all)

                p_w /= np.sum(p_w)
                p_y /= np.sum(p_y)
                p_w_y /= np.expand_dims(np.sum(p_w_y, axis=1), 1)

                # H(Y) on log2 base
                h_y = p_y * np.log(p_y) / np.log(2.0)
                h_y = np.sum(h_y)
                # print(h_y)

                # H(W) on log2 base
                h_w = p_w * np.log(p_w) / np.log(2.0)
                h_w = np.sum(h_w)
                # print(h_w)

                # H(W|Y) on log2 base
                h_w_y = p_w_y * np.log(p_w_y) / np.log(2.0)
                h_w_y = np.sum(h_w_y, axis=0)
                h_w_y = p_y * h_w_y
                h_w_y = np.sum(h_w_y)
                # print(h_w_y)

                val_dict = {
                    'count': self._train_dict['count']
                }
                return_dict = self._session.run(val_dict)
                return_dict['mi'] = [-h_w + h_w_y, -h_w, -h_y, -h_w_y]
                #
                # data_all = list(zip(output_all, labels_all))
                #
                # # pick subset to fit onto the gpu
                # sub_percentage = 1.0
                # sub_indices = np.random.choice(len(data_all), int(sub_percentage * len(data_all)), replace=False)
                # sub_data = [data_all[i] for i in sub_indices]
                #
                # features_all, labels_all = zip(*sub_data)
                #
                # # mi_test = sum_mi / self._count_mi
                # # mi_vald = self._session.run(self._mutual_information, feed_dict={self._ph_train: False, self._ph_features:
                # #     features_all, self._ph_labels: labels_all})
                # sub_dict = dict((k, self._train_dict[k])
                #                 for k in ('count', 'accuracy_combination', 'mi', 'accuracy')
                #                 if k in self._train_dict)
                # return_dict = self._session.run(sub_dict,
                #                                 feed_dict={self._placeholders['ph_train']: False,
                #                                            self._placeholders['ph_features']: features_all,
                #                                            self._placeholders['ph_labels']: labels_all,
                #                                            self._placeholders['ph_last_layer']: False})

                self._train_writer.add_summary(self._summary.validation_logs(return_dict), return_dict['count'])
                self._train_writer.flush()

                # save depending on model
                self._saver.save(return_dict)

                # # check for reducing learning rate
                # if self._saver.current_count > Settings.lr_epoch_decrease:
                #     print('Validation not getting better for ' + str(Settings.lr_epoch_decrease) +
                #           ' Epochs')
                #     print('Reducing learning rate by ' + str(Settings.lr_decay))
                #     self._session.run(self._mult_lr_op)
                #     self._saver.current_count = 0

                # increase epoch counter
                _, self._settings.current_lr = self._session.run([self._increment_epoch_op,
                                                                  self._variables['learning_rate']])


                break

    @show_timing(text='p_s_m', verbosity=2)
    def create_p_s_m(self):

        self._feeder.init_train()

        # set model.train to False to avoid training
        # model.train = False
        while True:
            try:
                feat, labs = self._session.run([self._input_train[0], self._input_train[1]])

                nom_vq, den_vq = self._session.run(self._train_dict['data_vq'], feed_dict={self._placeholders['ph_train']: False,
                                                                                self._placeholders['ph_features']: feat,
                                                                                self._placeholders['ph_labels']: labs})

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

                # reset den and nom, set variable
                self._session.run([self._misc.reset_variable(self._variables['nominator']),
                                   self._misc.reset_variable(self._variables['denominator']),
                                   tf.assign(self._variables['conditioned_probability'], prob)])

                break

    def do_inference(self, stats_file='stats_20k.mat', cond_prob_file='p_s_m.mat', transform_prob=True, log_output=True):
        inference = InferenceModel(self._session, stats_file,
                                   cond_prob_file, transform_prob=transform_prob, log_output=log_output)
        # model_continuous.do_inference(20, 'features/train_20k/feats', '../tmp/tmp_testing')
        inference.do_inference(30, 'test', 'tmp/tmp_testing')
