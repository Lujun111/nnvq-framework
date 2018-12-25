import tensorflow as tf
import numpy as np


class Summary(object):
    def __init__(self, settings):
        self._settings = settings

    def train_logs(self, add_dict):
        with tf.variable_scope('SummaryHelper/train_logs'):
            summary_tmp = tf.Summary()

            if self._settings.identifier == 'combination':
                summary_tmp.value.add(tag='train/mutual_information', simple_value=add_dict['mi'][0])
                summary_tmp.value.add(tag='train/H(w)', simple_value=add_dict['mi'][1])
                summary_tmp.value.add(tag='train/H(y)', simple_value=add_dict['mi'][2])
                summary_tmp.value.add(tag='train/H(w|y)', simple_value=add_dict['mi'][3])
                summary_tmp.value.add(tag='train/acc_vanilla', simple_value=add_dict['accuracy_vanilla'][0])
                summary_tmp.value.add(tag='train/accuracy',
                                      simple_value=add_dict['accuracy_combination'][0])
            elif self._settings.identifier in ['nnvq', 'nnvq_tri']:
                # pass
                summary_tmp.value.add(tag='train/mutual_information', simple_value=add_dict['mi'][0])
                summary_tmp.value.add(tag='train/normalized_mutual_information',
                                      simple_value=(2 * add_dict['mi'][0]/(add_dict['mi'][1] + add_dict['mi'][2])))
                summary_tmp.value.add(tag='train/H(w)', simple_value=add_dict['mi'][1])
                summary_tmp.value.add(tag='train/H(y)', simple_value=add_dict['mi'][2])
                summary_tmp.value.add(tag='train/H(w|y)', simple_value=add_dict['mi'][3])
                # summary_tmp.value.add(tag='misc/learning_rate', simple_value=self._settings.learning_rate_pre)
                # summary_tmp.value.add(tag='misc/conditioned_entropy', simple_value=-np.sum(add_dict['joint_prob'] *
                #                                                                            np.log(add_dict['cond_prob'])))
            elif self._settings.identifier == 'vanilla':
                summary_tmp.value.add(tag='train/accuracy', simple_value=add_dict['accuracy'][0])
                # summary_tmp.value.add(tag='misc/learning_rate', simple_value=self._settings.learning_rate_pre)
            elif self._settings.identifier == 'restore':
                summary_tmp.value.add(tag='train/accuracy', simple_value=add_dict['accuracy_combination'][0])
                # summary_tmp.value.add(tag='misc/learning_rate', simple_value=self._settings.learning_rate_pre)
            elif self._settings.identifier == 'front':
                summary_tmp.value.add(tag='train/mutual_information', simple_value=add_dict['mi'][0])
                summary_tmp.value.add(tag='train/H(w)', simple_value=add_dict['mi'][1])
                summary_tmp.value.add(tag='train/H(y)', simple_value=add_dict['mi'][2])
                summary_tmp.value.add(tag='train/H(w|y)', simple_value=add_dict['mi'][3])
                # summary_tmp.value.add(tag='misc/learning_rate', simple_value=self._settings.learning_rate_pre)
                summary_tmp.value.add(tag='train/acc_vanilla', simple_value=add_dict['accuracy_vanilla'][0])
            summary_tmp.value.add(tag='misc/learning_rate', simple_value=self._settings.current_lr)
            summary_tmp.value.add(tag='train/loss', simple_value=add_dict['loss'])

            return summary_tmp

    def validation_logs(self, add_dict):
        with tf.variable_scope('SummaryHelper/validation_logs'):
            summary_tmp = tf.Summary()
            if self._settings.identifier == 'combination':
                summary_tmp.value.add(tag='validation/mutual_information', simple_value=add_dict['mi'][0])
                summary_tmp.value.add(tag='validation/accuracy', simple_value=add_dict['accuracy_combination'][0])
            elif self._settings.identifier == 'vanilla':
                summary_tmp.value.add(tag='validation/accuracy', simple_value=add_dict['accuracy'][0])
            elif self._settings.identifier == 'nnvq':
                summary_tmp.value.add(tag='validation/mutual_information', simple_value=add_dict['mi'][0])
                summary_tmp.value.add(tag='validation/normalized_mutual_information',
                                      simple_value=(2 * add_dict['mi'][0]/(add_dict['mi'][1] + add_dict['mi'][2])))
            elif self._settings.identifier == 'nnvq_tri':
                summary_tmp.value.add(tag='validation/mutual_information', simple_value=add_dict['mi'][0])
            elif self._settings.identifier == 'restore':
                summary_tmp.value.add(tag='validation/accuracy', simple_value=add_dict['accuracy_combination'][0])

            return summary_tmp
