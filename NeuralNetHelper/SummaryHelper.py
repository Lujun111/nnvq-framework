import tensorflow as tf


class Summary(object):
    def __init__(self, settings):
        self._settings = settings

    def train_logs(self, add_dict):
        summary_tmp = tf.Summary()

        if self._settings.identifier == 'combination':
            summary_tmp.value.add(tag='train/mutual_information', simple_value=add_dict['mi'][0])
            summary_tmp.value.add(tag='train/H(w)', simple_value=add_dict['mi'][1])
            summary_tmp.value.add(tag='train/H(y)', simple_value=add_dict['mi'][2])
            summary_tmp.value.add(tag='train/H(w|y)', simple_value=add_dict['mi'][3])
            summary_tmp.value.add(tag='misc/learning_rate', simple_value=self._settings.learning_rate_pre)
            summary_tmp.value.add(tag='train/acc_vanilla', simple_value=add_dict['accuracy_vanilla'][0])
            summary_tmp.value.add(tag='train/accuracy',
                                  simple_value=add_dict['accuracy_combination'][0])
        elif self._settings.identifier == 'nnvq':
            summary_tmp.value.add(tag='train/mutual_information', simple_value=add_dict['mi'][0])
            summary_tmp.value.add(tag='train/H(w)', simple_value=add_dict['mi'][1])
            summary_tmp.value.add(tag='train/H(y)', simple_value=add_dict['mi'][2])
            summary_tmp.value.add(tag='train/H(w|y)', simple_value=add_dict['mi'][3])
            summary_tmp.value.add(tag='misc/learning_rate', simple_value=self._settings.learning_rate_pre)
        elif self._settings.identifier == 'vanilla':
            summary_tmp.value.add(tag='train/accuracy', simple_value=add_dict['accuracy'][0])
            summary_tmp.value.add(tag='misc/learning_rate', simple_value=self._settings.learning_rate_pre)
        elif self._settings.identifier == 'restore':
            summary_tmp.value.add(tag='train/accuracy', simple_value=add_dict['accuracy_combination'][0])
            summary_tmp.value.add(tag='misc/learning_rate', simple_value=self._settings.learning_rate_pre)
        elif self._settings.identifier == 'front':
            summary_tmp.value.add(tag='train/mutual_information', simple_value=add_dict['mi'][0])
            summary_tmp.value.add(tag='train/H(w)', simple_value=add_dict['mi'][1])
            summary_tmp.value.add(tag='train/H(y)', simple_value=add_dict['mi'][2])
            summary_tmp.value.add(tag='train/H(w|y)', simple_value=add_dict['mi'][3])
            summary_tmp.value.add(tag='misc/learning_rate', simple_value=self._settings.learning_rate_pre)
            summary_tmp.value.add(tag='train/acc_vanilla', simple_value=add_dict['accuracy_vanilla'][0])
        summary_tmp.value.add(tag='train/loss', simple_value=add_dict['loss'])

        return summary_tmp

    def validation_logs(self, add_dict):
        summary_tmp = tf.Summary()
        if self._settings.identifier == 'combination':
            summary_tmp.value.add(tag='validation/mutual_information', simple_value=add_dict['mi'][0])
            summary_tmp.value.add(tag='validation/accuracy', simple_value=add_dict['accuracy_combination'][0])
        elif self._settings.identifier == 'vanilla':
            summary_tmp.value.add(tag='validation/accuracy', simple_value=add_dict['accuracy'][0])
        elif self._settings.identifier == 'nnvq':
            summary_tmp.value.add(tag='validation/mutual_information', simple_value=add_dict['mi'][0])
        elif self._settings.identifier == 'restore':
            summary_tmp.value.add(tag='validation/accuracy', simple_value=add_dict['accuracy_combination'][0])

        return summary_tmp
