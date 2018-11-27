import tensorflow as tf


class Saver(object):
    def __init__(self, settings, session):
        self._settings = settings
        self._saver = tf.train.Saver()
        self._session = session
        self._current_value = -100.0

    def save(self, save_dict):
        with tf.variable_scope('SaverHelper/save'):
            if self._settings.identifier == 'vanilla':
                if save_dict['accuracy'][0] > self._current_value:
                    print('Saving better model...')
                    self._saver.save(self._session, self._settings.path_checkpoint + '/saved_model')
                    self._current_value = save_dict['accuracy'][0]
            elif self._settings.identifier == 'combination':
                if save_dict['accuracy_combination'][0] > self._current_value:
                    print('Saving better model...')
                    self._saver.save(self._session, self._settings.path_checkpoint + '/saved_model')
                    self._current_value = save_dict['accuracy_combination'][0]
            elif self._settings.identifier in ['nnvq', 'nnvq_tri']:
                if save_dict['mi'][0] > self._current_value:
                    print('Saving better model...')
                    self._saver.save(self._session, self._settings.path_checkpoint + '/saved_model')
                    self._current_value = save_dict['mi'][0]
