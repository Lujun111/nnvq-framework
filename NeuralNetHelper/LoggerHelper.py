import tensorflow as tf

class Logger(object):
    def __init__(self, identifier=None):
        self._identifier = identifier

    def create_logs(self):

        if self._identifier == 'nnvq':
            tf.summary.scalar('train/loss', self._train_dict['nnvq']['loss'])
            tf.summary.scalar('train/mutual_information', self._train_dict['nnvq']['mi'][0])
            tf.summary.scalar('train/H(w)', self._train_dict['nnvq']['mi'][1])
            # tf.summary.scalar('train/H(y)', self._train_dict['nnvq']['mi'][2])
            # tf.summary.scalar('train/H(w|y)', self._train_dict['nnvq']['mi'][3])