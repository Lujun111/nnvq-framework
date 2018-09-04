import tensorflow as tf


class Optimizer(object):
    def __init__(self, settings, loss, global_step, name='ADAM'):
        self._name = name
        self._loss = loss
        self._learning_rate = settings.learning_rate
        self._global_step = global_step
        self._optimizer = None

    def train_op(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients = self._optimizer.compute_gradients(self._loss, var_list=None)  # list_restore[5:]
            train_op = self._optimizer.apply_gradients(gradients, global_step=self._global_step)
        return train_op

    def _set_optimizer(self):
        if self._name == 'ADAM':
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        else:
            # TODO add other optimizer
            raise NotImplementedError('Not implemented')


