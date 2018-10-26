import tensorflow as tf


class Optimizer(object):
    def __init__(self, learning_rate, loss, name='ADAM', control_dep=None):
        self._name = name
        self._learning_rate = learning_rate
        self._loss = loss
        self._optimizer = None
        self._control = control_dep

        # some methods
        self._set_optimizer()

    def get_train_op(self, var_list=None, global_step=None):
        with tf.control_dependencies(self._control):
            # gradients = self._optimizer.compute_gradients(self._loss, var_list=var_list)  # list_restore[5:]
            gradients, variables = zip(*self._optimizer.compute_gradients(self._loss, var_list=var_list))
            gradients, _ = tf.clip_by_global_norm(gradients, 0.25)
            # print(gradients)
            return self._optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
            # return self._optimizer.apply_gradients(gradients, global_step=global_step)

    def _set_optimizer(self):
        if self._name == 'ADAM':
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        else:
            # TODO add other optimizer
            raise NotImplementedError('Not implemented')


