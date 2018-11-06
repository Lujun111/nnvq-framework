import tensorflow as tf


class Optimizer(object):
    def __init__(self, learning_rate, loss, opt_name='ADAM'):
        """
        Defining the optimizer which is used for training our model

        :param learning_rate:   learning rate for training
        :param loss:            loss for training
        :param opt_name:        define the optimizer which is applied
        """
        self._opt_name = opt_name
        self._learning_rate = learning_rate
        self._loss = loss
        self._optimizer = None

        # create optimizer
        self._set_optimizer()

    def get_train_op(self, global_step, clipping=True, clip_norm=0.75, var_list=None):
        """
        Get the train op for the optimizer

        :param global_step:     global step in training
        :param clipping:        flag for performing clipping
        :param clip_norm:       norm for clipping
        :param var_list:        list of weights which should be optimized
        :return:                train op
        """
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            if clipping:
                gradients, variables = zip(*self._optimizer.compute_gradients(self._loss, var_list=var_list))
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
                return self._optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
            else:
                gradients = self._optimizer.compute_gradients(self._loss, var_list=var_list)
                return self._optimizer.apply_gradients(gradients, global_step=global_step)

    def _set_optimizer(self):
        if self._opt_name == 'ADAM':
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        else:
            # TODO add other optimizer
            raise NotImplementedError('Not implemented')


