import tensorflow as tf
import numpy as np
from kaldi_io import kaldi_io


class MiscNN(object):
    def __init__(self, codebook_size):
        self.cb_size = codebook_size

    def calculate_mi_tf(self, y_nn, phonemes):
        alpha = 1.0
        beta = -1.0

        # get p_w, p_y and p_w_y from helper
        p_w, p_y, p_w_y = self._helper_mi_tf(y_nn, phonemes)


        # normalize
        p_w /= tf.reduce_sum(p_w)
        p_y /= tf.reduce_sum(p_y)
        p_w_y = tf.divide(p_w_y,
                          tf.expand_dims(tf.clip_by_value(tf.reduce_sum(p_w_y, axis=1),
                                                          1e-8, 1e6), 1))
        # # H(Y) on log2 base
        h_y = tf.multiply(p_y, tf.log(tf.clip_by_value(p_y, 1e-8, 1e6)) / tf.log(2.0))
        h_y = tf.reduce_sum(h_y)

        # H(W) on log2 base
        h_w = tf.multiply(p_w, tf.log(tf.clip_by_value(p_w, 1e-8, 1e6)) / tf.log(2.0))
        h_w = tf.reduce_sum(h_w)

        # H(W|Y) on log2 base
        h_w_y = p_w_y * tf.log(tf.clip_by_value(p_w_y, 1e-12, 1e6)) / tf.log(2.0)  # log2 base
        h_w_y = tf.reduce_sum(h_w_y, axis=0)  # reduce sum
        h_w_y = tf.multiply(p_y, h_w_y)
        h_w_y = tf.reduce_sum(h_w_y)

        return -alpha * h_w - beta * h_w_y, -h_w, -h_y, -h_w_y

    def _helper_mi_tf(self, labels, alignments):
        p = 41

        pwtmp = tf.Variable(tf.zeros(p), trainable=False, dtype=tf.float32)
        pytmp = tf.Variable(tf.zeros(self.cb_size), trainable=False, dtype=tf.float32)
        pw_y_tmp = tf.Variable(tf.zeros([p, self.cb_size]), trainable=False, dtype=tf.float32)

        # use input array as indexing array
        pwtmp = pwtmp.assign(tf.fill([p], 0.0))  # reset Variable/floor
        pwtmp = tf.scatter_add(pwtmp, alignments, tf.ones(tf.shape(alignments)))

        pytmp = pytmp.assign(tf.fill([self.cb_size], 0.0))  # reset Variable/floor
        pytmp = tf.scatter_add(pytmp, labels, tf.ones(tf.shape(labels)))

        pw_y_tmp = pw_y_tmp.assign(tf.fill([p, self.cb_size], 0.0))  # reset Variable/floor
        pw_y_tmp = tf.scatter_nd_add(pw_y_tmp,
                                     tf.concat([tf.cast(alignments, dtype=tf.int64), tf.expand_dims(labels, 1)],
                                               axis=1), tf.ones(tf.shape(labels)))
        # adding to graph
        tf.identity(pwtmp, 'p_w')
        tf.identity(pytmp, 'p_y')
        tf.identity(pw_y_tmp, 'p_yw')

        return pwtmp, pytmp, pw_y_tmp

    def conditioned_probability(self, output_nn, phonemes, discrete=False):
        p = 41  # num of phones
        eps = 0.5

        # get data
        phonemes = tf.cast(phonemes, dtype=tf.int32)  # cast to int and put them in [[alignments]]

        # create var for joint probability
        nominator = tf.Variable(tf.zeros([p, self.cb_size]), trainable=False, dtype=tf.float32, name='nom_test')
        nominator = nominator.assign(tf.fill([p, self.cb_size], eps))  # reset Variable/floor
        denominator = tf.Variable(tf.zeros([self.cb_size]), trainable=False, dtype=tf.float32, name='den_test')
        denominator = denominator.assign(tf.fill([self.cb_size], p * eps))  # reset Variable/floor

        output_dis = None
        if discrete:
            output_dis = tf.argmax(output_nn, axis=1)

        if discrete:
            nominator = tf.scatter_nd_add(nominator, tf.concat([tf.cast(phonemes, dtype=tf.int64),
                                                                tf.expand_dims(output_dis, 1)],
                                                               axis=1), tf.ones(tf.shape(output_dis)))
            denominator = tf.scatter_add(denominator, output_dis, tf.ones(tf.shape(output_dis)[0]))
            # nominator = tf.Print(nominator, [tf.shape(nominator)])
            # denominator = tf.Print(denominator, [denominator], summarize=400, message='t1')
            # denominator_tmp = tf.reduce_sum(output_nn, axis=0)
            # denominator = tf.Print(denominator, [denominator], summarize=400, message='t2')

        else:
            nominator = tf.scatter_nd_add(nominator, phonemes, output_nn)
            denominator = tf.reduce_sum(output_nn, axis=0)
            denominator += p * eps

        if discrete:
            conditioned_prob = tf.div(nominator, tf.expand_dims(denominator, 0))
        else:
            conditioned_prob = tf.divide(nominator, denominator)
        # conditioned_prob = tf.Print(conditioned_prob, [tf.reduce_max(conditioned_prob)])

        return conditioned_prob

    def joint_probability(self, output_nn, phonemes):
        # j is size of codebook
        p = 41  # num of phones
        batch_size = tf.cast(tf.shape(output_nn)[0], dtype=tf.float32)  # input size

        # create var for joint probability
        joint_prob = tf.Variable(tf.zeros([p, self.cb_size]), trainable=False, dtype=tf.float32)
        joint_prob = joint_prob.assign(tf.fill([p, self.cb_size], 0.0))  # reset Variable/floor


        # get data
        phonemes = tf.cast(phonemes, dtype=tf.int32)  # cast to int and put them in [[alignments]]

        # add to sclices
        joint_prob = tf.scatter_nd_add(joint_prob, phonemes, output_nn)

        joint_prob = tf.div(joint_prob, batch_size)

        return joint_prob

    def vq_data(self, output_nn, phonemes, nominator, denominator):
        p = 41  # num of phones
        eps = 0.5

        # get data
        phonemes = tf.cast(phonemes, dtype=tf.int64)  # cast to int and put them in [[alignments]]

        # create var for joint probability

        #   # reset Variable/floor
        # con_prob = tf.divide(con_prob, tf.add(p * eps, tf.reduce_sum(input, axis=0)))

        labels_softmax = tf.argmax(output_nn, axis=1)
        # labels_softmax = output_soft

        nominator = tf.scatter_nd_add(nominator, tf.concat([tf.cast(phonemes, dtype=tf.int64),
                                                            tf.expand_dims(labels_softmax, 1)],
                                                           axis=1), tf.ones(tf.shape(labels_softmax)))
        # nominator = tf.scatter_nd_add(nominator, alignments, labels_softmax)


        # denominator = denominator.assign(tf.fill([cb], 0.0))  # reset Variable/floor
        denominator = tf.scatter_add(denominator, labels_softmax, tf.ones(tf.shape(labels_softmax)[0]))
        #
        # denominator = tf.reduce_sum(labels_softmax, axis=0)
        # denominator += p * eps

        # conditioned_prob = tf.divide(nominator, denominator)

        return nominator, denominator

    def testing_stuff(self, output_nn, cond_prob, phonemes):
        # labels_softmax = output_nn
        phonemes = tf.cast(phonemes, dtype=tf.int32)
        cond_prob = tf.transpose(cond_prob)
        # cond_prob = tf.Print(cond_prob, [cond_prob])

        s_k = tf.Variable(tf.zeros([41]), trainable=False, dtype=tf.float32)
        s_k.assign(tf.fill([41], 0.0))

        out_nn = tf.Variable(tf.zeros([400]), trainable=False, dtype=tf.float32)
        out_nn.assign(tf.fill([400], 0.0))

        labels_softmax = tf.argmax(output_nn, axis=1)

        s_k = tf.scatter_add(s_k, phonemes, tf.ones(tf.shape(phonemes)))
        s_k = tf.clip_by_value(s_k, 1e-15, 1e6)
        s_k /= tf.reduce_sum(s_k)

        # out_nn = tf.scatter_add(out_nn, labels_softmax, tf.ones(tf.shape(phonemes)[0]))
        # out_nn = tf.clip_by_value(out_nn, 1e-15, 1e6)
        # out_nn /= tf.reduce_sum(out_nn)

        tmp_out = tf.reduce_sum(output_nn, axis=0)
        out_nn = tmp_out / tf.reduce_sum(tmp_out)

        # out_nn = tf.reduce_sum(output_nn, axis=0) / tf.cast(tf.shape(output_nn)[0], dtype=tf.float32)

        joint = tf.tile(tf.expand_dims(out_nn, 1), [1, 41]) * cond_prob

        tmp = joint / tf.reduce_sum(joint, axis=1, keepdims=True)

        return tmp



