import tensorflow as tf
from kaldi_io import kaldi_io
import numpy as np


class MiscNN(object):
    """
    MiscNN object contains auxiliary functions which are needed for
    logging and to compute functions in the graph
    """
    def __init__(self, settings):
        """
        Init MiscNN

        :param codebook_size: size of the codebook
        :param num_labels:    number of labels (e.g. for monophone states=127)
        """
        self.cb_size = settings.codebook_size
        self.num_labels = settings.num_labels
        # self.p_w = tf.Variable(tf.zeros(settings.num_labels), trainable=False, dtype=tf.float32, name='p_w'),
        # self.p_y = tf.Variable(tf.zeros(settings.codebook_size), trainable=False, dtype=tf.float32, name='p_y'),
        # self.p_w_y = tf.Variable(tf.zeros([settings.num_labels, settings.codebook_size]), trainable=False,
        #                          dtype=tf.float32, name='p_w_y')
        # self.reset_p_w = tf.assign(self.p_w, tf.zeros([self.num_labels]))
        # self.reset_p_y = tf.assign(self.p_y, tf.zeros([self.cb_size]))
        # self.reset_p_w_y = tf.assign(self.p_w_y, tf.zeros([self.num_labels, self.cb_size]))

    def calculate_mi_tf(self, y_nn, labels, nn_output=True):
        """
        Calculate the mutual information between the neural net output and
        the labels (phonemes/states)

        :param y_nn:        output of the neural net
        :param labels:      phonemes or states of the data (coming from the alignments of kaldi)
        :param nn_output:   calculate MI with with own labels
        :return:            mutual information
        """
        with tf.variable_scope('MiscNNHelper/calculate_mi_tf'):
            labels = tf.cast(labels, dtype=tf.int32)

            # take the argmax of y_nn to get the class label determined by the
            # neural network
            if nn_output:
                y_labels = tf.argmax(y_nn, axis=1)
            else:
                y_labels = y_nn
            # y_labels = tf.Print(y_labels, [y_labels], summarize=400)

            # get p_w, p_y and p_w_y from helper
            p_w, p_y, p_w_y = self.helper_mi_tf(y_labels, labels)

            # H(Y) on log2 base
            h_y = tf.multiply(p_y, tf.log(tf.clip_by_value(p_y, 1e-8, 1e6)) / tf.log(2.0))
            h_y = tf.reduce_sum(h_y)

            # H(W) on log2 base
            h_w = tf.multiply(p_w, tf.log(tf.clip_by_value(p_w, 1e-8, 1e6)) / tf.log(2.0))
            h_w = tf.reduce_sum(h_w)

            # H(W|Y) on log2 base
            h_w_y = p_w_y * tf.log(tf.clip_by_value(p_w_y, 1e-12, 1e6)) / tf.log(2.0)
            h_w_y = tf.reduce_sum(h_w_y, axis=0)
            h_w_y = tf.multiply(p_y, h_w_y)
            h_w_y = tf.reduce_sum(h_w_y)

            return -h_w + h_w_y, -h_w, -h_y, -h_w_y

    def helper_mi_tf(self, y_labels, labels):
        """
        Create P(w), P(y) and P(w|y) using the output labels of the neural network
        For deeper understanding how we create these probability, please check the
        TF-API

        :param y_labels:                output labels of the neural net
        :param labels:                  phonemes or states of the data (coming from the alignments of kaldi)
        :return pwtmp, pytmp, pw_y_tmp: return P(w), P(y) and P(w|y)
        """
        with tf.variable_scope('MiscNNHelper/helper_mi_tf'):
            # define tf variables to use scatter_nd and scatter_nd_add
            # pwtmp = tf.Variable(tf.zeros(self.num_labels), trainable=False, dtype=tf.float32)
            # pytmp = tf.Variable(tf.zeros(self.cb_size), trainable=False, dtype=tf.float32)
            # pw_y_tmp = tf.Variable(tf.zeros([self.num_labels, self.cb_size]), trainable=False, dtype=tf.float32)
            pwtmp = tf.get_default_graph().get_tensor_by_name('p_w:0')
            pytmp = tf.get_default_graph().get_tensor_by_name('p_y:0')
            pw_y_tmp = tf.get_default_graph().get_tensor_by_name('p_w_y:0')

            # self.reset_p_w


            # create P(w)
            pwtmp = tf.assign(pwtmp, tf.zeros([self.num_labels]))  # reset Variable/floor
            # pwtmp = self.reset_variable(pwtmp)
            pwtmp = tf.scatter_add(pwtmp, labels, tf.ones(tf.shape(labels)))

            # create P(y)
            pytmp = tf.assign(pytmp, tf.zeros([self.cb_size]))  # reset Variable/floor
            pytmp = tf.scatter_add(pytmp, y_labels, tf.ones(tf.shape(y_labels)))

            # create P(w|y)
            pw_y_tmp = tf.assign(pw_y_tmp, tf.zeros([self.num_labels, self.cb_size]))  # reset Variable/floor
            pw_y_tmp = tf.scatter_nd_add(pw_y_tmp,
                                         tf.concat([tf.cast(labels, dtype=tf.int64), tf.expand_dims(y_labels, 1)],
                                                   axis=1), tf.ones(tf.shape(y_labels)))

            # adding to graph for visualisation in tensorboard
            # tf.identity(pwtmp, 'P_w')
            # tf.identity(pytmp, 'P_y')
            # tf.identity(pw_y_tmp, 'P_w_y')

            # normalize
            pwtmp = tf.divide(pwtmp, tf.reduce_sum(pwtmp))
            pytmp = tf.divide(pytmp, tf.reduce_sum(pytmp))
            pw_y_tmp = tf.divide(pw_y_tmp, tf.expand_dims(tf.clip_by_value(tf.reduce_sum(pw_y_tmp, axis=1), 1e-8, 1e6), 1))

            return pwtmp, pytmp, pw_y_tmp

    def create_stats_val(self, y_labels, labels):

        with tf.variable_scope('MiscNNHelper/create_stats_val'):
            labels = tf.cast(labels, dtype=tf.int32)
            # define tf variables to use scatter_nd and scatter_nd_add
            # pwtmp = tf.Variable(tf.zeros(self.num_labels), trainable=False, dtype=tf.float32)
            # pytmp = tf.Variable(tf.zeros(self.cb_size), trainable=False, dtype=tf.float32)
            # pw_y_tmp = tf.Variable(tf.zeros([self.num_labels, self.cb_size]), trainable=False, dtype=tf.float32)
            pwtmp = tf.get_default_graph().get_tensor_by_name('p_w:0')
            pytmp = tf.get_default_graph().get_tensor_by_name('p_y:0')
            pw_y_tmp = tf.get_default_graph().get_tensor_by_name('p_w_y:0')

            # self.reset_p_w


            # create P(w)
            pwtmp = tf.assign(pwtmp, tf.zeros([self.num_labels]))  # reset Variable/floor
            # pwtmp = self.reset_variable(pwtmp)
            pwtmp = tf.scatter_add(pwtmp, labels, tf.ones(tf.shape(labels)))

            # create P(y)
            pytmp = tf.assign(pytmp, tf.zeros([self.cb_size]))  # reset Variable/floor
            pytmp = tf.scatter_add(pytmp, y_labels, tf.ones(tf.shape(y_labels)))

            # create P(w|y)
            pw_y_tmp = tf.assign(pw_y_tmp, tf.zeros([self.num_labels, self.cb_size]))  # reset Variable/floor
            pw_y_tmp = tf.scatter_nd_add(pw_y_tmp,
                                         tf.concat([tf.cast(labels, dtype=tf.int64), tf.expand_dims(y_labels, 1)],
                                                   axis=1), tf.ones(tf.shape(y_labels)))

            return pwtmp, pytmp, pw_y_tmp

    def conditioned_probability(self, y_nn, labels, discrete=False, conditioned='m_j'):
        """
        Create the conditioned probability P(s_k|m_j) using the output of the neural network
        and the target labels coming out of kaldi

        :param y_nn:        output of the neural net
        :param labels:      phonemes or states of the data (coming from the alignments of kaldi)
        :param discrete:    flag for creating P(s_k|m_j) in the discrete way (check dissertation
                            of Neukirchen, p.62 (5.51))
        :param conditioned: condition on 'm_j' or 'y_k'
        :return:            return P(s_k|m_j)
        """
        with tf.variable_scope('MiscNNHelper/conditioned_probability'):
            # small delta for smoothing P(s_k|m_j), necessary if we log the probability
            eps = tf.constant(1e-2)  # 1e-2 (mono) 1e-4 (tri)

            # cast labels into int32
            labels = tf.cast(labels, dtype=tf.int32)  # cast to int and put them in [[alignments]]

            # create variable in order to use scatter_nd_add and scatter_add (discrete creation of P(s_k|m_j))
            # nominator = tf.Variable(tf.zeros([self.num_labels, self.cb_size]), trainable=False, dtype=tf.float32, name='nom_test')
            nominator = tf.get_default_graph().get_tensor_by_name('var_nominator:0')
            nominator = tf.assign(nominator, tf.fill([self.num_labels, self.cb_size], eps))  # reset Variable/floor
            # denominator = tf.Variable(tf.zeros([self.cb_size]), trainable=False, dtype=tf.float32, name='den_test')
            denominator = tf.get_default_graph().get_tensor_by_name('var_denominator:0')
            tf.assign(denominator, tf.fill([self.cb_size], self.num_labels * eps))  # reset Variable/floor

            if discrete:
                # discretize the output of the neural network
                output_dis = tf.argmax(y_nn, axis=1)

                # get nominator
                tf.scatter_nd_add(nominator, tf.concat([tf.cast(labels, dtype=tf.int64),
                                                                    tf.expand_dims(output_dis, 1)],
                                                                   axis=1), tf.ones(tf.shape(output_dis)))
                # get denominator
                tf.scatter_add(denominator, output_dis, tf.ones(tf.shape(output_dis)[0]))

                if conditioned == 'y_k':
                    # create P(m_j|y_k) in the discrete way
                    conditioned_prob = tf.div(nominator, tf.expand_dims(denominator, 0))
                elif conditioned == 'm_j':
                    # create P(y_k|m_j) in the discrete way
                    conditioned_prob = tf.div(nominator, tf.expand_dims(denominator, 0))
                else:
                    raise NotImplementedError

            else:
                # get nominator
                nominator = tf.scatter_nd_add(nominator, labels, y_nn)
                # nominator = tf.Print(nominator, [nominator[1, 1]])

                if conditioned == 'y_k':
                    # get denominator
                    denominator = tf.reduce_sum(y_nn, axis=1)
                    # smoothing the probability
                    denominator += self.num_labels * eps
                    # create P(m_j|y_k) in the discrete way
                    # conditioned_prob = tf.divide(nominator, denominator)
                    conditioned_prob = tf.div(nominator, tf.reduce_sum(nominator, axis=1, keepdims=True))
                elif conditioned == 'm_j':
                    # get denominator
                    denominator = tf.reduce_sum(y_nn, axis=0)
                    # smoothing the probability
                    denominator += self.num_labels * eps
                    # create P(y_k|m_j) in the discrete way
                    # conditioned_prob = tf.divide(nominator, denominator)
                    conditioned_prob = tf.div(nominator, tf.reduce_sum(nominator, axis=0, keepdims=True))
                else:
                    raise NotImplementedError
                #
                # # create P(s_k|m_j) in the continuous way
                # conditioned_prob = tf.divide(nominator, denominator)
            # nominator = tf.scatter_nd_add(nominator, labels, y_nn)
            # nominator += eps
            # conditioned_prob = tf.div(nominator, tf.reduce_sum(nominator, axis=1, keepdims=True))
            # nominator += eps
            # denominator = tf.reduce_sum(y_nn, axis=1)
            # # smoothing the probability
            # denominator += self.num_labels * eps
            # create P(m_j|y_k) in the discrete way
            # conditioned_prob = tf.divide(nominator, denominator)

            return conditioned_prob

    def joint_probability(self, y_nn, labels):
        # TODO I don't know if it works properly because I only use it for logging
        """
        Create joint probability P(s_k, m_j)

        :param y_nn:    output of the neural net
        :param labels:  phonemes or states of the data (coming from the alignments of kaldi)
        :return:        return P(s_k, m_j)
        """
        with tf.variable_scope('MiscNNHelper/joint_probability'):
            # determine batch size
            batch_size = tf.cast(tf.shape(y_nn)[0], dtype=tf.float32)

            # create variable in order to use scatter_nd_add
            joint_prob = tf.Variable(tf.zeros([self.num_labels, self.cb_size]), trainable=False, dtype=tf.float32)
            joint_prob = joint_prob.assign(tf.fill([self.num_labels, self.cb_size], 0.0))  # reset Variable/floor

            # cast labels to int32
            labels = tf.cast(labels, dtype=tf.int32)

            # create P(s_k, m_j), (check dissertation of Neukirchen, p.61 (5.46))
            joint_prob = tf.scatter_nd_add(joint_prob, labels, y_nn)
            joint_prob = tf.div(joint_prob, batch_size)

            return joint_prob

    def vq_data(self, y_nn, labels, nominator, denominator, discrete=True):
        """
        Create the nominator and denominator for P(s_k|m_j)
        This function is used for using all the training data to create P(s_k|m_j)

        :param y_nn:        output of the neural net
        :param labels:      phonemes or states of the data (coming from the alignments of kaldi)
        :param nominator:   nominator for P(s_k|m_j)
        :param denominator: denominator for P(s_k|m_j)
        :param discrete:    flag for creating P(s_k|m_j) in a discrete way
        :return:            return nominator and denominator of creating P(s_k|m_j)
        """
        with tf.variable_scope('MiscNNHelper/vq_data'):
            # cast labels to int32
            labels = tf.cast(labels, dtype=tf.int64)

            y_labels = tf.argmax(y_nn, axis=1)
            # labels_softmax = output_soft

            if discrete:
                # create nominator
                nominator = tf.scatter_nd_add(nominator, tf.concat([tf.cast(labels, dtype=tf.int64),
                                                                    tf.expand_dims(y_labels, 1)],
                                                                   axis=1), tf.ones(tf.shape(y_labels)))

                # create dominator
                denominator = tf.scatter_add(denominator, y_labels, tf.ones(tf.shape(y_labels)[0]))
            else:
                raise NotImplementedError("Not implemented!")

            return nominator, denominator

    @staticmethod
    def reset_variable(variable):
        with tf.variable_scope('MiscNNHelper/reset_variable'):
            return variable.assign(tf.fill(tf.shape(variable), 0.0))

    def reset_mi_variables(self):
        tf.assign(self.p_w, tf.zeros([self.num_labels]))
        tf.assign(self.p_y, tf.zeros([self.cb_size]))
        tf.assign(self.p_w_y, tf.zeros([self.num_labels, self.cb_size]))

    def testing_stuff(self, output_nn, cond_prob, phonemes):
        """
        deprecated!

        :param output_nn:
        :param cond_prob:
        :param phonemes:
        :return:
        """
        # labels_softmax = output_nn
        phonemes = tf.cast(phonemes, dtype=tf.int32)
        cond_prob = tf.transpose(cond_prob)
        # cond_prob = tf.Print(cond_prob, [cond_prob])

        s_k = tf.Variable(tf.zeros([127]), trainable=False, dtype=tf.float32)
        s_k.assign(tf.fill([127], 0.0))

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

        joint = tf.tile(tf.expand_dims(out_nn, 1), [1, 127]) * cond_prob

        tmp = joint / tf.reduce_sum(joint, axis=1, keepdims=True)

        return tmp

    def set_probabilities(self, file):
        # TODO hard coded for getting class counts --> make sure that file class.counts exists
        # TODO and contains the key class_counts
        """
        Set P(s_k|m_j) and prior P(s_k) from training

        :param file: path to P(s_k|m_j) file (kaldi-format, must contain the key 'p_s_m')
        """
        # Set P(s_k|m_j)
        for key, mat in kaldi_io.read_mat_ark(file):
            if key == 'p_s_m':
                print('Setting P(s_k|m_j)')
                return tf.convert_to_tensor(np.transpose(mat))  # we transpose for later dot product
                # print(np.sum(self.cond_prob, axis=1))
                # print(np.shape(np.sum(self.cond_prob, axis=1)))
            else:
                print('No probability found')
                return None

    def label_smoothing(self, labels, epsilon=0.1):
        """
        We implement the label smoothing

        :param epsilon: variable for smoothing (default 0.1)
        :param labels:  labels for smoothing (one-hot-encoded)
        :return:        smoothed labels
        """
        with tf.variable_scope('MiscNNHelper/label_smoothing'):
            # determine the dimension of the one hot for u(k)
            k = tf.cast(tf.shape(labels)[1], dtype=tf.float32)

            # create new labels
            smoothed_labels = (1.0 - epsilon) * labels + epsilon / k

            return smoothed_labels
