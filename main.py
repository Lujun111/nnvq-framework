#!/home/ga96yar/tensorflow_py3/bin/python
import tensorflow as tf
import sys
import argparse
from tensorflow.python import debug as tf_debug
import pandas as pd
from NeuralNetHelper.TrainHelper import Train
from NeuralNetHelper.Settings import Settings
from NeuralNetHelper.MiscNNHelper import MiscNN
from NeuralNetHelper.ModelHelper import Model
from NeuralNetHelper.DataFeedingHelper import DataFeeder
from NeuralNetHelper.LossHelper import Loss
from NeuralNetHelper.OptimizerHelper import Optimizer
from NeuralNetHelper.SaverHelper import Saver
from KaldiHelper.InferenceHelper import InferenceModel


def train():

    # TODO refactor
    # Save the current setting for the experiment
    with open(Settings.path_checkpoint + '/settings.txt', 'w') as file:
        for key, value in sorted(Settings.__dict__.items()):
            if '__' not in key:
                file.write(key + ': ' + str(value) + '\n')

    # placeholders
    placeholders = {
        'ph_train': tf.placeholder(tf.bool, name="is_train"),
        'ph_labels': tf.placeholder(tf.float32, shape=[None, Settings.dim_labels], name='ph_labels'),
        # 'ph_lr': tf.placeholder(tf.float32, shape=[], name='learning_rate'),
        'ph_features': tf.placeholder(tf.float32, shape=[None, Settings.dim_features], name='ph_features'),
        'ph_conditioned_probability': tf.placeholder(tf.float32, shape=[Settings.codebook_size, Settings.num_labels],
                                                     name='ph_conditioned_probability'),
        'ph_last_layer': tf.placeholder(tf.bool, name="train_output"),
    }

    # variables
    variables = {
        'global_step': tf.Variable(0, trainable=False),
        'nominator': tf.Variable(tf.zeros([Settings.num_labels, Settings.codebook_size]), trainable=False,
                                 dtype=tf.float32, name='var_nominator'),
        'denominator': tf.Variable(tf.zeros([Settings.codebook_size]), trainable=False, dtype=tf.float32,
                                   name='var_denominator'),
        'conditioned_probability': tf.Variable(tf.fill([Settings.num_labels, Settings.codebook_size],
                                                       1.0 / Settings.num_labels), trainable=False, dtype=tf.float32,
                                               name='var_cond_prob'),
        'p_w': tf.Variable(tf.zeros(Settings.num_labels), trainable=False, dtype=tf.float32, name='p_w'),
        'p_y': tf.Variable(tf.zeros(Settings.codebook_size), trainable=False, dtype=tf.float32, name='p_y'),
        'p_w_y': tf.Variable(tf.zeros([Settings.num_labels, Settings.codebook_size]), trainable=False,
                             dtype=tf.float32, name='p_w_y'),
        'epoch': tf.Variable(0, trainable=False),
        'learning_rate': tf.Variable(Settings.learning_rate, trainable=False)
    }

    # auxiliary functions
    misc = MiscNN(Settings)

    # model
    model = Model(placeholders['ph_train'], placeholders['ph_features'], Settings,
                  ph_output=placeholders['ph_last_layer'])

    loss = Loss(model, placeholders['ph_labels'], Settings)

    variables['learning_rate'] = tf.train.exponential_decay(Settings.learning_rate, variables['epoch'],
                                               Settings.lr_epoch_decrease, Settings.lr_decay, staircase=True)

    optimizer = Optimizer(variables['learning_rate'], loss.loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "tueimmk-apo6:7000")
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        data_feeder = DataFeeder(Settings, sess)
        saver = Saver(Settings, sess)
        train_model = Train(sess, Settings, model, misc, optimizer, loss, data_feeder, saver, placeholders, variables)
        # train_model.restore_model('model_checkpoint')
        tf.get_default_graph().finalize()
        if not Settings.inference:
            print('Training model...')
            for i in range(Settings.epoch_size):
                print('Epoch ' + str(i))

                # train_model.create_p_s_m()

                print('Training base network')
                train_model.train_single_epoch()

                # if i % 10 == 0:
                #     print('Creating P(s_k|m_j)...')
                #     train_model.create_p_s_m()

                print('Doing Validation...')
                train_model.do_validation()

        else:
            print('Doing inference...')
            train_model.do_inference()


def str2bool(v):
    """
    Converts string argument to bool
    :param v:
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(arguments):
    """
    Create argument parser to execute python file from console
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # add codebook size
    parser.add_argument('--cb_size', type=int, help='size of clusters', default=400)
    # parser.add_argument('--cmvn', type=str2bool, help='flag for cmvn or global normalization',
    #                     default=True)
    # parse all arguments to parser

    parser.add_argument('--tensorboard', type=str, help='path to tensorboard', default='tensorboard')
    parser.add_argument('--checkpoint', type=str, help='path to model checkpoint',
                        default='model_checkpoint')
    parser.add_argument('--path_train', type=str, help='path to train data',
                        default='tf_data/splice_1f/train_pdf_20k_splice_1f_cmn')
    parser.add_argument('--path_test', type=str, help='path to test data',
                        default='tf_data/splice_1f/test_pdf_20k_splice_1f_cmn')
    parser.add_argument('--path_dev', type=str, help='path to dev data',
                        default='tf_data/splice_1f/dev_pdf_20k_splice_1f_cmn')
    parser.add_argument('--dim_features', type=int, help='dim input features', default=39)
    parser.add_argument('--dropout', type=float, help='dropout value', default=0.1)
    parser.add_argument('--lr_epoch_decrease', type=int, help='dropout value', default=3)
    args = parser.parse_args(arguments)

    # set new field values TODO best solution?
    Settings.codebook_size = args.cb_size
    Settings.path_checkpoint = args.checkpoint
    Settings.path_tensorboard = args.tensorboard
    Settings.path_train = args.path_train
    Settings.path_test = args.path_test
    Settings.path_dev = args.path_dev
    Settings.dim_features = args.dim_features
    Settings.do_rate = args.dropout
    Settings.lr_epoch_decrease = args.lr_epoch_decrease

    # print the arguments which we fed into
    for arg in vars(args):
        print("Argument {:14}: {}".format(arg, getattr(args, arg)))

    # perform task
    train()

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
