import tensorflow as tf
from NeuralNetHelper.TrainHelper import Train
from NeuralNetHelper import Settings
from NeuralNetHelper.MiscNNHelper import MiscNN
from NeuralNetHelper.ModelHelper import Model
from NeuralNetHelper.DataFeedingHelper import DataFeeder
from NeuralNetHelper.LossHelper import Loss
from NeuralNetHelper.OptimizerHelper import Optimizer
from KaldiHelper.InferenceHelper import InferenceModel

if __name__ == "__main__":

    # init all for training
    # session
    session = tf.Session()

    # data feeder
    data_feeder = DataFeeder(Settings, session)

    # placeholders
    placeholders = {
        'ph_train': tf.placeholder(tf.bool, name="is_train"),
        'ph_labels': tf.placeholder(tf.float32, shape=[None, Settings.dim_labels], name='ph_labels'),
        'ph_lr': tf.placeholder(tf.float32, shape=[], name='learning_rate'),
        'ph_features': tf.placeholder(tf.float32, shape=[None, Settings.dim_features], name='ph_features'),
        'ph_conditioned_probability': tf.placeholder(tf.float32, shape=None, name='ph_conditioned_probability'),
        'ph_last_layer': tf.placeholder(tf.bool, name="train_output")
    }

    # ph_train = tf.placeholder(tf.bool, name="is_train")
    # ph_labels = tf.placeholder(tf.float32, shape=[None, Settings.dim_labels], name='ph_labels')
    # ph_lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    # ph_features = tf.placeholder(tf.float32, shape=[None, Settings.dim_features], name='ph_features')
    # ph_conditioned_probability = tf.placeholder(tf.float32, shape=None, name='ph_conditioned_probability')
    # ph_last_layer = tf.placeholder(tf.bool, name="train_output")

    # variables
    variables = {
        'global_step': tf.Variable(0, trainable=False),
        'nominator': tf.Variable(tf.zeros([Settings.num_labels, Settings.codebook_size]), trainable=False,
                                 dtype=tf.float32),
        'denominator': tf.Variable(tf.zeros([Settings.codebook_size]), trainable=False, dtype=tf.float32)
    }

    # global_step = tf.Variable(0, trainable=False)
    # nominator = tf.Variable(tf.zeros([Settings.num_labels, Settings.codebook_size]), trainable=False,
    #                         dtype=tf.float32)
    # denominator = tf.Variable(tf.zeros([Settings.codebook_size]), trainable=False, dtype=tf.float32)

    # auxiliary functions
    misc = MiscNN(Settings)

    # model
    p_s_m = misc.set_probabilities('model_checkpoint/vq_graph/p_s_m.mat')
    model = Model(placeholders['ph_train'], placeholders['ph_features'], Settings, input_tensor=p_s_m,
                  ph_output=placeholders['ph_last_layer'])

    # saver
    saver = tf.train.Saver()

    train_model = Train()
    # train_model.restore_model('model_checkpoint')

    if not Settings.inference:
        print('Training model...')
        for i in range(Settings.epoch_size):
            print('Epoch ' + str(i))

            # print('Training base network')
            # train_model.train_single_epoch(identifier=Settings.identifier)

            print('Training front...')
            Settings.identifier = 'front'
            train_model.train_single_epoch(train_bn=True)

            print('Training comb...')
            Settings.identifier = 'restore'
            train_model.train_single_epoch(train_bn=False)

            # if i % 10 == 0:
            #     print('Creating P(s_k|m_j)...')
            #     train_model.create_p_s_m()
            #
            # # print('Training output layer')
            # # train_model.train_single_epoch(train_last_layer=True)
            #
            print('Doing Validation...')
            train_model.do_validation()

    else:
        print('Doing inference...')
        train_model.do_inference()
