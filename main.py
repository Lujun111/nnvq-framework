import tensorflow as tf
from tensorflow.python import debug as tf_debug
from NeuralNetHelper.TrainHelper import Train
from NeuralNetHelper import Settings
from NeuralNetHelper.MiscNNHelper import MiscNN
from NeuralNetHelper.ModelHelper import Model
from NeuralNetHelper.DataFeedingHelper import DataFeeder
from NeuralNetHelper.LossHelper import Loss
from NeuralNetHelper.OptimizerHelper import Optimizer
from NeuralNetHelper.SaverHelper import Saver
from KaldiHelper.InferenceHelper import InferenceModel

if __name__ == "__main__":
    # init all for training

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
