import tensorflow as tf
import pandas as pd
from tensorflow.python import debug as tf_debug
import os
import time
import numpy as np
from kaldi_io import kaldi_io
from NeuralNetHelper.MiscNN import MiscNN
from NeuralNetHelper.ModelHelper import Model
from NeuralNetHelper import Settings
from NeuralNetHelper.DataFeedingHelper import DataFeeder


# defining some operations
global_step = tf.Variable(0, trainable=False)
nom_var = tf.Variable(tf.zeros([Settings.num_labels, Settings.codebook_size]), trainable=False, dtype=tf.float32)
den_var = tf.Variable(tf.zeros([Settings.codebook_size]), trainable=False, dtype=tf.float32)
nom_init = nom_var.assign(tf.fill([Settings.num_labels, Settings.codebook_size], 0.0))
den_init = den_var.assign(tf.fill([Settings.codebook_size], 0.0))

# create iterator, select tfrecord-files for training, 3 feeding pipelines
file_list_train = [Settings.path_train + '/' + s for s in os.listdir(Settings.path_train)]
file_list_test = [Settings.path_test + '/' + s for s in os.listdir(Settings.path_test)]
file_list_dev = [Settings.path_dev + '/' + s for s in os.listdir(Settings.path_dev)]

train_feeder = DataFeeder(file_list_train, Settings.batch_size, Settings.dim_features, Settings.dim_labels)
test_feeder = DataFeeder(file_list_test, Settings.batch_size, Settings.dim_features, Settings.dim_labels)
dev_feeder = DataFeeder(file_list_dev, Settings.batch_size, Settings.dim_features, Settings.dim_labels)

features_train, labels_train = train_feeder.iterator.get_next()
features_test, labels_test = test_feeder.iterator.get_next()
features_dev, labels_dev = dev_feeder.iterator.get_next()

# placeholders for model and for later inference
# training = tf.placeholder(tf.bool, shape=[], name='ph_training')
features = tf.placeholder(tf.float32, shape=[None, Settings.dim_features], name='ph_features')
labels = tf.placeholder(tf.float32, shape=[None, Settings.dim_labels], name='ph_labels')
lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')

lr_decay = tf.train.exponential_decay(Settings.learning_rate, global_step, 400, 0.90, staircase=True)

log_prob = None
if Settings.vqing:
    log_prob = tf.placeholder(tf.float32, shape=[Settings.num_phonemes, Settings.codebook_size], name='prob')

# define model
model = Model(features, Settings.scale_soft, Settings.codebook_size)
misc = MiscNN(Settings.codebook_size, Settings.num_labels)
# model_train = tf.estimator.Estimator(model_fn=model.inference)
# y = model.inference(features)
y = model.inference
# y = tf.Print(y, [tf.reduce_sum(tf.cast(tf.is_nan(y), dtype=tf.int32))])

# y = tf.Print(y, [tf.shape(y)])
# model_train.train(features)
# vqing data of the whole dataset
data_vqed = misc.vq_data(y, labels, nom_var, den_var)


# mutual information
mutual_info = misc.calculate_mi_tf(y, tf.cast(labels, dtype=tf.int32))

cond_prob = misc.conditioned_probability(y, labels, discrete=Settings.sampling_discrete)
testing = misc.testing_stuff(y, cond_prob, labels)

# loss
# regularizer = tf.contrib.layers.l1_regularizer(scale=1e-3)
# weights = tf.trainable_variables()
# reg_term = tf.contrib.layers.apply_regularization(regularizer, weights)
# loss = model.new_loss(labels, testing)
# loss = model.loss(labels, log_prob)
loss = None
if Settings.vqing:
    # loss = model.loss(labels, log_prob)
    loss = model.loss(labels, cond_prob)
else:
    loss = model.loss(labels, cond_prob)
    # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    # loss += 1e-5 * l2_loss
# print(loss)
# loss += reg_term
# loss += 1e-1 * mutual_info[2]
tf.summary.scalar('train/loss', loss)

joint_probability = misc.joint_probability(y, labels)
# joint_probability = tf.Print(joint_probability, [tf.reduce_sum(tf.cast(tf.is_nan(joint_probability), dtype=tf.int32))])
conditioned_entropy = -tf.reduce_sum(joint_probability * tf.log(cond_prob))
tf.summary.scalar('misc/conditioned_entropy', conditioned_entropy)

# TODO get batch_norm running
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=0.0)
    gradients = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(gradients, global_step=global_step)



# init all values
init = tf.global_variables_initializer()
# init_local = tf.local_variables_initializer()
sess = tf.Session()
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "tueimmk-apo6:7000")
sess.run(init)

# merge all summaries
merged = tf.summary.merge_all()
time_string = time.strftime('%d.%m.%Y - %H:%M:%S')
train_writer = tf.summary.FileWriter(Settings.path_tensorboard + '/training_' + time_string, sess.graph)

# create saver
saver = tf.train.Saver()


# variables_names = [v.name for v in tf.trainable_variables()]
# values = sess.run(variables_names)
# for k, v in zip(variables_names, values):
#     print("Variable: ", k)
#     print("Shape: ", v.shape)
#     print(v)

# optimizer.compute_gradients(zip(variables_names, values))
current_mi = -10.0
count_mi = 0
prob = None
nom_vq = den_vq = None

# train
for i in range(Settings.epoch_size):
    print("Epoch " + str(i))
    sess.run(train_feeder.iterator.initializer)
    sess.run(test_feeder.iterator.initializer)
    sess.run(dev_feeder.iterator.initializer)

    if Settings.vqing:
        # 1. step: vq data
        print('Vqing data...')
        # set model.train to False to avoid training
        # model.train = False
        while True:
            try:
                feat, labs = sess.run([features_train, labels_train])

                # print(feat)

                nom_vq, den_vq = sess.run(data_vqed, feed_dict={"is_train:0": False, features: feat,
                                                                         labels: labs, lr: Settings.learning_rate})

            except tf.errors.OutOfRangeError:
                nom_vq += Settings.delta
                den_vq += Settings.num_labels * Settings.delta
                prob = nom_vq / den_vq

                # reset den and nom
                sess.run([nom_init, den_init])

                print(prob)
                break

        # 2. step: traing model with statistics of step 1
        # resetting the iterator
        sess.run(train_feeder.iterator.initializer)

    print('Doing single epoch...')
    while True:
        try:
            feat, labs, new_lr = sess.run([features_train, labels_train, lr_decay])

            # check for exponential decayed learning rate and set it
            if Settings.exponential_decay:
                Settings.learning_rate = new_lr

            if Settings.vqing:
                _, loss_value, summary, count, mi, y_print, test = sess.run(
                    [train_op, loss, merged, global_step, mutual_info, y, testing],
                    feed_dict={"is_train:0": True, features: feat, labels: labs, lr: Settings.learning_rate,
                               log_prob: prob})
            else:
                _, loss_value, summary, count, mi, y_print, test = sess.run(
                    [train_op, loss, merged, global_step, mutual_info, y, testing],
                    feed_dict={"is_train:0": True, features: feat, labels: labs, lr: Settings.learning_rate})
            # print(kernel)
            # print(feat)
            # print(labs)
            # print(sess.run("BatchNorm/AssignMovingAvg_1:0"))
            if count % 100:
                train_writer.add_summary(summary, count)
                summary_tmp = tf.Summary()
                summary_tmp.value.add(tag='train/mutual_information', simple_value=mi[0])
                summary_tmp.value.add(tag='train/H(w)', simple_value=mi[1])
                summary_tmp.value.add(tag='train/H(y)', simple_value=mi[2])
                summary_tmp.value.add(tag='train/H(w|y)', simple_value=mi[3])
                summary_tmp.value.add(tag='misc/learning_rate', simple_value=Settings.learning_rate)
                train_writer.add_summary(summary_tmp, count)
                train_writer.flush()

        except tf.errors.OutOfRangeError:
            # print(nom_vq/den_vq)
            print('loss: ' + str(loss_value))
            print('max: ' + str(np.max(y_print)))
            print('min: ' + str(np.min(y_print)))
            train_writer.add_summary(summary, count)
            summary_tmp = tf.Summary()
            train_writer.add_summary(summary_tmp, count)
            train_writer.flush()
            break

    # 3. step: Check the error on the dev-set
    sum_mi = 0.0
    features_all = []
    labels_all = []
    print('Doing validation...')
    # model.train = False
    while True:
        try:
            feat, labs = sess.run([features_dev, labels_dev])
            features_all.append(feat)
            labels_all.append(labs)

        except tf.errors.OutOfRangeError:
            # reshape data
            features_all = np.concatenate(features_all)
            labels_all = np.concatenate(labels_all)

            # mi_test = sum_mi / count_mi
            mi_vald, test_py, test_pw, test_pyw = sess.run([mutual_info, "p_y:0", "p_w:0", "p_yw:0"],
                                                           feed_dict={"is_train:0": False, features: features_all,
                                                                            labels: labels_all})
            print(mi_vald)
            summary_tmp = tf.Summary()
            summary_tmp.value.add(tag='validation/mutual_information', simple_value=mi_vald[0])
            train_writer.add_summary(summary_tmp, count)
            train_writer.flush()

            # print(count_mi)
            if mi_vald[0] > current_mi and mi_vald[2] > 6.50:
                print('Saving better model...')
                saver.save(sess, Settings.path_checkpoint + '/saved_model', global_step=i)
                current_mi = mi_vald[0]
                # learning_rate *= 1e-1
            # else:
            #     count_mi += 1
            #     # print('Count MI: ' + str(count_mi))
            #     if count_mi > 2:
            #         learning_rate *= 1e-1
            #         print('Reducing learning rate to ' + str(learning_rate))
            #         count_mi = 0

            # save counts of
            tmp_pywtest = pd.DataFrame(test_py)
            tmp_pywtest.to_csv(Settings.path_meta + '/py_nnvq.txt', header=False, index=False)
            tmp_pywtest = pd.DataFrame(test_pw)
            tmp_pywtest.to_csv(Settings.path_meta + '/pw_nnvq.txt', header=False, index=False)
            tmp_pywtest = pd.DataFrame(test_pyw)
            tmp_pywtest.to_csv(Settings.path_meta + '/pwy_nnvq.txt', header=False, index=False)

            break

    if Settings.create_conditioned_prob:
        # resetting the iterator
        sess.run(train_feeder.iterator.initializer)
        print('Create P(s_k|m_j) from training data...')
        while True:
            try:
                feat, labs = sess.run([features_train, labels_train])
                nom_vq, den_vq = sess.run(data_vqed, feed_dict={"is_train:0": False, features: feat,
                                                                labels: labs, lr: Settings.learning_rate})

            except tf.errors.OutOfRangeError:
                nom_vq += Settings.delta
                den_vq += Settings.num_phonemes * Settings.delta
                prob = nom_vq / den_vq

                # saving matrix with kaldi_io
                save_dict = {'p_s_m': prob}
                print('Saving P(s_k|m_j)')
                with open('p_s_m.mat', 'wb') as f:
                    for key, mat in list(save_dict.items()):
                        kaldi_io.write_mat(f, mat, key=key)

                # reset den and nom
                sess.run([nom_init, den_init])

                break

print("Training done")
