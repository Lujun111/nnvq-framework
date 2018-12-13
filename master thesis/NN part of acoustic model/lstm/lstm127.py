import kaldi_io
import numpy as np
import pandas as pd
import tensorflow as tf
import random

tf.reset_default_graph()

#hyperparameter
num_epochs = 12
time_series = 67 #=left_context+1+right_context
learning_rate = 1e-3
features_dim = 39
labels_dim = 20 #=chunk_width
cell_dim = 128
layers_num = 3
in_keep_prob = 1.0
out_keep_prob = 1.0
keep_prob = 1.0
momentum = 0.5
pdf_nums = 127

chunk_width = 20
chunk_left_context = 40
chunk_right_context = 0
model_left = 0
model_right = 7
label_delay = 5



######################LSTM#####################################################

global_step = tf.Variable(0, trainable=False)
training = tf.placeholder(tf.bool, name="is_train")
splice_features = tf.placeholder(tf.float32,shape=[None,features_dim*time_series], name='splice_features')
labels = tf.placeholder(tf.float32,shape=[None,labels_dim],name = 'labels')
input = tf.reshape(splice_features,shape=[-1,time_series,features_dim], name='input')

###tdnn1
tdnn1=tf.layers.conv1d(input,
                       filters=56,
                       kernel_size=5,
                       strides=1,
                       activation=tf.nn.relu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                       name='tdnn1')
BN1=tf.layers.batch_normalization(tdnn1,training=training,name='BN1')

def get_a_cell(lstm_size,keep_prob):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop


cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(cell_dim, keep_prob) for _ in range(layers_num)])

lstm_output, states = tf.nn.dynamic_rnn(cell=cell, inputs=BN1, dtype=tf.float32)

fc_output = tf.layers.dense(lstm_output, units=pdf_nums, name= 'fu_output')
output = fc_output[:,(-chunk_width):,:]

###log_posterior
log_post = tf.nn.log_softmax(fc_output[:,chunk_left_context,:],dim=-1,name = 'log_posterior')

#################################################################################

with tf.name_scope("loss"):
    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), pdf_nums, axis=-1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=onehot_labels))
    tf.summary.scalar('loss', loss)


with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(output, axis=-1), tf.argmax(onehot_labels, axis=-1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


learning_rate=tf.train.exponential_decay(
        learning_rate=learning_rate,
        global_step=global_step,
        decay_steps=10000,
        decay_rate=0.9)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate,name='optimizer')
    gradients = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(gradients, global_step=global_step)
###################################################################################

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
max_acc_train = 0
max_acc_dev = 0

merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('logs/127lstm12/train', sess.graph)
dev_writer = tf.summary.FileWriter('logs/127lstm12/dev', sess.graph)


##################load counts file from kaldi######################
def load_counts(pdf_counts_file):
    with open(pdf_counts_file) as f:
        row = next(f).strip().strip('[]').strip()
        counts = np.array([ np.float32(v) for v in row.split() ])
        return counts

##################Splice features####################################
class Splice(object):
    def __init__(self, context):
        self.left_context, self.right_context = context

    def __call__(self, x):
        T, D = x.shape
        context = (self.left_context + 1 + self.right_context)
        s = np.zeros((T, D * context), dtype=x.dtype)
        for t in range(T):
            for c in range(context):
                idx = min(max(t + c - self.left_context, 0), T - 1)
                s[t, c * D:c * D + D] = x[idx]
        return s

##################Chunk labels/pdfs####################################
class Chunk_pdf(object):
    def __init__(self, numofoutput):
        self.numofoutput  = numofoutput

    def __call__(self, x):
        T = len(x)
        x = np.array(x)
        x = x.reshape(T,1)
        numofoutput = self.numofoutput
        s = np.zeros((T, numofoutput), dtype=x.dtype)
        for t in range(T):
            for c in range(numofoutput):
                idx = min(t+c, T - 1)
                s[t,c:c+1] = x[idx]
        return s

#########################################################################
step_dev = 0

nj_train_end = 32
nj_all = 35

left_context = 40
right_context = 26
context = (left_context,right_context)
splice = Splice(context)

numofoutput = 20 ##same with 'chunk_width'
chunk_pdf = Chunk_pdf(numofoutput)

# for i in range(1,num_epochs+1):
#     print ('Epoch: '+str(i)+' for training')
#
#     for j in range(1, nj_train_end+1):
#         features_train_path = '/home/ge69yif/kaldi/egs/tedlium/s5_r2/data/train/split35/'+str(j)+'/39mfcc.scp'
#         pdfs_train_path = '/home/ge69yif/kaldi/egs/tedlium/s5_r2/exp/tri3/127dataset/127pdf.'+str(j)+'.ark'
#
#         fea_train_reader = {k: m for k, m in kaldi_io.read_mat_scp(features_train_path)}
#         pdf_train_reader = {k: v for k, v in kaldi_io.read_vec_int_ark(pdfs_train_path) if k in fea_train_reader}
#         fea_train_reader = {k: m for k, m in fea_train_reader.items() if k in pdf_train_reader}
#
#         fea_train_keys = [ key for key in fea_train_reader.keys() ]
#         random.shuffle(fea_train_keys)
#         for key in fea_train_keys:
#             if key in pdf_train_reader.keys():
#                 pdf_vec = pdf_train_reader[key]
#                 fea = fea_train_reader[key]
#                 sentence_length = len(fea)
#                 mat_fea = splice(fea)
#                 mat_pdf = chunk_pdf(pdf_vec)
#                 train_summary, train_optimizer, train_loss, train_accuracy, step = sess.run([merged_summary, train_op, loss, accuracy, global_step],
#                                                                                     feed_dict={splice_features:mat_fea,labels:mat_pdf,training:True})
#                 train_writer.add_summary(train_summary, step)
#
#                 if train_accuracy > max_acc_train:
#                     max_acc_train = train_accuracy
#                     saver.save(sess,"models/127lstm12/train/model.ckpt", global_step=step+1)
#
#                 if step % 100 == 0:
#                     print('train_loss in step %s: %s' % (step, train_loss))
#                     print('train_accuracy in step %s: %s' % (step,train_accuracy))
#
#
#     print ('Epoch: '+str(i)+' training done!')
#
#     print ('Epoch: '+str(i)+' for validation')
#
#     for j in range(nj_train_end+1, nj_all+1):
#         features_dev_path = '/home/ge69yif/kaldi/egs/tedlium/s5_r2/data/train/split35/'+str(j)+'/39mfcc.scp'
#         pdfs_dev_path = '/home/ge69yif/kaldi/egs/tedlium/s5_r2/exp/tri3/127dataset/127pdf.'+str(j)+'.ark'
#
#         fea_dev_reader = {k: m for k, m in kaldi_io.read_mat_scp(features_dev_path)}
#         pdf_dev_reader = {k: v for k, v in kaldi_io.read_vec_int_ark(pdfs_dev_path) if k in fea_dev_reader}
#         fea_dev_reader = {k: m for k, m in fea_dev_reader.items() if k in pdf_dev_reader}
#
#         fea_dev_keys = [ key for key in fea_dev_reader.keys() ]
#         random.shuffle(fea_dev_keys)
#         for key in fea_dev_keys:
#             if key in pdf_dev_reader.keys():
#                 pdf_vec = pdf_dev_reader[key]
#                 step_dev += 1
#                 fea = fea_dev_reader[key]
#                 sentence_length = len(fea)
#                 mat_fea = splice(fea)
#                 mat_pdf = chunk_pdf(pdf_vec)
#                 dev_summary, dev_loss, dev_accuracy = sess.run([merged_summary, loss, accuracy ],feed_dict={splice_features:mat_fea,labels:mat_pdf,training:False})
#                 dev_writer.add_summary(dev_summary, step_dev)
#
#                 if dev_accuracy > max_acc_train:
#                     max_acc_dev = dev_accuracy
#                     saver.save(sess,"models/127lstm12/dev/model.ckpt")
#
#                 if step_dev % 100 == 0:
#                     print('dev_loss in step_dev %s: %s' % (step_dev, dev_loss))
#                     print('dev_accuracy in step_dev %s: %s' % (step_dev, dev_accuracy))
#
#     print ('Epoch: '+str(i)+' validation done')
#
# train_writer.close()
# dev_writer.close()

######################test data for decoding#####################
nj = 30
path = '/home/ge69yif/kaldi/egs/tedlium/s5_r2/data/test/split30'
counts = load_counts('127_pdf.counts')
saver.restore(sess, tf.train.latest_checkpoint('models/127lstm12/train/'))
for i in range(1,nj+1):
    out_file = 'ark:| copy-feats --compress=true ark:- ark:'+path+'/'+str(i)+'/post.ark'
    post_file = kaldi_io.open_or_fd(out_file, 'wb')

    fea = {k: m for k, m in kaldi_io.read_mat_scp(path+'/'+str(i)+'/39mfcc.scp')}
    for key,fea in fea.items():
        sentence_length = len(fea)
        print(sentence_length)
        mat_fea = splice(fea)
        test_log_post = sess.run(log_post, feed_dict={splice_features: mat_fea, training: False})
        mat = test_log_post-np.log(counts/np.sum(counts))

        kaldi_io.write_mat(post_file,mat,key)
        print (mat.shape)
    post_file.close()
#########################################################################







