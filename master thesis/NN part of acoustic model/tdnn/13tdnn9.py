import kaldi_io
import numpy as np
import pandas as pd
import tensorflow as tf
import random

tf.reset_default_graph()

####hyperparameter
num_epochs = 12
left_context = 13
right_context = 9
time_series = left_context+1+right_context
learning_rate = 1e-3
features_dim = 39
labels_dim = 1
pdf_nums = 127


###################with subsampling (kaldi script [t-13,t+9])
global_step = tf.Variable(0, trainable=False)
training = tf.placeholder(tf.bool, name="is_train")
splice_features = tf.placeholder(tf.float32,shape=[None,features_dim*time_series], name='splice_features')
labels = tf.placeholder(tf.float32,shape=[None,labels_dim],name = 'labels')
input = tf.reshape(splice_features,shape=[-1,time_series,features_dim], name='input')

###tdnn1
tdnn1=tf.layers.conv1d(input,
                       filters=39,
                       kernel_size=5,
                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                       strides=3,
                       activation=tf.nn.relu,
                       name='tdnn1')
BN1=tf.layers.batch_normalization(tdnn1,training=training, name = 'BN1')

###tdnn2
tdnn2_left=BN1[:,:4,:]
tdnn2_right=BN1[:,3:,:]
tdnn2=tf.concat([tdnn2_left,tdnn2_right],axis=1)

tdnn2=tf.layers.conv1d(tdnn2,
                       filters=65,
                       kernel_size=2,
                       strides=2,
                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                       activation=tf.nn.relu,
                       name='tdnn2')
BN2=tf.layers.batch_normalization(tdnn2, training=training, name = 'BN2')


###tdnn3
tdnn3=tf.layers.conv1d(BN2,
                       filters=89,
                       kernel_size=2,
                       strides=2,
                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                       activation=tf.nn.relu,
                       name='tdnn3')
BN3=tf.layers.batch_normalization(tdnn3, training=training, name = 'BN3')


###tdnn4
tdnn4=tf.layers.conv1d(BN3,
                       filters=112,
                       kernel_size=2,
                       strides=1,
                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                       activation=tf.nn.relu,
                       name='tdnn4')
BN4=tf.layers.batch_normalization(tdnn4, training=training, name='BN4')

output = tf.layers.dense(BN4,units=127,name='fully-connected')

###log_posterior
log_post = tf.nn.log_softmax(output,dim=-1,name = 'log_posterior')
#########################################################

#########################################for train (paper [t-13,t+9])
with tf.name_scope("loss"):
    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), 127, axis=-1)
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
train_writer = tf.summary.FileWriter('logs/127tdnn12/train', sess.graph)
dev_writer = tf.summary.FileWriter('logs/127tdnn12/dev', sess.graph)

#############Splice features#########################
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

########################load counts file from kaldi##########
def load_counts(pdf_counts_file):
    with open(pdf_counts_file) as f:
        row = next(f).strip().strip('[]').strip()
        counts = np.array([ np.float32(v) for v in row.split() ])
        return counts

############################################################
step_dev = 0

nj_train_end = 32
nj_all = 35

context = (left_context,right_context)
splice = Splice(context)

for i in range(1,num_epochs+1):
    print ('Epoch: '+str(i)+' for training')

    for j in range(1, nj_train_end+1):
        features_train_path = '/home/ge69yif/kaldi/egs/tedlium/s5_r2/data/train/split35/'+str(j)+'/39mfcc.scp'
        pdfs_train_path = '/home/ge69yif/kaldi/egs/tedlium/s5_r2/exp/tri3/127dataset/127pdf.'+str(j)+'.ark'

        fea_train_reader = {k: m for k, m in kaldi_io.read_mat_scp(features_train_path)}
        pdf_train_reader = {k: v for k, v in kaldi_io.read_vec_int_ark(pdfs_train_path) if k in fea_train_reader}
        fea_train_reader = {k: m for k, m in fea_train_reader.items() if k in pdf_train_reader}

        fea_train_keys = [ key for key in fea_train_reader.keys() ]
        random.shuffle(fea_train_keys)
        for key in fea_train_keys:
            if key in pdf_train_reader.keys():
                pdf_vec = pdf_train_reader[key]
                fea = fea_train_reader[key]
                sentence_length = len(fea)
                mat_fea = splice(fea)
                mat_pdf = np.reshape(pdf_vec,[sentence_length,1])
                train_summary, train_optimizer, train_loss, train_accuracy, step = sess.run([merged_summary, train_op, loss, accuracy, global_step],
                                                                                    feed_dict={splice_features:mat_fea,labels:mat_pdf,training:True})
                train_writer.add_summary(train_summary, step)

                if train_accuracy > max_acc_train:
                    max_acc_train = train_accuracy
                    saver.save(sess,"models/127tdnn12/train/model.ckpt", global_step=step+1)

                if step % 100 == 0:
                    print('train_loss in step %s: %s' % (step, train_loss))
                    print('train_accuracy in step %s: %s' % (step,train_accuracy))


    print ('Epoch: '+str(i)+' training done!')

    print ('Epoch: '+str(i)+' for validation')

    for j in range(nj_train_end+1, nj_all+1):
        features_dev_path = '/home/ge69yif/kaldi/egs/tedlium/s5_r2/data/train/split35/'+str(j)+'/39mfcc.scp'
        pdfs_dev_path = '/home/ge69yif/kaldi/egs/tedlium/s5_r2/exp/tri3/127dataset/127pdf.'+str(j)+'.ark'

        fea_dev_reader = {k: m for k, m in kaldi_io.read_mat_scp(features_dev_path)}
        pdf_dev_reader = {k: v for k, v in kaldi_io.read_vec_int_ark(pdfs_dev_path) if k in fea_dev_reader}
        fea_dev_reader = {k: m for k, m in fea_dev_reader.items() if k in pdf_dev_reader}

        fea_dev_keys = [ key for key in fea_dev_reader.keys() ]
        random.shuffle(fea_dev_keys)
        for key in fea_dev_keys:
            if key in pdf_dev_reader.keys():
                pdf_vec = pdf_dev_reader[key]
                step_dev += 1
                fea = fea_dev_reader[key]
                sentence_length = len(fea)
                mat_fea = splice(fea)
                mat_pdf = np.reshape(pdf_vec,[sentence_length,1])
                dev_summary, dev_loss, dev_accuracy = sess.run([merged_summary, loss, accuracy ],feed_dict={splice_features:mat_fea,labels:mat_pdf,training:False})
                dev_writer.add_summary(dev_summary, step_dev)

                if dev_accuracy > max_acc_dev:
                    max_acc_dev = dev_accuracy
                    saver.save(sess,"models/127tdnn12/dev/model.ckpt")

                if step_dev % 100 == 0:
                    print('dev_loss in step_dev %s: %s' % (step_dev, dev_loss))
                    print('dev_accuracy in step_dev %s: %s' % (step_dev, dev_accuracy))

    print ('Epoch: '+str(i)+' validation done')

train_writer.close()
dev_writer.close()

######test data for decoding###########
nj = 30
test_path = '/home/ge69yif/kaldi/egs/tedlium/s5_r2/data/test/split30'
counts = load_counts('127_pdf.counts')
saver.restore(sess, tf.train.latest_checkpoint('models/127tdnn12/dev/'))
for i in range(1,nj+1):
    out_file = 'ark:| copy-feats --compress=true ark:- ark:'+test_path+'/'+str(i)+'/post.ark'
    post_file = kaldi_io.open_or_fd(out_file, 'wb')

    fea = {k: m for k, m in kaldi_io.read_mat_scp(test_path+'/'+str(i)+'/39mfcc.scp')}
    for key,fea in fea.items():
        sentence_length = len(fea)
        print (sentence_length)
        mat_fea = splice(fea)
        test_log_post = sess.run(log_post, feed_dict={splice_features: mat_fea,training:False})
        mat = np.squeeze(test_log_post-np.log(counts/np.sum(counts)))
        print (mat.shape)

        kaldi_io.write_mat(post_file,mat,key)
    post_file.close()
#######################################