"""
Setting file to define global variables
"""


# path to data
# path_train = 'scripts/tf_data'
path_train = 'tf_data/train_20k_state'
path_test = 'tf_data/test'
path_dev = 'tf_data/dev'
# path_train = 'tf_data/train_pdf'
# path_test = 'tf_data/test_pdf'
# path_dev = 'tf_data/dev_pdf'

path_tensorboard = 'tensorboard'
path_checkpoint = 'model_checkpoint'
path_restore = path_checkpoint + '/best_model_127pdf'
path_meta = 'meta_data'

# Hyperparameter
num_labels = 127
delta = 0.01    # delta for P(s_k|m_j)
codebook_size = 400
batch_size = 7500
epoch_size = 101
dim_features = 39
dim_labels = 1

# Network parameter
scale_soft = 20.0
learning_rate_pre = 1e-5
learning_rate_post = 1e-3
exponential_decay = False

# train Network
restore = False
train_prob = False
vqing = False
sampling_discrete = False
# create P(s_k|m_j) from training data
create_conditioned_prob = False
identifier = 'nnvq'     # possible identifiers: nnvq, nnvq_tri, vanilla, combination

# Inference
inference = False
