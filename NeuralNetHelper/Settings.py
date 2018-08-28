"""
Setting file to define global variables
"""


# path to data
# path_train = 'scripts/tf_data'
path_train = 'tf_data/train_20k_state'
path_test = 'tf_data/test_state'
path_dev = 'tf_data/dev_state'

path_tensorboard = 'tensorboard'
path_checkpoint = 'model_checkpoint'
path_meta = 'meta_data'

# Hyperparameter
num_labels = 127
delta = 0.01    # delta for P(s_k|m_j)
codebook_size = 400
batch_size = 15000
epoch_size = 100
dim_features = 39
dim_labels = 1

# Network parameter
scale_soft = 20.0
learning_rate = 1e-5
exponential_decay = False

# train Network
vqing = False
sampling_discrete = False
# create P(s_k|m_j) from training data
create_conditioned_prob = False
