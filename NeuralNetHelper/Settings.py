"""
Setting file to define global variables
"""


# path to data
# path_train = 'scripts/tf_data'
# path_train = 'tf_data/train_20k_state'
# path_test = 'tf_data/test'
# path_dev = 'tf_data/dev'
path_train = 'tf_data/train_pdf_20k_splice'
path_test = 'tf_data/test_pdf_20k_splice'
path_dev = 'tf_data/dev_pdf_20k_splice'

path_tensorboard = 'tensorboard'
path_checkpoint = 'model_checkpoint'
path_restore = path_checkpoint + '/best_model_127pdf'
path_meta = 'meta_data'

# Hyperparameter
num_labels = 127   # 4131(tri3)    # 2026 (tri1)    # 127 (mono)
delta = 0.001   # delta for P(s_k|m_j) 0.001
codebook_size = 400   # 13000
batch_size = 10000
epoch_size = 101
dim_features = 117
dim_labels = 1

# Network parameter
scale_soft = 10.0
learning_rate_pre = 1e-2
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
