"""
Setting file to define global variables
"""


class Settings(object):
    # path to data
    path_train = 'tf_data/splice_1f/train_pdf_20k_splice_1f_cmn'
    path_test = 'tf_data/splice_1f/test_pdf_20k_splice_1f_cmn'
    path_dev = 'tf_data/splice_1f/dev_pdf_20k_splice_1f_cmn'

    path_tensorboard = 'tensorboard'
    path_checkpoint = 'model_checkpoint'
    path_restore = path_checkpoint + '/best_model_127pdf'
    path_meta = 'meta_data'

    # Hyperparameter
    num_labels = 127   # 4131(tri3)    # 2026 (tri1)    # 127 (mono)
    delta = 0.001   # delta for P(s_k|m_j) 0.001
    codebook_size = 1000  # 13000
    batch_size = 15000
    epoch_size = 51
    dim_features = 39  # 39 (1f)   # 117 (3f)  # 195 (5f)
    dim_labels = 1

    # Network parameter
    scale_soft = 15.0
    learning_rate = current_lr = 1e-2
    lr_epoch_decrease = 10
    lr_decay = 0.5
    learning_rate_post = 1e-3
    do_rate = 0.2
    exponential_decay = False

    # train Network
    l2_loss_reg = True
    scale_l2 = 1e-7
    restore = False
    train_prob = False
    vqing = False
    sampling_discrete = False
    # create P(s_k|m_j) from training data
    create_conditioned_prob = False
    identifier = 'nnvq'     # possible identifiers: nnvq, nnvq_tri, vanilla, combination

    # Inference
    inference = False
