from KmeansVqHelper import KmeansVqMmi
from kaldi_io import kaldi_io
import numpy as np


# test = KmeansVqMmi(400)
# # test.create_dataset(35, 0.001, '../plain_feats_20k/train', 'vq_mmi/')
# test.init_training('vq_mmi/dataset.mat', 'vq_mmi/codebook_single.mat', 0.5)
# test.do_mmi()

test = kaldi_io.read_ali_ark('../alignments/nnet_labels/all_ali')


alignment_dict = {}

for key, mat in test:
    alignment_dict[key] = mat