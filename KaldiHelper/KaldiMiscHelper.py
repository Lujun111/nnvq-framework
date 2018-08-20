import pandas as pd
import collections
import re
import numpy as np
from kaldi_io import kaldi_io
from KaldiHelper.IteratorHelper import DataIterator, AlignmentIterator
from KaldiHelper.MiscHelper import Misc


class KaldiMiscHelper(object):
    def __init__(self):
        pass

    def _create_and_save_stats(self, mat):
        tmp_mean = np.mean(mat, axis=0)
        tmp_std = np.std(mat, axis=0)

        # saving to stats matrix
        stats_dict = {
            'mean': np.expand_dims(tmp_mean, 1),
            'std': np.expand_dims(tmp_std, 1)
        }
        print(stats_dict)
        with open('stats.mat', 'wb') as f:
            for key, mat in list(stats_dict.items()):
                kaldi_io.write_mat(f, mat.astype(np.float32, copy=False), key=key)

    def merge_data_phonemes(self, nj, path_data, path_phonemes, output_folder):
        assert type(path_data) == str and type(path_phonemes) == str

        # create Iterators
        dataset = DataIterator(nj, path_data)
        phonemes = AlignmentIterator(nj, path_phonemes)
        misc = Misc()

        # iterate through data
        count = 1
        tmp_dict = {}
        while True:
            try:
                for (key_data, mat_data), (key_pho, mat_pho) in zip(kaldi_io.read_mat_ark(dataset.next_file()),
                                                                    kaldi_io.read_ali_ark(phonemes.next_file())):
                    # check for same key
                    if key_data == key_pho:
                        print(key_data)
                        tmp_dict[key_data] = pd.concat([pd.DataFrame(mat_data),
                                                        pd.DataFrame(mat_pho)],
                                                       axis=1)

                with open(output_folder + '/feats_vq_' + str(count), 'wb') as f:
                    for key, mat in list(tmp_dict.items()):
                        kaldi_io.write_mat(f, mat.values.astype(np.float32, copy=False), key=key)

                tmp_dict = {}
                count += 1

            except StopIteration:
                break

    def merge_special(self, nj, path_data, path_phonemes, output_folder):
        dataset = DataIterator(nj, path_data)

        create_stats = True
        if path_data == 'test' or 'dev':
            create_stats = False
            # adding_string = ''

        print('Loading alignment dict')
        alignment_dict = {}
        for key, mat in kaldi_io.read_ali_ark(path_phonemes):
            alignment_dict[key] = mat

        print('Loading done')

        count = 1
        tmp_dict = {}
        gather_data = []
        while True:
            try:
                for key, mat in kaldi_io.read_mat_ark(dataset.next_file()):
                    if key in list(alignment_dict.keys()) and \
                                    mat.shape[0] == alignment_dict[key].shape[0]:
                        tmp_dict[key] = pd.concat([pd.DataFrame(mat),
                                                   pd.DataFrame(alignment_dict[key])], axis=1)
                        gather_data.append(mat)

                od = collections.OrderedDict(sorted(tmp_dict.items()))

                # write features + phonemes
                with open(output_folder + '/feats_vq_' + str(count), 'wb') as f:
                    for key, mat in list(od.items()):
                        kaldi_io.write_mat(f, mat.values.astype(np.float32, copy=False), key=key)

                with open(output_folder + '/features_' + str(count), 'wb') as f:
                    for key, mat in list(od.items()):
                        kaldi_io.write_mat(f, mat.values.astype(np.float32, copy=False)[:, :39], key=key)
                tmp_dict = {}
                count += 1

            except StopIteration:
                if create_stats:
                    print('Saving std and mean of data to stats.mat')
                    self._create_and_save_stats(np.concatenate(gather_data))
                break


if __name__ == "__main__":
    test = KaldiMiscHelper()
    # test.merge_data_phonemes(20, 'train_20kshort_nodup', '../alignments/nnet_labels/all_ali', '../')
    # test.merge_special(20, 'train_40kshort_nodup', '../alignments/tri3/all_ali', '../')
    test.merge_special(30, 'dev', 'tmp/state_labels/all_ali_dev', 'tmp')
    # test.merge_special(30, 'test', '../alignments/test/all_ali', '../')
    # test.merge_special(20, 'test', '../alignments/test/all_ali', '../')
    # test.merge_special(20, 'train_20kshort_nodup', 'tmp/state_labels/all_ali_train', 'tmp')
