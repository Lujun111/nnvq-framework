import pandas as pd
import collections
import sys
import argparse
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

    def concat_data(self, nj, splice, path_data, path_phonemes, output_folder):
        dataset = DataIterator(nj, splice, path_data)

        create_stats = True
        if path_data in ['test', 'dev']:
            create_stats = False

        # set dim for splice
        # TODO could be solved in a better way
        if splice:
            dim = 117
        else:
            dim = 39

        print('Loading alignment dict')
        alignment_dict = {}
        for key, mat in kaldi_io.read_ali_ark(path_phonemes):
            alignment_dict[key] = mat

        print('Loading done')

        count = 1
        tmp_dict = {}
        gather_data = []
        print('Creating filtered training data and merge them with the labels')
        while True:
            try:
                for key, mat in kaldi_io.read_mat_ark(dataset.next_file()):
                    # we need to filter the training data because we don't have the alignments for all the
                    # training data. Therefor, we have to avoid to use this data for training our HMMs
                    # TODO Could also work with --> check performance difference
                    if key in list(alignment_dict.keys()) and \
                                    mat.shape[0] == alignment_dict[key].shape[0]:
                        tmp_dict[key] = pd.concat([pd.DataFrame(mat),
                                                   pd.DataFrame(alignment_dict[key])], axis=1)
                        gather_data.append(mat)

                od = collections.OrderedDict(sorted(tmp_dict.items()))

                # write filtered training data and the labels to files
                with open(output_folder + '/feats_vq_' + str(count), 'wb') as f:
                    for key, mat in list(od.items()):
                        kaldi_io.write_mat(f, mat.values.astype(np.float32, copy=False), key=key)
                # write the filtered training data
                with open(output_folder + '/features_' + str(count), 'wb') as f:
                    for key, mat in list(od.items()):
                        kaldi_io.write_mat(f, mat.values.astype(np.float32, copy=False)[:, :dim], key=key)
                tmp_dict = {}
                count += 1

            except StopIteration:
                if create_stats:
                    print('Saving std and mean of data to stats.mat')
                    self._create_and_save_stats(np.concatenate(gather_data))
                break


def str2bool(v):
    """
    Converts string argument to bool
    :param v:
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(arguments):
    """
    Create argument parser to execute python file from console
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # add number of jobs / how the data is split
    parser.add_argument('--nj', type=int, help='number of jobs', default=20)
    # define source data folder where the features are saved
    parser.add_argument('data', type=str, help='data folder which contains the features')
    # define alignment folder where the labels for the training data are coming from
    parser.add_argument('ali', type=str, help='alignment folder which contains the labels of the data')
    # define the output folder where to save the concat data
    parser.add_argument('out', type=str, help='output folder to save the concat data')
    # define splice-feats or not
    parser.add_argument('--splice', type=str2bool, help='flag for spliced features', default=False)
    # parse all arguments to parser
    args = parser.parse_args(arguments)

    # print the arguments which we fed into
    for arg in vars(args):
        print("Argument {:14}: {}".format(arg, getattr(args, arg)))

    # create object and perform task
    kaldi_misc_helper = KaldiMiscHelper()
    kaldi_misc_helper.concat_data(args.nj, args.splice, args.data, args.ali, args.out)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
    # test = KaldiMiscHelper()
    # test.merge_data_phonemes(20, 'train_20kshort_nodup', '../alignments/nnet_labels/all_ali', '../')
    # test.merge_special(20, 'train_40kshort_nodup', '../alignments/tri3/all_ali', '../')
    # test.concat_data(30, 'dev', 'tmp/state_labels/all_ali_dev', 'tmp')
    # test.merge_special(30, 'test', '../alignments/test/all_ali', '../')
    # test.merge_special(20, 'test', '../alignments/test/all_ali', '../')
    # test.merge_special(20, 'train_20kshort_nodup', 'tmp/state_labels/all_ali_train', 'tmp')
