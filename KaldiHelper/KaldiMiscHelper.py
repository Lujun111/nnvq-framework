#!/home/ga96yar/tensorflow_py3/bin/python
import pandas as pd
import collections
import sys
import argparse
import numpy as np
from kaldi_io import kaldi_io
from KaldiHelper.IteratorHelper import DataIterator, AlignmentIterator
from KaldiHelper.MiscHelper import Misc


class KaldiMiscHelper(object):
    def __init__(self, nj, splice, cmvn, dim=39):
        # TODO can we determine the dim on the fly?
        self._nj = nj
        self._splice = splice
        self._dim = dim
        self._cmvn = cmvn

    def _create_and_save_stats(self, mat):
        # tmp_mean = np.mean(mat, axis=0)
        # tmp_std = np.std(mat, axis=0)

        # get global mean and std
        num_samples = np.sum(mat[:, 0])
        dim = int((mat.shape[1] - 1) / 2)

        # global mean
        tmp_mean = np.sum((np.expand_dims(mat[:, 0], 1) * mat[:, 1:(dim+1)]), axis=0) / num_samples

        # global var
        tmp_std = np.sum(np.expand_dims(mat[:, 0], 1) * (mat[:, (dim+1):] + np.square(mat[:, 1:(dim+1)] - tmp_mean)),
                         axis=0) / num_samples
        tmp_std = np.sqrt(tmp_std)

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
        dataset = DataIterator(self._nj, self._splice, self._cmvn, path_data)
        phonemes = AlignmentIterator(nj, path_phonemes)

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

    def concat_data(self, path_data, path_phonemes, output_folder):
        dataset = DataIterator(self._nj, self._splice, self._cmvn, path_data)

        create_stats = True
        if path_data in ['test', 'dev']:
            create_stats = False

        # set dim
        dim = self._dim * (2 * self._splice + 1)

        print('Loading alignment dict')
        alignment_dict = {}
        for key, mat in kaldi_io.read_ali_ark(path_phonemes):
            alignment_dict[key] = mat

        print('Loading done')
        count = 1
        tmp_dict = {}
        # gather_stats: n, mean, var (nj rows)
        gather_stats = np.zeros([self._nj, 2 * dim + 1])
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
                # save stats for single file
                tmp_data = np.concatenate(gather_data)
                gather_stats[count-1, 0] = tmp_data.shape[0]   # add number of samples
                gather_stats[count-1, 1:(dim+1)] = np.mean(tmp_data, axis=0)  # add mean of file
                gather_stats[count-1, (dim+1):] = np.var(tmp_data, axis=0)  # add var of file
                # print(gather_stats)
                count += 1
                gather_data = []    # reset gather_data

            except StopIteration:
                if create_stats:
                    print('Saving std and mean of data to stats.mat')
                    self._create_and_save_stats(gather_stats)
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
    # define splice-feats or not
    parser.add_argument('--splice', type=int, help='flag for spliced features with context width',
                        default=0)
    # cmvn or global normalization
    parser.add_argument('--cmvn', type=str2bool, help='flag for cmvn or global normalization',
                        default=True)
    # define source data folder where the features are saved
    parser.add_argument('data', type=str, help='data folder which contains the features')
    # define alignment folder where the labels for the training data are coming from
    parser.add_argument('ali', type=str, help='alignment folder which contains the labels of the data')
    # define the output folder where to save the concat data
    parser.add_argument('out', type=str, help='output folder to save the concat data')
    # parse all arguments to parser
    args = parser.parse_args(arguments)

    # print the arguments which we fed into
    for arg in vars(args):
        print("Argument {:14}: {}".format(arg, getattr(args, arg)))

    # create object and perform task
    kaldi_misc_helper = KaldiMiscHelper(args.nj, args.splice, args.cmvn)
    kaldi_misc_helper.concat_data(args.data, args.ali, args.out)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
