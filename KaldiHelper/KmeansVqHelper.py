#!/home/ga96yar/tensorflow_py3/bin/python
# -*- coding: utf-8 -*-
"""
Calculate k-means.
"""
import os
import sys
import argparse
import pandas as pd
import pickle
import copy
import numpy as np
import collections
from kaldi_io import kaldi_io
from sklearn.cluster import MiniBatchKMeans, KMeans
from scipy.cluster.vq import vq, whiten
from KaldiHelper.MiscHelper import Misc
from KaldiHelper.IteratorHelper import DataIterator


class KmeansVq(object):
    def __init__(self, cluster, multiple=False, whitening=False, MiniBatch=True, KaldiFormatting=False):
        self._kaldi_formatting = KaldiFormatting
        self._multiple = multiple
        self._whitening = whitening
        self._MiniBatch = MiniBatch
        self._num_cluster = cluster
        self._dict_codebook = None
        self.codebook = None
        # self._load_codebook_pd('codebook.pd')
        # self._build_dataset()

    def create_codebook(self, nj, data_folder):
        # create keys for enumeration
        if self._multiple:
            keys = ['energy', 'raw', 'delta', 'dd']
        else:
            keys = ['simple']

        # init 4 minibatchkmeans for energy, raw, delta and delta delta features
        dict_kmeans = {}
        for key in keys:
            dict_kmeans[key] = MiniBatchKMeans(n_clusters=self._num_cluster, init='random', batch_size=200,
                                               verbose=1, reassignment_ratio=0.001, max_no_improvement=100,
                                               n_init=self._num_cluster)

        # create dataiterator
        dataset = DataIterator(nj, data_folder)

        # iterator and do kmeans
        df = pd.DataFrame()
        while True:
            try:
                data_path = dataset.next_file()
                print(data_path)
                for key, mat in kaldi_io.read_mat_ark(data_path):
                    tmp_df = pd.DataFrame(mat)
                    df = df.append(tmp_df.sample(int(tmp_df.shape[0] * 1.0)))

                    if df.shape[0] > 1000:
                        # so kmeans for every features
                        if self._multiple:
                            dict_kmeans['energy'].partial_fit(whiten(df.values[:, [0, 13, 26]]))
                            dict_kmeans['raw'].partial_fit(whiten(df.values[:, range(1, 13, 1)]))
                            dict_kmeans['delta'].partial_fit(whiten(df.values[:, range(14, 26, 1)]))
                            dict_kmeans['dd'].partial_fit(whiten(df.values[:, range(27, 39, 1)]))
                        else:
                            if self._whitening:
                                dict_kmeans['simple'].partial_fit(whiten(df.values))
                            else:
                                dict_kmeans['simple'].partial_fit(df.values)
                        self._dict_codebook = dict_kmeans
                        df = pd.DataFrame()  # clean up
            except StopIteration:
                break

    def load_pickle_codebook(self, infile):
        self._dict_codebook = pickle.loads(infile.read())

    def save_pickle_codebook(self, outfile):
        outfile.write(pickle.dumps(self._dict_codebook))

    def load_codebook(self, path):
        if not self._kaldi_formatting:
            raise TypeError
        for key, mat in kaldi_io.read_mat_ark(path):
            self.codebook = mat

    def save_codebook(self, path):
        if not self._kaldi_formatting:
            raise TypeError
        # prepare codebook for saving
        path_new = str.split(path, '.')
        assert len(self._dict_codebook) > 0

        if len(self._dict_codebook) > 1:
            # prepare codebook for multiple vqs
            self.codebook = np.zeros([self._num_cluster, 39])  # 39 is the feature dimension
            keys = ['energy', 'raw', 'delta', 'dd']
            dict_indicies = {'energy': [0, 13, 26], 'raw': range(1, 13, 1), 'delta': range(14, 26, 1),
                             'dd': range(27, 39, 1)}
            for key in keys:
                self.codebook[:, dict_indicies[key]] = self._dict_codebook[key].cluster_centers_
            path = path_new[0] + '_multiple.' + path_new[1]
        else:
            self.codebook = self._dict_codebook['simple'].cluster_centers_
            path = path_new[0] + '_single.' + path_new[1]

        with open(path, 'wb') as f:
            # print(self.codebook)
            kaldi_io.write_mat(f, self.codebook, key='cb')

    def vq_data(self, nj, data_folder, output_folder):
        # vqing traing data
        assert self.codebook.shape[0] > 0
        print('VQing training data...')

        dataset = DataIterator(nj, data_folder)

        keys = []
        dict_vq, dict_indicies = {}, {}
        if self._multiple:
            keys = ['energy', 'raw', 'delta', 'dd']
            dict_indicies = {'energy': [0, 13, 26], 'raw': range(1, 13, 1), 'delta': range(14, 26, 1),
                             'dd': range(27, 39, 1)}
        else:
            keys = ['simple']
            dict_indicies = {'simple': range(0, 39)}

        for key in keys:
            dict_vq[key] = self.codebook[:, dict_indicies[key]]

        tmp_dict = {}
        labels_all = []
        phoneme_all = []
        count = 1
        while True:
            try:
                data_path = dataset.next_file()
                print("Data path is in ", data_path)
                for key, mat in kaldi_io.read_mat_ark(data_path):
                    if self._multiple:
                        # getting label for every vq
                        df = pd.DataFrame(
                            vq(whiten(mat[:, dict_indicies['energy']]), dict_vq['energy'])[0][:, np.newaxis])
                        df = pd.concat([df, pd.DataFrame(vq(whiten(mat[:, dict_indicies['raw']]),
                                                            dict_vq['raw'])[0][:, np.newaxis])], axis=1)
                        df = pd.concat([df, pd.DataFrame(vq(whiten(mat[:, dict_indicies['delta']]),
                                                            dict_vq['delta'])[0][:, np.newaxis])], axis=1)
                        df = pd.concat([df, pd.DataFrame(vq(whiten(mat[:, dict_indicies['dd']]),
                                                            dict_vq['dd'])[0][:, np.newaxis])], axis=1)
                    else:
                        if self._whitening:
                            df = pd.DataFrame(vq(whiten(mat[:, :39]), dict_vq['simple'])[0][:, np.newaxis])
                            labels_all.append(df.values)
                        else:
                            df = pd.DataFrame(vq(mat[:, :39], dict_vq['simple'])[0][:, np.newaxis])
                            labels_all.append(df.values)

                        if np.shape(mat)[1] > 39:
                            phoneme_all.append(mat[:, 39])

                    # add to tmp_dict for later saving
                    tmp_dict[key] = df

                # ordered dict
                od = collections.OrderedDict(sorted(tmp_dict.items()))

                # save label-stream from vq
                with open(output_folder + '/feats_vq_' + str(count), 'wb') as f:
                    for key, mat in list(od.items()):
                        kaldi_io.write_mat(f, mat.values.astype(np.float32, copy=False), key=key)

                tmp_dict = {}
                count += 1

            except StopIteration:
                # calc MI
                if False:
                    misc = Misc()
                    labels_all = np.concatenate(labels_all)
                    # labels_all = np.reshape(labels_all, [np.shape(labels_all)[0] * np.shape(labels_all)[1]],
                    #                         np.shape(labels_all)[2])
                    phoneme_all = np.concatenate(phoneme_all)
                    # phoneme_all = np.reshape(phoneme_all, [np.shape(phoneme_all)[0] * np.shape(phoneme_all)[1]],
                    #                          np.shape(phoneme_all)[2])
                    print(misc.calculate_mi(labels_all, phoneme_all))
                break


class KmeansVqMmi(object):

    def __init__(self, cb_size):
        self._cbsize = cb_size
        self._weights = None
        self._dataset = None
        self._mi = None
        self._delta = None
        self._sorted_labels = None

        # init methods
        # self._load_weights(path_weights)
        # self._load_dataset(path_data)

    def _get_sort_label_stream(self, stream, cb_size):
        """

        :param stream: input label stream
        :param cb_size: size of codebook
        :return: return occupancy of label stream
        """

        # define counter array for labels
        count_array = np.zeros(cb_size)

        # count occupancy
        for element in stream:
            count_array[element] += 1.0

        # sort of accupancy and reverse result (from big to small)
        occupany = np.argsort(count_array)[::-1]

        # some debugging
        # print(count_array)
        # print(occupany)

        return occupany

    def _modify_weight(self, weights, index_tuple, delta):
        """
        :param index_tuple: index tuple
        :param delta: delta parameter
        """

        weights[index_tuple[0], index_tuple[1]] += delta

        return weights

    def _load_weights(self, path):
        for key, mat in kaldi_io.read_mat_ark(path):
            self._weights = mat

    def _load_dataset(self, path):
        for key, mat in kaldi_io.read_mat_ark(path):
            self._dataset = mat

    def create_dataset(self, nj, frac, path_data, output_folder):

        dataset = DataIterator(nj, path_data)

        data = []
        misc = Misc()
        count_size = 0
        while True:
            try:
                data_path = dataset.next_file()
                print(data_path)
                for key, mat in kaldi_io.read_mat_ark(data_path):
                    df_mat = pd.DataFrame(mat)
                    np_mat = df_mat.sample(frac=frac).values
                    # np_mat[:, 39] = misc.trans_vec_to_phones(np_mat[:, 39])
                    data.append(np_mat)

            except StopIteration:
                data_sample = np.concatenate(data)
                print(data_sample.shape)
                data_dict = {}
                data_dict['data'] = data_sample

                with open(output_folder + '/dataset.mat', 'wb') as f:
                    for key, mat in list(data_dict.items()):
                        kaldi_io.write_mat(f, mat.astype(np.float32, copy=False), key=key)

                break

    def do_mmi(self):

        misc = Misc()

        i = 0  # sorted index
        j = 0  # activation index
        iter = 0
        print('Iteration: ' + str(iter))

        while j < 39:
            print('Activation: ' + str(j))
            tmp_weight = copy.deepcopy(self._weights)
            new_weights = self._modify_weight(tmp_weight,
                                              (self._sorted_labels[i], j), self._delta)
            new_labels, _ = vq(whiten(self._dataset[:, :39]), new_weights)

            mi = misc.calculate_mi(np.expand_dims(new_labels, 1), self._dataset[:, 39])
            # print('Tmp MI: ' + str(mi))

            if mi - self._mi > 0:
                self._weights = new_weights
                self._mi = mi
                self._save_weights()
                print(self._mi)
                j += 1
                self._delta = np.abs(self._delta)
            else:
                if self._delta < 0:
                    self._delta = np.abs(self._delta)
                    j += 1

                else:
                    self._delta = -self._delta

            if j == 39:
                j = 0
                i += 1
                print('Label number: ' + str(i))

            if i == 400:
                i = 0
                iter += 1
                # x.append(iter)
                # y.append(mi_old)
                # sc.set_offsets(np.c_[x, y])
                # fig.canvas.draw_idle()
                # plt.pause(0.1)
                print('Iteration: ' + str(iter))
                if iter > 20:
                    break
                print('Creating new sorted labels...')
                self._sorted_labels = self._get_sort_label_stream(new_labels, self._cbsize)

    def init_training(self, path_data, path_weights, delta):
        self._load_weights(path_weights)
        self._load_dataset(path_data)
        self._delta = delta
        assert self._weights is not None or self._dataset is not None

        misc = Misc()

        label, _ = vq(whiten(self._dataset[:, :39]), self._weights)

        # calc mutual information
        self._mi = misc.calculate_mi(label, self._dataset[:, 39])

        print('Init MI: ' + str(self._mi))

        codebook_size = 400

        self._sorted_labels = self._get_sort_label_stream(label, self._cbsize)

    def _save_weights(self):
        weights_dict = {'weights': self._weights}
        with open('weights_tmp.mat', 'wb') as f:
            for key, mat in list(weights_dict.items()):
                kaldi_io.write_mat(f, mat.astype(np.float32, copy=False), key=key)


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--numjobs", nargs=2, type=int, help="number of jobs", default=10)
    parser.add_argument("-k", "--kaldiformat", help="use kaldi-formatted codebook-file", action="store_true")
    subparsers = parser.add_subparsers(help='sub-command train|vq', dest='command')

    parser_train = subparsers.add_parser('train', help='training kmeans')
    parser_train.add_argument("-m", "--multiple", action="store_true")
    parser_train.add_argument("num_classes", type=int, help="number of classes or clusters")
    parser_train.add_argument('inputdir', help="Input Files", type=str)
    parser_train.add_argument('-o', '--outfile', help="Output file",
                        default=sys.stdout, type=argparse.FileType('wb'))

    parser_vq = subparsers.add_parser('vq', help='quanitzing a dataset')
    parser_vq.add_argument('codebook', help="codebook file", type=argparse.FileType('rb'))
    parser_vq.add_argument('outdir', help="Output directory", type=str)

    args = parser.parse_args(arguments)

    for arg in vars(args):
        print (" Argument {:14}: {}".format(arg, getattr(args, arg)))

    if args.command == 'train':
        print('   ----    training:')
        if not os.path.isdir(args.inputdir):
            raise AssertionError("inputdir {0} is not a directory".format(args.inputdir))
        kmeans = KmeansVq(args.num_classes, multiple=args.multiple)
        kmeans.create_codebook(args.numjobs, args.inputdir)
        kmeans.save_pickle_codebook(args.outfile)
    else:
        print('vq not yet implemented')

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

    # tmp.load_codebook('codebook_single.mat')
    # tmp.vq_data(35, '../plain_feats_20k/train_20k', '../plain_feats/backup_20k_vq/vq_train')
    # tmp.vq_data(30, '../plain_feats_20k/test', '../plain_feats/backup_20k_vq/vq_test')
    # tmp.mutual_information(35)
    # tmp.multiple_vq_data(20, 'train_20kshort_nodup', '../plain_feats/backup_20k_vq/vq_train')

    # tmp.vq_data(20, 'train_20kshort_nodup', '../exp/test_400_0/vq_train')
    # tmp.vq_data(30, 'test', '../exp/test_400_0/vq_test')
