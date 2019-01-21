#!/home/ga96yar/tensorflow_py3/bin/python
import os
import re
import glob
import numpy as np


class DataIterator(object):
    """
    DataIterator object for iterating though the folders which were created by
    kaldi (split_data.sh from kaldi).
    E.g. split20 --> contains folders with name 1-20
    ATTENTION: The DataIterator only handles the path (string) to the split folders
    and does not actually load any data!!!
    """
    def __init__(self, nj, folder, splice=0, cmvn=True):
        """
        Init DataIterator

        :param nj:      number of jobs (e.g. split20 --> nj=20)
        :param folder:  path to data
        """
        # TODO self.path is hard coded
        # TODO cmvn
        self.path = '/home/ga96yar/kaldi/egs/tedlium/s5_r2'
        self._nj = nj
        self._splice = splice
        self._folder = folder
        self._generator = None
        self._cmvn = cmvn

        # create iterator for iterating through data
        self._create_iterator()

    def _create_iterator(self):
        """
        Create the generator for iteration

        The function differs between the default kaldi data path and a custom
        data path.
        Option 1: If the string in self.folder contains '/' or '..' the function uses the
        custom path to the data folder
        Option 2: If the string doesn't contain '/' or '..' the functions looks into the
        default data path of kaldi (e.g. /kaldi/egs/tedlium/s5_r2/data)
        """
        assert type(self._nj) == int and type(self._folder) == str

        base_str = 'copy-matrix scp:' + self.path + '/data/' + self._folder + '/split' + str(self._nj) + '/' + '{i}' + \
            '/feats.scp ark:-|'
        add_deltas_str = 'add-deltas ark:- ark:-|'

        if ('/' or '..') not in self._folder:
            assert (os.path.isdir(self.path + '/data/' + self._folder))
            # create splice string
            if self._splice > 0:
                splice_str = 'splice-feats --left-context=' + str(self._splice) + ' --right-context=' + \
                             str(self._splice) + ' ark:- ark:-|'
            else:
                splice_str = ''

            # create cmvn string
            if self._cmvn:
                cmvn_str = 'apply-cmvn --norm-vars=false --utt2spk=ark:' + self.path + '/data/' + self._folder + \
                           '/split' + str(self._nj) + '/' + '{i}' + '/utt2spk scp:' + \
                           self.path + '/data/' + self._folder + '/split' + str(self._nj) + '/' + '{i}' + \
                           '/cmvn.scp ark:- ark:- |'
                # + self.path + '/data/' + self._folder + \
                # '/split' + str(self._nj) + '/' + '{i}' + '/feats.scp

            else:
                cmvn_str = ''

            self._generator = ((base_str + cmvn_str + splice_str + add_deltas_str).format(i=i)
                               for i in range(1, self._nj + 1))

            # TODO cleanup
            # self._generator = ('apply-cmvn --norm-vars=true --utt2spk=ark:' + self.path + '/data/' +
            #                    self._folder + '/split' + str(self._nj) + '/' + str(i) + '/utt2spk scp:' +
            #                    self.path + '/data/' + self._folder + '/split' + str(self._nj) + '/' + str(i) +
            #                    '/cmvn.scp scp:' + self.path + '/data/' + self._folder +
            #                    '/split' + str(self._nj) + '/' + str(i) + '/feats.scp ark:- |'' \
            #                    ''splice-feats --left-context=' + str(self._splice) + ' --right-context=' +
            #                    str(self._splice) + ' ark:- ark:-| add-deltas ark:- ark:-|'
            #                    for i in range(1, self._nj + 1))
            # else:
            #     self._generator = ('splice-feats --left-context=' + str(self._splice) + ' --right-context=' +
            #                        str(self._splice) + ' scp:' + self.path + '/data/' + self._folder +
            #                        '/split' + str(self._nj) + '/' + str(i) +
            #                        '/feats.scp ark:-| add-deltas ark:- ark:-|' for i in range(1, self._nj + 1))
            # else:
            #     self._generator = ('add-deltas scp:' + self.path + '/data/' + self._folder + '/split' +
            #                        str(self._nj) + '/' + str(i) + '/feats.scp ark:-|' for i in range(1, self._nj + 1))
        else:
            # TODO no implementation of splice-feats for own folders, necessary?
            path_generator = [self._folder + '/' + s for s in os.listdir(self._folder)]
            # sort list for later processing
            convert = lambda text: int(text) if text.isdigit() else text
            path_generator.sort(key=lambda key: [convert(c) for c in re.split('([0-9]+)', key)])
            self._generator = iter(path_generator)

    def next_file(self):
        """
        Get next string to data (check the python documentary of next())
        :return:    return the next content of the generator
        """
        return next(self._generator)

    def get_size(self):
        """
        Get the number of folders
        :return:    return number of folders of the split data
        """
        return self._nj


class AlignmentIterator(DataIterator):
    """
    AlignmentIterator object for iterating though number of alignments. Usually has the
    same number of folders as DataIterator
    """
    def __init__(self, nj, folder, state_based=True, convert=False):
        """
        Init AlignmentIterator using the base class DataIterator

        :param state_based:     take state based or phone labels
        :param convert:         convert model states to monophone states
        :param dim:             dim of the stream
        """
        self._state_based = state_based
        self._convert = convert
        self.dim = 0
        super().__init__(nj, folder)

    def _create_iterator(self):
        """
        Override the function to create a generator to iterate over alignment files
        :return:
        """
        assert type(self._nj) == int and type(self._folder) == str

        # path to mono model for converting to monophone states
        path_mono = self.path + '/exp/mono'
        convert_str = '{state_str} {path}/final.mdl "ark,t:gunzip -c {path}/ali.{i}.gz|" ark:-|'

        # state based or phone based labels
        if self._state_based:
            state_based_str = 'ali-to-pdf'

            # convert triphone states to monophone states
            if self._convert:
                tmp_str = 'convert-ali {path}/final.mdl {path_mono}/final.mdl {path_mono}/tree ' \
                          '"ark,t:gunzip -c {path}/ali.{i}.gz|" ark:-|'
                convert_str = tmp_str + ' {state_str} {path_mono}/final.mdl ark:- ark:-|'
                self.dim = 127
            else:
                with open(self.path + '/exp/' + self._folder + '/final.occs', 'r') as f:
                    self.dim = np.array(f.readline().replace('[', '').replace(']', '').strip().split(' ')).shape[0]
        else:
            state_based_str = 'ali-to-phones --per-frame'
            self.dim = 41

        if ('/' or '..') not in self._folder:
            assert (os.path.isdir(self.path + '/exp/' + self._folder))

            path_tmp = self.path + '/exp/' + self._folder
            # base_str.format(path=self.path + '/exp/' + self._folder)
            # convert.format(path=self.path + '/exp/' + self._folder)
            self._generator = (convert_str.format(state_str=state_based_str, path=path_tmp, path_mono=path_mono, i=i)
                               for i in range(1, self._nj + 1))

        else:
            raise NotImplementedError
            # base_str.format(path=self._folder)
            # convert.format(path=self._folder)
            # self._generator = ((base_str + ' ' + convert).format(i=i) for i in range(1, self._nj + 1))
            # # filter ali out of the folder
            # path_generator = [f for f in os.listdir(self._folder) if re.match(r'ali\.[0-9]+.*\.gz', f)]
            #
            # assert(len(path_generator) == self._nj)
            # # sort list for later processing
            # convert = lambda text: int(text) if text.isdigit() else text
            # path_generator.sort(key=lambda key: [convert(c) for c in re.split('([0-9]+)', key)])
            # self._generator = iter(path_generator)