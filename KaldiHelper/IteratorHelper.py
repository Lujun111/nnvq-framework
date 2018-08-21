import os
import re
import kaldi_io


class DataIterator(object):
    def __init__(self, nj, folder):
        self.path = '/home/ga96yar/kaldi/egs/tedlium/s5_r2'
        self._nj = nj
        self._folder = folder
        self._generator = None

        # create iterator for iterating through data
        self._create_iterator()

    def _create_iterator(self):
        assert type(self._nj) == int and type(self._folder) == str
        if ('/' or '..') not in self._folder:
            assert (os.path.isdir(self.path + '/data/' + self._folder))
            self._generator = ('add-deltas scp:' + self.path + '/data/' + self._folder + '/split' + str(self._nj) + '/' + str(i) +
                               '/feats.scp ark:-|' for i in range(1, self._nj + 1))
        else:
            path_generator = [self.path + '/' + self._folder + '/' + s for s in os.listdir(self.path + '/' + self._folder)]
            # sort list for later processing
            convert = lambda text: int(text) if text.isdigit() else text
            path_generator.sort(key=lambda key: [convert(c) for c in re.split('([0-9]+)', key)])
            self._generator = iter(path_generator)

    def next_file(self):
        return next(self._generator)

    def get_size(self):
        return self._nj


class AlignmentIterator(DataIterator):
    def __init__(self, nj, folder):
        super().__init__(nj, folder)

    def _create_iterator(self):
        assert type(self._nj) == int and type(self._folder) == str

        path_generator = [self._folder + '/' + s for s in os.listdir(self._folder)]
        # sort list for later processing
        convert = lambda text: int(text) if text.isdigit() else text
        path_generator.sort(key=lambda key: [convert(c) for c in re.split('([0-9]+)', key)])
        self._generator = iter(path_generator)