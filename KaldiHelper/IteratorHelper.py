import os
import re


class DataIterator(object):
    """
    DataIterator object for iterating though the folders which were created by
    kaldi (split_data.sh from kaldi).
    E.g. split20 --> contains folders with name 1-20
    ATTENTION: The DataIterator only handles the path (string) to the split folders
    and does not actually load any data!!!
    """
    def __init__(self, nj, folder):
        """
        Init DataIterator

        :param nj:      number of jobs (e.g. split20 --> nj=20)
        :param folder:  path to data
        """
        # TODO self.path is hard coded
        self.path = '/home/ga96yar/kaldi/egs/tedlium/s5_r2'
        self._nj = nj
        self._folder = folder
        self._generator = None

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
    def __init__(self, nj, folder):
        """
        Init AlignmentIterator using the base class DataIterator
        """
        super().__init__(nj, folder)

    def _create_iterator(self):
        """
        Override the function to create a generator to iterate over alignment files
        :return:
        """
        assert type(self._nj) == int and type(self._folder) == str

        path_generator = [self._folder + '/' + s for s in os.listdir(self._folder)]
        # sort list for later processing
        convert = lambda text: int(text) if text.isdigit() else text
        path_generator.sort(key=lambda key: [convert(c) for c in re.split('([0-9]+)', key)])
        self._generator = iter(path_generator)