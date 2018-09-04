import tensorflow as tf


class DataFeeder(object):
    """
        DataFeeder feeding data into the network in training
    """
    def __init__(self, dict_lists, settings, sess):
        """
        :param list_tfrecords:  list of tfrecord files in a folder
        :param batch_size:      batch size for training
        :param dim_features:    dimension of features
        :param dim_phonemes:    dimension of labels
        """

        if {'train', 'test', 'dev'} <= set(dict_lists):
            self._dict_lists = dict_lists
        else:
            raise ValueError('You are missing keys in the dict')
        self._batch_size = settings.batch_size
        self._dim_features = settings.dim_features
        self._dim_phonemes = settings.dim_labels
        self._sess = sess

        # same fields
        self.train = None
        self.test = None
        self.dev = None

        self._input_fn()  # Dataset object from TF-API

    def _parse_function(self, example_proto):
        """
        Creates parse function for loading TFRecords files

        :param example_proto:   prototype coming from a TFRecords file
        :return:                data from TFRecords files
        """
        keys_to_features = {'x': tf.FixedLenFeature(self._dim_features, tf.float32),
                            'y': tf.FixedLenFeature(self._dim_phonemes, tf.float32)}
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        return parsed_features['x'], parsed_features['y']

    def _input_fn(self):
        """
        Create Dataset using TF-API and iterate through the dict
        """
        # get self.train, self.test and selt.dev references
        for key, item in self._dict_lists.items():
            dataset = tf.data.TFRecordDataset(item)
            # Parse the record into tensors.
            dataset = dataset.map(self._parse_function)
            dataset = dataset.shuffle(100000)
            dataset = dataset.batch(self._batch_size, drop_remainder=True)
            # dict_ref[key] = dataset.make_initializable_iterator()
            setattr(self, key, dataset.make_initializable_iterator())

    def init_all(self):
        """
        Reset all initializer
        """
        # if (self.train and self.dev and self.test) is not None:
        for key, _ in self._dict_lists.items():
            self._sess.run(getattr(self, key).initializer)

    def init_train(self):
        """
        Reset train initialzer
        """
        self._sess.run(self.train.initializer)

    def init_dev(self):
        """
        Reset dev initialzer
        """
        self._sess.run(self.dev.initializer)

    def init_test(self):
        """
        Reset test initialzer
        """
        self._sess.run(self.test.initializer)


