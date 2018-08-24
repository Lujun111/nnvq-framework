import tensorflow as tf


class DataFeeder(object):
    """
        DataFeeder feeding data into the network in training
    """
    def __init__(self, list_tfrecords, batch_size, dim_features, dim_phonemes):
        """
        :param list_tfrecords:  list of tfrecord files in a folder
        :param batch_size:      batch size for training
        :param dim_features:    dimension of features
        :param dim_phonemes:    dimension of labels
        """
        self._list_tf_recs = list_tfrecords
        self._batch_size = batch_size
        self._dim_features = dim_features
        self._dim_phonemes = dim_phonemes

        dataset = self._input_fn()  # Dataset object from TF-API
        self.iterator = dataset.make_initializable_iterator()

    def _input_fn(self):
        """
        Create Dataset using TF-API

        :return: Dataset object
        """
        def _parse_function(example_proto):
            """
            Creates parse function for loading TFRecords files

            :param example_proto:   prototype coming from a TFRecords file
            :return:                data from TFRecords files
            """
            keys_to_features = {'x': tf.FixedLenFeature(self._dim_features, tf.float32),
                                'y': tf.FixedLenFeature(self._dim_phonemes, tf.float32)}
            parsed_features = tf.parse_single_example(example_proto, keys_to_features)
            return parsed_features['x'], parsed_features['y']

        assert (isinstance(self._list_tf_recs, list))  # check if we passing list-object to TFRecordDataset
        dataset = tf.data.TFRecordDataset(self._list_tf_recs)

        # Parse the record into tensors.
        dataset = dataset.map(_parse_function)
        dataset = dataset.shuffle(100000)
        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        return dataset

