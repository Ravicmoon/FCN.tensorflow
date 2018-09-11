import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'An input color image',
    'gt': 'A ground truth image',
}

class TFRecordDataset:

    def __init__(self, tfrecord_dir, dataset_name):

        self.tfrecord_dir = tfrecord_dir
        self.dataset_name = dataset_name


    def _get_num_samples(self, start_pattern):

        num_samples = 0
        tfrecords_to_count = [os.path.join(self.tfrecord_dir, file) for file in os.listdir(self.tfrecord_dir) 
                              if file.startswith(start_pattern)]
        
        for tfrecord_file in tfrecords_to_count:
            for record in tf.python_io.tf_record_iterator(tfrecord_file):
                num_samples += 1

        return num_samples


    def _get_dataset(self, mode):

        start_pattern = self.dataset_name + '_' + mode

        reader = tf.TFRecordReader;

        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string),
            'image/format': tf.FixedLenFeature((), tf.string),
            'gt/encoded': tf.FixedLenFeature((), tf.string),
            'gt/format': tf.FixedLenFeature((), tf.string),
        }

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(),
            'gt': slim.tfexample_decoder.Image(image_key='gt/encoded', format_key='gt/format', channels=1),
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

        return slim.dataset.Dataset(
                data_sources = os.path.join(self.tfrecord_dir, start_pattern + '*'),
                reader = reader,
                decoder = decoder,
                num_samples=self._get_num_samples(start_pattern),
                items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)


    def load_batch(self, mode, batch_size = 32, height = 224, width = 224):
        """Loads a single batch of data.

        Args:
          dataset: The dataset to load.
          batch_size: The number of images in the batch.
          height: The size of each image after preprocessing.
          width: The size of each image after preprocessing.

        Returns:
          images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
          gts: A Tensor of size [batch_size, height, width, 1], image samples that can be used for visualization.
        """

        assert(mode in ['train', 'valid'])
    
        dataset = self._get_dataset(mode)
        shuffle = True if mode == 'train' else False
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=shuffle, common_queue_capacity = 512, common_queue_min = 256)
    
        [image, gt] = provider.get(['image', 'gt'])

        # image: resize with crop
        image = tf.image.resize_images(image, [height, width])
        image = tf.to_float(image)

        means = [123.68, 116.78, 103.94]

        num_channels = image.get_shape().as_list()[-1]
        channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] -= means[i]
        image = tf.concat(axis=2, values=channels)

        gt = tf.image.resize_images(gt, [height, width])
        gt = tf.to_int64(gt)

        # Batch it up.
        images, gts = tf.train.batch([image, gt], batch_size = batch_size, num_threads = 1, capacity = 2 * batch_size)

        return images, gts, dataset.num_samples