import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

# For reference
_MEAN_IMAGE_NET = [123.68, 116.78, 103.94]

_MEAN_KITTI = [89.4767, 96.2149, 93.3779]
_MEAN_VISTAS = [105.6721, 118.0413, 121.7196]
_MEAN_CITYSCAPES = [73.2730, 82.7704, 72.4525]
_MEAN_PG = [115.7738, 114.2015, 112.9132]

_STD_KITTI = [71.7813, 76.4437, 80.0591]
_STD_VISTAS = [64.5505, 67.9890, 76.0227]
_STD_CITYSCAPES = [46.2474, 47.0882, 46.4506]
_STD_PG = [64.8986, 67.8520, 74.0758]

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

        keys_to_features, items_to_handlers = self._get_decode_pattern()

        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

        return slim.dataset.Dataset(
                data_sources = os.path.join(self.tfrecord_dir, start_pattern + '*'),
                reader = reader,
                decoder = decoder,
                num_samples=self._get_num_samples(start_pattern),
                items_to_descriptions=self._items_to_description())


    def load_batch(self, mode, batch_size=32, height=224, width=224):
        """Loads a single batch of data.

        Args:
          dataset: The dataset to load.
          batch_size: The number of images in the batch.
          height: The size of each image after preprocessing.
          width: The size of each image after preprocessing.

        Returns:
          images: A Tensor of size [batch_size, height, width, 3], preprocessed input images.
          gts: A Tensor of size [batch_size, height, width, 1], annotated images.
        """

        assert(mode in ['train', 'valid'])
    
        dataset = self._get_dataset(mode)
        shuffle = True if mode == 'train' else False
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=shuffle, common_queue_capacity=512, common_queue_min=256)

        return self._preprocess(provider, mode, batch_size, height, width), dataset.num_samples


# TFRecord file reader for segmentation
class TFRecordSegDataset(TFRecordDataset):

    def __init__(self, tfrecord_dir, dataset_name):

        TFRecordDataset.__init__(self, tfrecord_dir, dataset_name)


    def _get_decode_pattern(self):

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

        return keys_to_features, items_to_handlers


    def _items_to_description(self):

        return {'image': 'An input color image',
                'gt': 'A ground truth image'}


    def _normalize_image(self, image, means, stds):
        ''' Revised from vgg_preprocessing.py in TensorFlowOnSpark
        (https://github.com/yahoo/TensorFlowOnSpark/tree/master/examples/slim/preprocessing)
        '''
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        num_channels = image.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')
        
        channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] = (channels[i] - means[i]) / stds[i]
        return tf.concat(axis=2, values=channels)


    def _preprocess(self, provider, mode, batch_size, height, width):

        [image, label] = provider.get(['image', 'gt'])

        # Resize and normalize input images
        org_image = tf.image.resize_images(image, [height, width])
        image = tf.to_float(org_image)
        if self.dataset_name == 'KITTI':
            image = self._normalize_image(image, _MEAN_KITTI, _STD_KITTI)
        if self.dataset_name.find('Vistas') != -1:
            image = self._normalize_image(image, _MEAN_VISTAS, _STD_VISTAS)
        if self.dataset_name.find('Cityscapes') != -1:
            image = self._normalize_image(image, _MEAN_CITYSCAPES, _STD_CITYSCAPES)
        if self.dataset_name.find('PG') != -1:
            image = self._normalize_image(image, _MEAN_PG, _STD_PG)

        # Resize ground truth label
        label = tf.image.resize_images(label, [height, width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label = tf.to_int64(label)

        # Batch it up.
        images, labels, org_images = tf.train.batch([image, label, org_image], batch_size=batch_size, capacity=2*batch_size)

        return images, labels, org_images