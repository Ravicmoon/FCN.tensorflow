import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

def FCN8_atonce(images, num_classes):
    paddings = tf.constant([[0, 0], [96, 96], [96, 96], [0, 0]])
    pad_images = tf.pad(images, paddings, 'CONSTANT')

    height = images.shape[1].value
    width = images.shape[2].value

    model = nets.vgg
    with slim.arg_scope(model.vgg_arg_scope()):
        score, end_points = model.vgg_16(pad_images, num_classes, spatial_squeeze=False)
    
    with tf.variable_scope('FCN'):
        score_pool3 = slim.conv2d(0.0001 * end_points['vgg_16/pool3'], num_classes, 1, scope='score_pool3')
        score_pool4 = slim.conv2d(0.01 * end_points['vgg_16/pool4'], num_classes, 1, scope='score_pool4')
    
        score_pool3c = tf.image.crop_to_bounding_box(score_pool3, 12, 12, int(height / 8), int(width / 8))
        score_pool4c = tf.image.crop_to_bounding_box(score_pool4, 6, 6, int(height / 16), int(width / 16))

        up_score = slim.conv2d_transpose(score, num_classes, 4, stride=2, scope='up_score')
        fuse1 = tf.add(up_score, score_pool4c, name='fuse1')

        up_fuse1 = slim.conv2d_transpose(fuse1, num_classes, 4, stride=2, scope='up_fuse1')
        fuse2 = tf.add(up_fuse1, score_pool3c, name='fuse2')

        up_fuse2 = slim.conv2d_transpose(fuse2, num_classes, 16, stride=8, scope='up_fuse2')

        pred = tf.argmax(up_fuse2, 3, name='pred')

    return tf.expand_dims(pred, 3), up_fuse2
