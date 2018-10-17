import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

def FCN8_atonce(images, num_classes):
    paddings = tf.constant([[0, 0], [96, 96], [96, 96], [0, 0]])
    pad_images = tf.pad(images, paddings, 'CONSTANT')

    model = nets.vgg
    with slim.arg_scope(model.vgg_arg_scope()):
        score, end_points = model.vgg_16(pad_images, num_classes, spatial_squeeze=False)
    
    with tf.variable_scope('FCN'):
        score_pool3 = slim.conv2d(0.0001 * end_points['vgg_16/pool3'], num_classes, 1, scope='score_pool3')
        score_pool4 = slim.conv2d(0.01 * end_points['vgg_16/pool4'], num_classes, 1, scope='score_pool4')
    
        score_pool3c = tf.image.central_crop(score_pool3, 7 / 13)
        score_pool4c = tf.image.central_crop(score_pool4, 7 / 13)

        up_score = slim.conv2d_transpose(score, num_classes, 4, stride=2, scope='up_score')
        fuse1 = tf.add(up_score, score_pool4c, name='fuse1')

        up_fuse1 = slim.conv2d_transpose(fuse1, num_classes, 4, stride=2, scope='up_fuse1')
        fuse2 = tf.add(up_fuse1, score_pool3c, name='fuse2')

        up_fuse2 = slim.conv2d_transpose(fuse2, num_classes, 16, stride=8, scope='up_fuse2')

        pred = tf.argmax(up_fuse2, 3, name='pred')

    return tf.expand_dims(pred, 3), up_fuse2
