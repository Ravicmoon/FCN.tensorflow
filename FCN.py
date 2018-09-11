import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import numpy as np
import os
import cv2

from dataset import TFRecordDataset
from PIL import Image


tf.logging.set_verbosity(tf.logging.INFO)


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('num_steps', '300000', 'number of steps for optimization')
tf.flags.DEFINE_integer('batch_size', '20', 'batch size for training')
tf.flags.DEFINE_integer('num_classes', '3', 'number of classes in dataset')
tf.flags.DEFINE_float('learning_rate', '1e-10', 'fixed learning rate for Momentum Optimizer')
tf.flags.DEFINE_float('momentum', '0.99', 'momentum for Momentum Optimizer')
tf.flags.DEFINE_string('ckpt_path', 'vgg_16_160830.ckpt', 'path to checkpoint')
tf.flags.DEFINE_string('log_dir', 'ckpt_180911_v1', 'path to logging directory')
tf.flags.DEFINE_string('data_dir', 'data', 'path to dataset')
tf.flags.DEFINE_string('data_name', 'Cityscapes', 'name of dataset')
tf.flags.DEFINE_string('mode', 'valid', 'either train or valid')


def FCN8(images, num_classes):
    
    model = nets.vgg
    with slim.arg_scope(model.vgg_arg_scope()):
        score, end_points = model.vgg_16(images, num_classes, spatial_squeeze=False)
    
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


def IOU_for_label(gt, pred, label):
    
    gt[gt != label] = 0
    pred[pred != label] = 0
                
    I = np.logical_and(gt, pred)
    U = np.logical_or(gt, pred)
    return np.count_nonzero(I) / np.count_nonzero(U)


def main(_):
    
    log_dir = FLAGS.log_dir

    '''
     Setting up the model
    '''
    dataset = TFRecordDataset(FLAGS.data_dir, FLAGS.data_name)
    images, gts, num_samples = dataset.load_batch(FLAGS.mode, FLAGS.batch_size if FLAGS.mode == 'train' else 1)

    paddings = tf.constant([[0, 0], [96, 96], [96, 96], [0, 0]])
    pad_images = tf.pad(images, paddings, 'CONSTANT')

    pred, logits = FCN8(pad_images, FLAGS.num_classes)

    if FLAGS.mode == 'valid':

        saver = tf.train.Saver(slim.get_variables_to_restore())
        coord = tf.train.Coordinator()
        
        with tf.Session() as sess:
            
            '''
             Restore parameters from check point
            '''
            saver.restore(sess, tf.train.latest_checkpoint(log_dir))

            tf.train.start_queue_runners(sess, coord)
            
            eval_dir = os.path.join(log_dir, 'eval')
            if not tf.gfile.Exists(eval_dir):
                tf.gfile.MakeDirs(eval_dir)

            mIOU = 0
            exp = int(np.log10(num_samples)) + 1
            
            for i in range(num_samples):

                gt_img, pred_img = sess.run([gts, pred])

                gt_img = np.squeeze(gt_img)
                gt_img = gt_img.astype(np.uint8)
                pred_img = np.squeeze(pred_img)
                pred_img = pred_img.astype(np.uint8)

                mIOU += IOU_for_label(gt_img, pred_img, 2)
                
                img_res = gt_img.shape;
                output = np.zeros((img_res[0], 2 * img_res[1]), dtype=np.uint8)

                output[:, 0*img_res[1]:1*img_res[1]] = gt_img * 100
                output[:, 1*img_res[1]:2*img_res[1]] = pred_img * 100

                cv2.imwrite(os.path.join(eval_dir, FLAGS.mode + str(i).zfill(exp) + '.png'), output)

            coord.request_stop()
            coord.join()

            mIOU /= num_samples
            print('mIU: ' + str(mIOU))


    elif FLAGS.mode == 'train':
        '''
         Define the loss function
        '''
        loss = tf.losses.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(gts))
        total_loss = tf.losses.get_total_loss()

        '''
         Define summaries
        '''
        tf.summary.image('image', images)
        tf.summary.image('gt', tf.cast(gts * 100, tf.uint8))
        tf.summary.image('pred', tf.cast(pred * 100, tf.uint8))
        tf.summary.scalar('loss', loss)

        '''
         Define initialize function
        '''
        exclude = ['vgg_16/fc8', 'FCN']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(FLAGS.ckpt_path, variables_to_restore, ignore_missing_vars = True)

        '''
         Set learning rates and optimizer
         (Fixed learning rate ofr Momentum Optimizer)
        '''
        optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum)
    
        '''
         Training phase
        '''
        if not tf.gfile.Exists(log_dir):
          tf.gfile.MakeDirs(log_dir)

        with open(os.path.join(log_dir + 'info.txt'), 'w') as f:
            f.write('num_steps: ' + str(FLAGS.num_steps) + '\n')
            f.write('batch_size: ' + str(FLAGS.batch_size) + '\n')
            f.write('learning_rate: ' + str(FLAGS.learning_rate) + '\n')
            f.write('momentum: ' + str(FLAGS.momentum) + '\n')
            f.write('ckpt_path: ' + FLAGS.ckpt_path + '\n')
            f.write('data_dir: ' + FLAGS.data_dir + '\n')
            f.write('data_name: ' + FLAGS.data_name + '\n')
            f.write('mode: ' + FLAGS.mode)

        train_op = slim.learning.create_train_op(total_loss, optimizer)

        final_loss = slim.learning.train(
                train_op = train_op,
                logdir = log_dir,
                init_fn = init_fn,
                number_of_steps = FLAGS.num_steps,
                summary_op = tf.summary.merge_all(),
                save_summaries_secs = 120,
                save_interval_secs = 240)

        print('Finished training. Final batch loss %f' %final_loss)


    else:

        print('Unknown mode')


if __name__ == "__main__":
    tf.app.run()
