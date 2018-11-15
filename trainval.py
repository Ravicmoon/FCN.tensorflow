import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import numpy as np
import time
import os
import cv2

from FCN import FCN8_atonce
from dataset import TFRecordSegDataset
from PIL import Image


tf.logging.set_verbosity(tf.logging.INFO)


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('num_epochs', '50', 'number of epochs for optimization')
tf.flags.DEFINE_integer('batch_size', '2', 'batch size for training')
tf.flags.DEFINE_integer('num_classes', '2', 'number of classes in dataset')
tf.flags.DEFINE_integer('height', '224', 'height of input images')
tf.flags.DEFINE_integer('width', '224', 'width of input images')
tf.flags.DEFINE_float('learning_rate', '2e-4', 'learning rate for optimizer')
tf.flags.DEFINE_float('momentum', '0.99', 'momentum for Momentum Optimizer')
tf.flags.DEFINE_float('lr_decay_rate', '0.99', 'decay rate of learning rate')
tf.flags.DEFINE_bool('lr_decay', 'True', 'exponentially decay learning rate')
tf.flags.DEFINE_string('ckpt_path', 'vgg_16_160830.ckpt', 'path to checkpoint')
tf.flags.DEFINE_string('log_dir', 'ckpt_181005_v1', 'path to logging directory')
tf.flags.DEFINE_string('data_dir', 'data', 'path to dataset')
tf.flags.DEFINE_string('data_name', 'Cityscapes', 'name of dataset')
tf.flags.DEFINE_string('mode', 'train', 'either train or valid')
tf.flags.DEFINE_string('optimizer', 'Adam', 'supports momentum and Adam')


def calc_IOU(label, pred, n):
    
    label_bin = np.copy(label)
    label_bin[label_bin != n] = 0

    pred_bin = np.copy(pred)
    pred_bin[pred_bin != n] = 0
                
    I = np.logical_and(label_bin, pred_bin)
    U = np.logical_or(label_bin, pred_bin)
    return np.count_nonzero(I) / np.count_nonzero(U)


def main(_):
    '''
     Shortcuts
    '''
    ckpt_path = FLAGS.ckpt_path
    log_dir = FLAGS.log_dir
    batch_size = FLAGS.batch_size if FLAGS.mode == 'train' else 1
    num_classes = FLAGS.num_classes
    data_name = FLAGS.data_name

    '''
     Setting up the model
    '''
    dataset = TFRecordSegDataset(FLAGS.data_dir, data_name)
    data, num_samples = dataset.load_batch(FLAGS.mode, batch_size, FLAGS.height, FLAGS.width)

    # make synonyms for data
    images = data[0]
    labels = data[1]
    org_images = data[2]

    pred, logits = FCN8_atonce(images, num_classes)

    
    if FLAGS.mode == 'valid':
        saver = tf.train.Saver(slim.get_variables_to_restore())
        coord = tf.train.Coordinator()
        
        with tf.Session() as sess:
            '''
             Restore parameters from check point
            '''
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

            tf.train.start_queue_runners(sess, coord)
            
            eval_dir = os.path.join(ckpt_path, 'eval')
            if not tf.gfile.Exists(eval_dir):
                tf.gfile.MakeDirs(eval_dir)

            IOU = 0
            exp = int(np.log10(num_samples)) + 1
            
            time_per_image = time.time()
            for i in range(num_samples):
                r_images, r_labels, r_pred = sess.run([org_images, labels, pred])
                
                r_images = np.squeeze(r_images)
                r_labels = np.squeeze(r_labels)
                r_labels = r_labels.astype(np.uint8)
                r_pred = np.squeeze(r_pred)
                r_pred = r_pred.astype(np.uint8)

                IOU += calc_IOU(r_labels, r_pred, 1)
                
                res = r_images.shape;
                output = np.zeros((res[0], 3 * res[1], 3), dtype=np.uint8)

                r_labels = cv2.applyColorMap(r_labels * 100, cv2.COLORMAP_JET)
                r_pred = cv2.applyColorMap(r_pred * 100, cv2.COLORMAP_JET)

                r_images = 0.8 * r_images + 0.2 * r_pred

                output[:, 0*res[1]:1*res[1], :] = r_images
                output[:, 1*res[1]:2*res[1], :] = r_labels
                output[:, 2*res[1]:3*res[1], :] = r_pred

                cv2.imwrite(os.path.join(eval_dir, FLAGS.mode + str(i).zfill(exp) + '.png'), output)

            coord.request_stop()
            coord.join()
            
            time_per_image = (time.time() - time_per_image) / num_samples
            print('time elapsed: ' + str(time_per_image))
            
            IOU /= num_samples
            print('IOU for foreground: ' + str(IOU))

    elif FLAGS.mode == 'train':
        '''
         Define the loss function
        '''
        loss_xentropy = tf.losses.sparse_softmax_cross_entropy(tf.squeeze(labels), logits)
        total_loss = tf.losses.get_total_loss()

        '''
         Define summaries
        '''
        tf.summary.image('label', tf.cast(labels * 100, tf.uint8))
        tf.summary.image('pred', tf.cast(pred * 100, tf.uint8))
        tf.summary.image('image', images)
        tf.summary.scalar('loss_xentropy', loss_xentropy)

        '''
         Define initialize function
        '''
        if ckpt_path == 'vgg_16_160830.ckpt':
            exclude = ['vgg_16/fc8', 'FCN']
        else:
            exclude = []
        
        if ckpt_path.find('.ckpt') == -1:
            ckpt_path = tf.train.latest_checkpoint(ckpt_path);
        
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(ckpt_path, variables_to_restore, ignore_missing_vars = True)

        '''
         Define the learning rate
        '''
        num_batches_per_epoch = np.ceil(num_samples / FLAGS.batch_size)
        num_steps = FLAGS.num_epochs * num_batches_per_epoch

        if FLAGS.lr_decay:
            num_epochs_before_decay = 2
            decay_steps = num_epochs_before_decay * num_batches_per_epoch

            lr = tf.train.exponential_decay(learning_rate = FLAGS.learning_rate,
                                            global_step = tf.train.get_or_create_global_step(),
                                            decay_steps = decay_steps,
                                            decay_rate = FLAGS.lr_decay_rate,
                                            staircase = True)
        else:
            lr = FLAGS.learning_rate

        '''
         Define the optimizer
        '''
        if FLAGS.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=FLAGS.momentum)
        elif FLAGS.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        else:
            print('Unknown name of optimizer')
    
        '''
         Training phase
        '''
        if not tf.gfile.Exists(log_dir):
            tf.gfile.MakeDirs(log_dir)

        # generate a log to save hyper-parameter info
        with open(os.path.join(log_dir, 'info.txt'), 'w') as f:
            f.write('num_epochs: ' + str(FLAGS.num_epochs) + '\n')
            f.write('num_steps: ' + str(num_steps) + '\n')
            f.write('batch_size: ' + str(FLAGS.batch_size) + '\n')
            f.write('num_classes: ' + str(FLAGS.num_classes) + '\n')
            f.write('height: ' + str(FLAGS.height) + '\n')
            f.write('width: ' + str(FLAGS.width) + '\n')
            f.write('learning_rate: ' + str(FLAGS.learning_rate) + '\n')
            f.write('momentum: ' + str(FLAGS.momentum) + '\n')
            f.write('lr_decay_rate: ' + str(FLAGS.lr_decay_rate) + '\n')
            f.write('lr_decay: ' + str(FLAGS.lr_decay) + '\n')
            f.write('ckpt_path: ' + FLAGS.ckpt_path + '\n')
            f.write('data_dir: ' + FLAGS.data_dir + '\n')
            f.write('data_name: ' + FLAGS.data_name + '\n')
            f.write('mode: ' + FLAGS.mode + '\n')
            f.write('optimizer: ' + FLAGS.optimizer)

        train_op = slim.learning.create_train_op(total_loss, optimizer)

        final_loss = slim.learning.train(
                train_op = train_op,
                logdir = log_dir,
                init_fn = init_fn,
                number_of_steps = num_steps,
                summary_op = tf.summary.merge_all(),
                save_summaries_secs = 60)

        print('Finished training. Final batch loss %f' %final_loss)

    else:
        print('Unknown mode')

if __name__ == "__main__":
    tf.app.run()