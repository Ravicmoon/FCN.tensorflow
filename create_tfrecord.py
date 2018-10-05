import tensorflow as tf
import numpy as np
import argparse
import os

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _chunkify(list, num):
    return [list[i::num] for i in range(num)]

parser = argparse.ArgumentParser()
parser.add_argument('--img_folder', type=str, default='data/Cityscapes/images/validation')
parser.add_argument('--gt_folder', type=str, default='data/Cityscapes/annotations/validation')
parser.add_argument('--data_name', type=str, default='Cityscapes')
parser.add_argument('--mode', type=str, default='valid')
parser.add_argument('--num_splits', type=int, default=5)

args = parser.parse_args()

for _, _, files in os.walk(args.img_folder):
    img_files = files

for _, _, files in os.walk(args.gt_folder):
    gt_files = files

assert(len(img_files) == len(gt_files))

digits = int(np.log10(args.num_splits)) + 1

img_files = _chunkify(img_files, args.num_splits)
gt_files = _chunkify(gt_files, args.num_splits)
        
for i in range(args.num_splits):
            
    writer = tf.python_io.TFRecordWriter(args.data_name + '_' + args.mode + str(i).zfill(digits) + '.tfrecord')

    file_pairs = zip(img_files[i], gt_files[i])

    for img_file, gt_file in file_pairs:

        print('processing ' + img_file + ' file...')

        _, img_ext = os.path.splitext(img_file)
        _, gt_ext = os.path.splitext(gt_file)

        img_ext = bytes(img_ext[1:], 'UTF-8')
        gt_ext = bytes(gt_ext[1:], 'UTF-8')

        img_path = os.path.join(args.img_folder, img_file)
        gt_path = os.path.join(args.gt_folder, gt_file)
                
        img = tf.gfile.FastGFile(img_path, 'rb').read()
        gt = tf.gfile.FastGFile(gt_path, 'rb').read()
    
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': _bytes_feature(img),
            'image/format': _bytes_feature(img_ext),
            'gt/encoded': _bytes_feature(gt),
            'gt/format': _bytes_feature(gt_ext)}))
    
        writer.write(example.SerializeToString())

    writer.close()