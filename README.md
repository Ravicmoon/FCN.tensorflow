# tf.FCN
Tensorflow implementation of [Fully Convolutional Networks for Semantic Segmentation](http://arxiv.org/pdf/1605.06211v1.pdf) (FCNs). 
![Fig1](/FCN-8s_VGG-16.png "Structure of FCN-8s based on VGG-16")

There are two objectives of this project.
1. Mimic [original Caffe implementation](https://github.com/shelhamer/fcn.berkeleyvision.org)
2. Use high-level API of Tensorflow

## Prerequisites
1. Python 3.6
2. Tensorflow rc1.10 or above
3. Numpy
4. Pillow
5. [Pretrained VGG-16 model](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)

## Preparing dataset
Our implementation uses _tfrecord_ files to efficiently manage the dataset. Note that you can generate _tfrecord_ files of the dataset by using the script, **create_tf_record.py**. Example of the usage is as follows.
```
python create_tf_record.py --img_folder=train/images, gt_folder=train/annotations, data_name=Cityscapes, --mode=train, --num_splits=5
```
After running the script, you can find five _tfrecord_ files in the working directory. To generate _tfrecord_ files for validation, you need to run the script, **create_tf_record.py** with `--mode=valid` option.

## Training and evaluation
We supports momentum and Adam optimizers to train the FCN. Original implementation uses the momentum optimizer but during our experiments, we found that Adam optimizer is better than the momentum in terms of training speed and quality. Also, you can control number of iterations, batch size, number of output labels, and learning rate. You can use a **trainval.py** script to both train and evaluate the FCN. Example of the usage is as follows.
```
python trainval.py --num_steps=100000 --batch_size=2 --num_classes=3 --learning_rate=2e-4 --lr_decay_rate=0.99 --lr_decay=True --ckpt_path=vgg_16_160830.ckpt --log_dir=ckpt_180917 --data_dir=data --data_name=Cityscapes --mode=train --optimizer=Adam
```
To evaluate the FCN, you need to use the `--mode=valid` option. In that case, other options are ignored except for `--log_dir`, `--data_dir`, and `--data_name`. Currently, evaluation code is not generalized enough and only designed for our application, road segmentation.

## Important note
Currently, this project is design to segment road regions from the scene; default number of classes is three (0: don't care, 1: background, 2: road). We use a _don't care_ label to distinguish road regions on oppossite direction, where ego-vehicle cannot drive. It is worth mentioning that we focus on visualizing outputs of three labels. If you want to visualize results of more than three labels, you need a slight customization of the code.
