# coding=utf-8
"""Implementation of SAEA-SI-TI-DIM Attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
import pandas as pd
import scipy.stats as st
from imageio import imread, imsave
from tensorflow.contrib.image import transform as images_transform
from tensorflow.contrib.image import rotate as images_rotate
import tensorflow as tf
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
import random

slim = tf.contrib.slim
tf.flags.DEFINE_integer('batch_size', 4, 'How many images process at one time.')
tf.flags.DEFINE_float('max_epsilon', 16.0, 'max epsilon.')
tf.flags.DEFINE_integer('num_iter', 10, 'max iteration.')
tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')
tf.flags.DEFINE_integer('m_saea', 32, 'hyper-parameter of saea, inner iterative number')
tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_float('prob', 0.5, 'probability of using diverse inputs.')
tf.flags.DEFINE_integer('image_resize', 331, 'Height of each input images.')
tf.flags.DEFINE_string('checkpoint_path', './models', 'Path to checkpoint for pretained models.')
tf.flags.DEFINE_string('input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string('output_dir', '', 'Output directory with images.')
tf.flags.DEFINE_integer('seed', 1000, 'seed num')


tf.flags.DEFINE_integer('seed', 1000, 'seed num')

FLAGS = tf.flags.FLAGS
np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)
random.seed(FLAGS.seed)

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2.ckpt'),
    'resnet_v2': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt')}


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


kernel = gkern(7, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, pilmode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def deepconv_graph(noiseinput):
    noiseoutput=tf.nn.depthwise_conv2d(noiseinput, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
    return noiseoutput


def single_graph_inc_v3(x, one_hot, num_classes = 1001):
    x_nes = x
    x_batch = tf.concat([x_nes, x_nes/2., x_nes/4., x_nes/8., x_nes/16.], axis=0)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_inc_v3, end_points_inc_v3 = inception_v3.inception_v3(
            input_diversity(x_batch), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    logits, auxlogits = logits_inc_v3, end_points_inc_v3['AuxLogits']
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits, label_smoothing=0.0, weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot, auxlogits, label_smoothing=0.0, weights=0.4)

    noise = tf.reduce_sum(tf.split(tf.gradients(cross_entropy, x_batch)[0], 5) * tf.constant([1, 1/2., 1/4., 1/8., 1/16.])[:, None, None, None, None], axis=0)

    return noise


def single_graph_inc_v4(x, one_hot, num_classes = 1001):
    x_nes = x
    x_batch = tf.concat([x_nes, x_nes/2., x_nes/4., x_nes/8., x_nes/16.], axis=0)
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_inc_v4, end_points_inc_v4 = inception_v4.inception_v4(
            input_diversity(x_batch), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    logits, auxlogits = logits_inc_v4, end_points_inc_v4['AuxLogits']
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits, label_smoothing=0.0, weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot, auxlogits, label_smoothing=0.0, weights=0.4)

    noise = tf.reduce_sum(tf.split(tf.gradients(cross_entropy, x_batch)[0], 5) * tf.constant([1, 1/2., 1/4., 1/8., 1/16.])[:, None, None, None, None], axis=0)

    return noise


def single_graph_inc_res(x, one_hot, num_classes = 1001):
    x_nes = x
    x_batch = tf.concat([x_nes, x_nes/2., x_nes/4., x_nes/8., x_nes/16.], axis=0)
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_inc_res, end_points_inc_res = inception_resnet_v2.inception_resnet_v2(
            input_diversity(x_batch), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    logits, auxlogits = logits_inc_res, end_points_inc_res['AuxLogits']
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits, label_smoothing=0.0, weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot, auxlogits, label_smoothing=0.0, weights=0.4)

    noise = tf.reduce_sum(tf.split(tf.gradients(cross_entropy, x_batch)[0], 5) * tf.constant([1, 1/2., 1/4., 1/8., 1/16.])[:, None, None, None, None], axis=0)

    return noise


def single_graph_res_v2(x, one_hot, num_classes = 1001):
    x_nes = x
    x_batch = tf.concat([x_nes, x_nes/2., x_nes/4., x_nes/8., x_nes/16.], axis=0)
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_res_v2, end_points_res_v2 = resnet_v2.resnet_v2_101(
            input_diversity(x_batch), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    logits = logits_res_v2
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits, label_smoothing=0.0, weights=1.0)

    noise = tf.reduce_sum(tf.split(tf.gradients(cross_entropy, x_batch)[0], 5) * tf.constant([1, 1/2., 1/4., 1/8., 1/16.])[:, None, None, None, None], axis=0)

    return noise


def image_augmentation(x):
    # img, noise
    one = tf.fill([tf.shape(x)[0], 1], 1.)
    zero = tf.fill([tf.shape(x)[0], 1], 0.)
    transforms = tf.concat([one, zero, zero, zero, one, zero, zero, zero], axis=1)
    rands = tf.concat([tf.truncated_normal([tf.shape(x)[0], 6], stddev=0.05), zero, zero], axis=1)
    return images_transform(x, transforms + rands, interpolation='BILINEAR')


def image_rotation(x):
    """ imgs, scale, scale is in radians """
    rands = tf.truncated_normal([tf.shape(x)[0]], stddev=0.05)
    return images_rotate(x, rands, interpolation='BILINEAR')


def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret



def main(_):
    f2l = load_labels('./dataset/val_rs.csv')
    eps = 2 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    beta = alpha
    momentum = FLAGS.momentum
    m_saea = FLAGS.m_saea
    num_classes = 1001
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    check_or_create_dir(FLAGS.output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        y_input = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        one_hot = tf.concat([tf.one_hot(y_input, num_classes)] * 5, axis=0)

        grad_inc_v3 = single_graph_inc_v3(x_input, one_hot)
        grad_inc_v4 = single_graph_inc_v4(x_input, one_hot)
        grad_inc_res = single_graph_inc_res(x_input, one_hot)
        grad_res_v2 = single_graph_res_v2(x_input, one_hot)

        noise_input = tf.placeholder(tf.float32, shape=batch_shape)
        noise_output = deepconv_graph(noise_input)
        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            s2.restore(sess, model_checkpoint_map['inception_v4'])
            s3.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            s4.restore(sess, model_checkpoint_map['resnet_v2'])

            idx = 0
            l2_diff = 0
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                idx = idx + 1
                print("start the i={} attack".format(idx))

                labels = []
                for filename in filenames:
                    labels.append(f2l[filename])

                x = np.copy(images)
                x_min = np.clip(x - eps, -1.0, 1.0)
                x_max = np.clip(x + eps, -1.0, 1.0)

                grad = np.zeros(shape=batch_shape, dtype=np.float32)
                for i in range(num_iter):
                    noise_inc_v3 = sess.run(grad_inc_v3, feed_dict={x_input: x, y_input: labels})
                    noise_res_v2 = sess.run(grad_res_v2, feed_dict={x_input: x, y_input: labels})
                    noise_inc_res = sess.run(grad_inc_res, feed_dict={x_input: x, y_input: labels})
                    noise_inc_v4 = sess.run(grad_inc_v4, feed_dict={x_input: x, y_input: labels})
                    x_inner = np.copy(x)
                    x_before = np.copy(x)
                    noise_inner_all = np.zeros([m_saea, *x.shape])
                    grad_inner = np.zeros_like(x)
                    for j in range(m_saea):
                        grad_single = np.random.choice([grad_inc_v4, grad_inc_res, grad_res_v2])
                        if grad_single == grad_inc_v3:
                            noise_inc_v3 = sess.run(grad_single, feed_dict={x_input: x_inner, y_input: labels})
                        elif grad_single == grad_inc_v4:
                            noise_inc_v4 = sess.run(grad_single, feed_dict={x_input: x_inner, y_input: labels})
                        elif grad_single == grad_inc_res:
                            noise_inc_res = sess.run(grad_single, feed_dict={x_input: x_inner, y_input: labels})
                        else:
                            noise_res_v2 = sess.run(grad_single, feed_dict={x_input: x_inner, y_input: labels})

                        noise_ensemble = (noise_inc_v3 + noise_inc_v4 + noise_inc_res + noise_res_v2 ) / 4
                        noise_ensemble = sess.run(noise_output, feed_dict={noise_input: noise_ensemble})
                        noise_ensemble = noise_ensemble / np.mean(np.abs(noise_ensemble), (1, 2, 3), keepdims=True)
                        grad_inner = momentum * grad_inner + noise_ensemble
                        x_inner = x_inner + beta * np.sign(grad_inner)
                        x_inner = np.clip(x_inner, x_min, x_max)
                        
                        noise_inner_all[j] = np.copy(grad_inner)
                    

                    noise = np.copy(noise_inner_all[-1])

                    noise=sess.run(noise_output, feed_dict={noise_input: noise})
                    noise = noise / np.mean(np.abs(noise), (1, 2, 3), keepdims=True)
                    grad = momentum * grad + noise

                    x = x_before + alpha * np.sign(grad)
                    x = np.clip(x, x_min, x_max)

                save_images(x, filenames, FLAGS.output_dir)

                diff = (x + 1) / 2 * 255 - (images + 1) / 2 * 255
                l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1))

            print('{:.2f}'.format(l2_diff * FLAGS.batch_size / 1000))


def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l


if __name__ == '__main__':
    tf.app.run()
