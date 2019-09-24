#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.metrics import Metric
from tensorflow.keras.backend import ctc_batch_cost, ctc_decode
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, MaxPool2D, Input, Permute, \
    Reshape, Bidirectional, LSTM, Dense, Softmax, Lambda
from typing import List, Tuple
from .config import Params


class ConvBlock(Layer):
    def __init__(self,
                 features: int,
                 kernel_size: int,
                 stride: Tuple[int, int],
                 cnn_padding: str,
                 pool_size: Tuple[int, int],
                 pool_strides: Tuple[int, int],
                 batchnorm: bool):
        super(ConvBlock, self).__init__()
        self.conv = Conv2D(features,
                           kernel_size,
                           strides=stride,
                           padding=cnn_padding)
        self.bn = BatchNormalization(renorm=True,
                                     renorm_clipping={'rmax': 1e2, 'rmin': 1e-1, 'dmax': 1e1},
                                     trainable=True) if batchnorm else None
        self.pool = MaxPool2D(pool_size=pool_size,
                              strides=pool_strides,
                              padding='same')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        if self.bn is not None:
            x = self.bn(x, training=training)
        if self.pool is not None:
            x = self.pool(x)
        x = tf.nn.relu(x)
        return x

    def get_config(self):
        config = super(ConvBlock, self).get_config()
        return config


class CERMetric(Metric):
    def __init__(self):
        super(CERMetric, self).__init__()

        self.distance = self.add_weight('distance')
        self.count_chars = self.add_weight('count_chars')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_pred needs to be decoded (its the logits)
        pred_codes_dense = ctc_decode(y_pred, pred_sequence_length, greedy=True)

        # create a sparse tensor
        idx = tf.where(tf.not_equal(pred_codes_dense, -1))
        pred_codes_sparse = tf.SparseTensor(idx, tf.gather_nd(pred_codes_dense, idx), pred_codes_dense.get_shape())

        distance = tf.reduce_sum(tf.edit_distance(pred_codes_sparse, y_true, normalize=False))
        self.distance.assign_add(distance)

        self.count_chars.assign_add(tf.reduce_sum(true_sequence_length))

    def result(self):
        return tf.divide(self.distance, self.count_chars) if tf.greater(self.count_chars, 0) else 1.0

    def reset_states(self):
        self.distance.assign(0)
        self.count_chars.assign(0)


def get_crnn_output(input_images, parameters: Params=None):

    # params of the architecture
    cnn_features_list = parameters.cnn_features_list
    cnn_kernel_size = parameters.cnn_kernel_size
    cnn_pool_size = parameters.cnn_pool_size
    cnn_pool_strides = parameters.cnn_pool_strides
    cnn_stride_size = parameters.cnn_stride_size
    cnn_batch_norm = parameters.cnn_batch_norm
    rnn_units = parameters.rnn_units

    # CNN layers
    cnn_params = zip(cnn_features_list, cnn_kernel_size, cnn_stride_size, cnn_pool_size,
                     cnn_pool_strides, cnn_batch_norm)
    conv_layers = [ConvBlock(ft, ks, ss, 'same', psz, pst, bn) for ft, ks, ss, psz, pst, bn in cnn_params]

    x = conv_layers[0](input_images)
    for conv in conv_layers[1:]:
        x = conv(x)

    # Permutation and reshape
    x = Permute((2, 1, 3))(x)
    shape = x.get_shape().as_list()
    x = Reshape((shape[1], shape[2] * shape[3]))(x)  # [B, W, H*C]

    # RNN layers
    rnn_layers = [Bidirectional(LSTM(ru, dropout=0.5, return_sequences=True, time_major=False)) for ru in
                  rnn_units]
    for rnn in rnn_layers:
        x = rnn(x)

    # Dense and softmax
    x = Dense(parameters.alphabet.n_classes)(x)
    net_output = Softmax(name='softmax_output')(x)

    return net_output


def get_model_train(parameters: Params,
                    file_writer=None):

    h, w = parameters.input_shape
    c = parameters.input_channels

    input_images = Input(shape=(h, w, c), name='input_images')
    input_seq_len = Input(shape=[1], dtype=tf.int32, name='input_seq_length')

    # if file_writer:
    #     with file_writer.as_default():
    #         tf.summary.image('augmented data', input_images, max_outputs=2)

    label_codes = Input(shape=(parameters.max_chars_per_string), dtype=tf.int32, name='label_codes')
    label_seq_length = Input(shape=[1], dtype='int64', name='label_seq_length')

    net_output = get_crnn_output(input_images, parameters)

    # Loss function
    def warp_ctc_loss(y_true, y_pred):
        return ctc_batch_cost(label_codes, y_pred, input_seq_len, label_seq_length)

    # tf.summary.scalar('loss', tf.reduce_mean(loss_ctc))

    # Metric function
    def warp_cer_metric(y_true, y_pred):
        pred_sequence_length, true_sequence_length = input_seq_len, label_seq_length

        # y_pred needs to be decoded (its the logits)
        pred_codes_dense = ctc_decode(y_pred, tf.squeeze(pred_sequence_length, axis=-1), greedy=True)
        pred_codes_dense = tf.squeeze(tf.cast(pred_codes_dense[0], tf.int64), axis=0)  # only [0] if greedy=true

        # create sparse tensor
        idx = tf.where(tf.not_equal(pred_codes_dense, -1))
        pred_codes_sparse = tf.SparseTensor(tf.cast(idx, tf.int64),
                                            tf.gather_nd(pred_codes_dense, idx),
                                            tf.cast(tf.shape(pred_codes_dense), tf.int64))

        idx = tf.where(tf.not_equal(label_codes, 0))
        label_sparse = tf.SparseTensor(tf.cast(idx, tf.int64),
                                       tf.gather_nd(label_codes, idx),
                                       tf.cast(tf.shape(label_codes), tf.int64))
        label_sparse = tf.cast(label_sparse, tf.int64)

        # Compute edit distance and total chars count
        distance = tf.reduce_sum(tf.edit_distance(pred_codes_sparse, label_sparse, normalize=False))
        count_chars = tf.reduce_sum(true_sequence_length)

        return tf.divide(tf.cast(distance, tf.int64), count_chars, name='CER')

    # Define model and compile it
    model = Model(inputs=[input_images, label_codes, input_seq_len, label_seq_length], outputs=net_output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=parameters.learning_rate)
    model.compile(loss=[warp_ctc_loss],
                  optimizer=optimizer,
                  metrics=[warp_cer_metric],
                  experimental_run_tf_function=False)

    return model


def get_model_inference(parameters: Params,
                        weights_dir: str=None):
    h, w = parameters.input_shape
    c = parameters.input_channels

    input_images = Input(shape=(h, w, c), name='input_images')
    input_seq_len = Input(shape=[1], dtype=tf.int32, name='input_seq_length')

    net_output = get_crnn_output(input_images, parameters)
    output_seq_len = tf.identity(input_seq_len)  # need this op to pass it to output

    model = Model(inputs=[input_images, input_seq_len], outputs=[net_output, output_seq_len])

    if weights_dir:
        model.load_weights(weights_dir)

    return model