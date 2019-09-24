#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import re
import numpy as np
import os
from .config import Params
import pandas as pd
from taputapu.io.image import get_image_shape_without_loading


def _convert_label_to_dense_codes(labels,
                                  split_char: str,
                                  max_width: int,
                                  table_str2int: dict):
    """

    :param labels:
    :param split_char:
    :param max_width:
    :param table_str2int:
    :return:
    """
    labels_chars = [[c for c in label.split(split_char) if c] for label in labels]
    codes_list = [[table_str2int[c] for c in list_char] for list_char in labels_chars]

    seq_lengths = [len(cl) for cl in codes_list]

    dense_codes = list()
    for ls in codes_list:
        dense_codes.append(ls + np.maximum(0, (max_width - len(ls))) * [0])

    return dense_codes, seq_lengths


def _compute_length_inputs(path, target_shape):

    w, h = get_image_shape_without_loading(path)
    ratio = w / h

    new_h = target_shape[0]
    new_w = np.minimum(new_h * ratio, target_shape[1])

    return new_w


def preprocess_csv(csv_filename: str,
                   parameters: Params,
                   output_csv_filename: str) -> int:
    """

    :param csv_filename:
    :param parameters:
    :param output_csv_filename:
    :return:
    """

    # Conversion table
    table_str2int = dict(zip(parameters.alphabet.alphabet_units, parameters.alphabet.codes))

    # Read file
    dataframe = pd.read_csv(csv_filename,
                            sep=parameters.csv_delimiter,
                            header=None,
                            names=['paths', 'labels'],
                            encoding='utf8',
                            escapechar="\\",
                            quoting=0)

    original_len = len(dataframe)

    dataframe['label_string'] = dataframe.labels.apply(lambda x: re.sub(re.escape(parameters.string_split_delimiter), '', x))
    dataframe['label_len'] = dataframe.label_string.apply(lambda x: len(x))

    # remove long labels
    dataframe = dataframe[dataframe.label_len <= parameters.max_chars_per_string]

    # Compute width images (after resizing)
    dataframe['input_length'] = dataframe.paths.apply(lambda x: _compute_length_inputs(x, parameters.input_shape))
    dataframe.input_length = dataframe.input_length.apply(lambda x: np.floor(x / parameters.n_pool))
    # Remove items with longer label than input
    dataframe = dataframe[dataframe.label_len < dataframe.input_length]

    final_length = len(dataframe)

    n_removed = original_len - final_length
    if n_removed > 0:
        print('-- Removed {} samples ({:.2f} %)'.format(n_removed,
                                                        100 * n_removed / original_len))

    # Convert fields to list
    paths = dataframe.paths.to_list()
    labels = dataframe.labels.to_list()

    # Convert string labels to dense codes
    label_dense_codes, label_seq_length = _convert_label_to_dense_codes(labels,
                                                                        parameters.string_split_delimiter,
                                                                        parameters.max_chars_per_string,
                                                                        table_str2int)
    # format in string to be easily parsed by tf.data
    string_label_codes = [[str(ldc) for ldc in list_ldc] for list_ldc in label_dense_codes]
    string_label_codes = [' '.join(list_slc) for list_slc in string_label_codes]

    data = {'paths': paths, 'label_codes': string_label_codes, 'label_len': label_seq_length}
    new_dataframe = pd.DataFrame(data)

    new_dataframe.to_csv(output_csv_filename,
                         sep=parameters.csv_delimiter,
                         header=False,
                         encoding='utf8',
                         index=False,
                         escapechar="\\",
                         quoting=0)
    return len(new_dataframe)


def data_preprocessing(params: Params) -> (str, str, str, str):
    """

    :param params:
    :return:
    """
    output_dir = os.path.join(params.output_model_dir, 'preprocessed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        'Output directory {} already exists'.format(output_dir)

    csv_train_output = os.path.join(output_dir, 'updated_train.csv')
    csv_eval_output = os.path.join(output_dir, 'updated_eval.csv')

    # Preprocess train csv
    n_samples_train = preprocess_csv(params.csv_files_train, params, csv_train_output)

    # Preprocess train csv
    n_samples_eval = preprocess_csv(params.csv_files_eval, params, csv_eval_output)

    return csv_train_output, csv_eval_output, n_samples_train, n_samples_eval
