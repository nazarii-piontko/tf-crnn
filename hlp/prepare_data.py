#!/usr/bin/env python

__author__ = "Nazarii Piontko"
__license__ = "MIT"

import os, sys
import re
from glob import glob

import cv2
import click
import pandas as pd
from sklearn.model_selection import train_test_split

from alphabet_helpers import generate_alphabet_file
from string_data_manager import tf_crnn_label_formatting


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

@click.command()
@click.option('--download_dir')
@click.option('--generated_data_dir')
def prepare_data(download_dir: str,
                 generated_data_dir: str):

    print('Generating files for the experiment...')

    mkdir(generated_data_dir)

    raw_df = pd.read_csv(os.path.join(download_dir, 'trainVal.csv'))
    gen_df = pd.DataFrame(columns=['img_url', 'label'])

    for i, row in raw_df.iterrows():
        if i % 1000 == 0:
            print("Processed {} rows".format(i))

        src_file = os.path.join(download_dir, row['image_path'])
        # dst_file = os.path.join(generated_data_dir, row['lp'] + '.jpg')

        # img = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)
        # cv2.imwrite(dst_file, img)

        src_file = src_file.replace('\\', '/')
        label = '|' + '|'.join(list(row['lp'].strip())) + '|'

        # gen_df.loc[i] = [dst_file, label]
        gen_df.loc[i] = [src_file, label]

    print('Saving CSVs...')

    train_df, test_df = train_test_split(gen_df, test_size=0.2)

    test_df, val_df = train_test_split(test_df, test_size=0.5)

    train_df.to_csv(os.path.join(generated_data_dir, 'train.csv'),
                    sep=';',
                    encoding='utf-8',
                    header=False,
                    index=False)
    test_df.to_csv(os.path.join(generated_data_dir, 'test.csv'),
                   sep=';',
                   encoding='utf-8',
                   header=False,
                   index=False)
    val_df.to_csv(os.path.join(generated_data_dir, 'val.csv'),
                  sep=';',
                  encoding='utf-8',
                  header=False,
                  index=False)

    print('Format string label to tf_crnn formatting...')

    for csv_filename in glob(os.path.join(generated_data_dir, '*.csv')):
        tf_crnn_label_formatting(csv_filename)

    print('Generating alphabet...')

    generate_alphabet_file(glob(os.path.join(generated_data_dir, '*.csv')),
                           os.path.join(generated_data_dir, 'alphabet_lookup.json'))

    return 0


if __name__ == '__main__':
    sys.exit(prepare_data())
