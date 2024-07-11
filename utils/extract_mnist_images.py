r"""
File to extract csv images from csv files for mnist dataset.
"""

import os
import cv2
from tqdm import tqdm
import numpy as np
import _csv as csv

def extract_images(save_dir, csv_fname):
    assert os.path.exists(save_dir), "Directory {} to save images does not exist".format(save_dir)
    assert os.path.exists(csv_fname), "Csv file {} does not exist".format(csv_fname)
    with open(csv_fname) as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            im = np.zeros((784))
            im[:] = list(map(int, row[1:]))
            im = im.reshape((28,28))
            if not os.path.exists(os.path.join(save_dir, row[0])):
                os.mkdir(os.path.join(save_dir, row[0]))
            cv2.imwrite(os.path.join(save_dir, row[0], '{}.png'.format(idx)), im)
            if idx % 1000 == 0:
                print('Finished creating {} images in {}'.format(idx+1, save_dir))
            
            
if __name__ == '__main__':
    extract_images('data/train/images', 'data/mnist_train.csv')
    extract_images('data/test/images', 'data/mnist_test.csv')
