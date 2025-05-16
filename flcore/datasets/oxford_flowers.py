import os
import random
from collections import defaultdict
from scipy.io import loadmat

from .base import DataPreProcess, Datum

from .utils import check_dirs_exist, read_json


class OxfordFlowers(DataPreProcess):

    dataset_dir = "oxford_flowers"

    def __init__(self, dataset_path, seed):
        self.dataset_dir = dataset_path
        self.img_folder = "jpg"
        self.image_dir = os.path.join(self.dataset_dir, self.img_folder)
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_OxfordFlowers.json')
        self.label_file = os.path.join(self.dataset_dir, "imagelabels.mat")
        self.lab2cname_file = os.path.join(self.dataset_dir, "cat_to_name.json")
        self.seed = seed
        self.train_ratio = 0.5
        self.val_ratio = 0.2

    def preprocess(self,):
        if check_dirs_exist(self.dataset_dir, ['train', 'val', 'test']):
            print('Dataset has already been preprocessed!')
            return
        else:
            self.split_by_json()