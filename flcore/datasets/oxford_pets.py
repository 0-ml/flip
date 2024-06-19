import os
import random
import math
import os.path as osp
import shutil
from collections import defaultdict

from .base import DataPreProcess, Datum
from .utils import check_dirs_exist, read_json


class OxfordPets(DataPreProcess):

    dataset_dir = "oxford_pets"

    def __init__(self, dataset_path, seed):
        self.dataset_dir = dataset_path
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.img_folder = "images"
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")
        self.seed = seed

    def preprocess(self, ):
        if check_dirs_exist(self.dataset_dir, ['train', 'val', 'test']):
            print('Dataset has already been preprocessed, skip...')
            return
        else:
            self.split_by_json()