import os
import re

from .base import DataPreProcess, Datum

from .utils import check_dirs_exist

class UCF(DataPreProcess):

    dataset_dir = "ucf"

    def __init__(self, dataset_path, seed):
        self.dataset_dir = dataset_path
        self.img_folder = "UCF-101-midframes"
        self.image_dir = os.path.join(self.dataset_dir, self.img_folder)
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_UCF101.json")
        self.seed = seed

    def preprocess(self,):
        if check_dirs_exist(self.dataset_dir, ['train', 'val', 'test']):
            print('Dataset has already been preprocessed!')
            return
        else:
            self.split_by_json(lower=True)