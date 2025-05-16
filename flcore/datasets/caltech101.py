import os
import json
from .base import DataPreProcess

from .utils import check_dirs_exist


IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES_CALTECH = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}



class Caltech101(DataPreProcess):

    dataset_dir = "caltech-101"

    def __init__(self, dataset_path, seed):
        self.dataset_dir = dataset_path
        self.img_folder = "101_ObjectCategories"
        self.image_dir = os.path.join(self.dataset_dir, self.img_folder)
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Caltech101.json')
        self.seed = seed

    def preprocess(self,):
        if check_dirs_exist(self.dataset_dir, ['train', 'val', 'test']):
            print('Dataset has already been preprocessed!')
            return
        else:
            self.split_by_json()
