import os
from .base import DataPreProcess

from .utils import check_dirs_exist

NEW_CNAMES_EURO = {
    "AnnualCrop": "Annual Crop Land",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation Land",
    "Highway": "Highway or Road",
    "Industrial": "Industrial Buildings",
    "Pasture": "Pasture Land",
    "PermanentCrop": "Permanent Crop Land",
    "Residential": "Residential Buildings",
    "River": "River",
    "SeaLake": "Sea or Lake",
}


class EuroSAT(DataPreProcess):

    dataset_dir = "eurosat"

    def __init__(self, dataset_path, seed):
        self.dataset_dir = dataset_path
        self.img_folder = "2750"
        self.image_dir = os.path.join(self.dataset_dir, self.img_folder)
        self.seed = seed

    def preprocess(self,):
        if check_dirs_exist(self.dataset_dir, ['train', 'val', 'test']):
            print('Dataset has already been preprocessed!')
            return
        else:
            self.split_train_val_test(self.dataset_dir, self.img_folder, self.seed,
                                                    train_ratio=0.8, val_ratio=0.1, )