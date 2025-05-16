import os

from scipy.io import loadmat
from .base import DataPreProcess, Datum
from .oxford_pets import OxfordPets
from .utils import check_dirs_exist

class StanfordCars(DataPreProcess):

    dataset_dir = "stanford_cars"

    def __init__(self, dataset_path, seed):
        self.dataset_dir = dataset_path
        self.img_folder = ""
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_StanfordCars.json')
        self.seed = seed
        self.train_dir = os.path.join(self.dataset_dir, 'cars_train')
        self.test_dir = os.path.join(self.dataset_dir, 'cars_test')

    def preprocess(self, ):
        if check_dirs_exist(self.dataset_dir, ['train', 'val', 'test']):
            print('Dataset has already been preprocessed, skip...')
            return
        else:
            self.split_by_json()