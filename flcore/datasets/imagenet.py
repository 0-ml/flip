import os
from .base import DataPreProcess

from .utils import check_dirs_exist


class ImageNet(DataPreProcess):
    """ImageNet
    """

    dataset_dir = "imagenet"

    def __init__(self, dataset_path, seed):
        self.dataset_dir = dataset_path
        self.img_folder = ""
        self.image_dir = os.path.join(self.dataset_dir, self.img_folder)
        self.seed = seed


    @property
    def real_class_names(self,):
        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        return self.read_classnames(text_file)
