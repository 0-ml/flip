import os
from .base import DataPreProcess

from .utils import check_dirs_exist


class ImageNetR(DataPreProcess):
    """ImageNet-R(endition).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-rendition"

    def __init__(self, dataset_path, seed):
        self.dataset_dir = dataset_path

    @property
    def class_names(self,):
        """ folder names of each class
        """
        test_dir = os.path.join(self.dataset_dir, 'test')
        return list(os.listdir(test_dir))

    @property
    def real_class_names(self,):
        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        return self.read_classnames(text_file)