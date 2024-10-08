import os
import torch
import numpy as np
import PIL.Image as Image

from .utils import check_dirs_exist
from torch.utils.data import Dataset



class CoCo2014(Dataset):

    dataset_dir = "coco2014"

    def __init__(self, dataset_path, split, transform, seed):
        self.dataset_dir = dataset_path
        self.img_folder = ""
        self.image_dir = os.path.join(self.dataset_dir, self.img_folder)
        self.split = split
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.transform = transform
        self.seed = seed

    def preprocess(self,):
        if check_dirs_exist(self.dataset_dir, ['train', 'val', 'test']):
            print('Dataset has already been preprocessed!')
            return
        else:
            raise NotImplementedError

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask