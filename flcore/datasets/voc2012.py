import os
import random
import math

from .base import BaseSegDataset
from .info import INFO

class VOC2012(BaseSegDataset):
    def __init__(
        self, dataset_dir, split, transform, n_sample=-1, seed=0):
        super(VOC2012, self).__init__(dataset_dir, split)
        self.dataset_dir = dataset_dir
        self.img_folder = 'VOCdevkit/VOC2012/JPEGImages'
        self.label_folder = 'VOCdevkit/VOC2012/SegmentationClass'
        self.class_names_path = 'VOCdevkit/VOC2012/ImageSets/Segmentation/class_names.txt'
        self.image_dir = os.path.join(self.dataset_dir, self.img_folder)
        self.label_dir = os.path.join(self.dataset_dir, self.label_folder)
        self.transform = transform
        random.seed(seed)
        if n_sample > 0 and len(self.list_sample) >= n_sample and split == "train":
            self.samples = random.sample(self.list_sample, n_sample)
        else:
            self.samples = self.list_sample
        self.real_class_names = self.read_class_names()
        self.classes = self.read_class_names()

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.dataset_dir, self.samples[index][0])
        label_path = os.path.join(self.dataset_dir, self.samples[index][1])
        image = self.img_loader(image_path, "RGB")
        label = self.img_loader(label_path, "L")
        image, label = self.transform(image, label)
        return image[0], label[0, 0].long()

    def __len__(self):
        return len(self.samples)

    def read_class_names(self, ):
        path = os.path.join(self.dataset_dir, self.class_names_path)
        class_names = [line.strip() for line in open(path, 'r')]
        return class_names
