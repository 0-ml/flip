import os
import pickle
import torch
import torchvision as tv
import functools
import math
import shutil
import copy
import numpy as np
from .utils import check_dirs_exist

from .utils import (ImageSubset, SegSubset, split_dataset, update_class_names,
                    split_seg_dataset)
import torchvision.transforms as transforms
from .info import INFO
from .randtransform import RandTransform
# preprocess datasets
from .oxford_pets import OxfordPets
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .eurosat import EuroSAT
from .fgvc_aircraft import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .ucf import UCF
from .sun397 import SUN397
from .stanford_cars import StanfordCars
from .imagenet import ImageNet
from .imagenet_a import ImageNetA
from .imagenet_r import ImageNetR
from .imagenet_sketch import ImageNetSketch
from .imagenetv2 import ImageNetV2
from .inaturalist import iNaturalist
from .domain_net import DomainNet
from .tiny_imagenet import TinyImageNet

from .caltech101 import NEW_CNAMES_CALTECH
from .eurosat import NEW_CNAMES_EURO

from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from .voc2012 import VOC2012
from .cityscapes import CityScapes
from .segaugment import seg_augment

SegDatasets = {
    'voc2012': VOC2012,
    'cityscapes': CityScapes,
}

replace_dicts = {
    'caltech101': NEW_CNAMES_CALTECH,
    'dtd': {},
    'oxford_pets': {},
    'eurosat': NEW_CNAMES_EURO,
    'fgvc_aircraft': {},
    'food101': {},
    'oxford_flowers': {},
    'ucf': {},
    'sun397': {},
    'stanford_cars': {},
    'imagenet': {},
    'inaturalist': {},
    'imagenet_a': {},
    'imagenet_r': {},
    'imagenet_s': {},
    'imagenetv2': {},
    'tiny_imagenet': {},
    'domain_net': {},
}


preprocess_map = {
    'caltech101': Caltech101,
    'dtd': DescribableTextures,
    'oxford_pets': OxfordPets,
    'eurosat': EuroSAT,
    'fgvc_aircraft': FGVCAircraft,
    'food101': Food101,
    'oxford_flowers': OxfordFlowers,
    'ucf': UCF,
    'sun397': SUN397,
    'stanford_cars': StanfordCars,
    'imagenet': ImageNet,
    'imagenet_a': ImageNetA,
    'imagenet_r': ImageNetR,
    'imagenet_s': ImageNetSketch,
    'imagenetv2': ImageNetV2,
    'tiny_imagenet': TinyImageNet,
    'inaturalist': iNaturalist,
    'domain_net': DomainNet,
    'clipart': DomainNet,
    'infograph': DomainNet,
    'painting': DomainNet,
    'quickdraw': DomainNet,
    'real': DomainNet,
    'sketch': DomainNet,

}

multi_domain_info = {
    'domain_net': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
    'office-home': [],
}


def split_base_new(dataset_path, class_names):
    if check_dirs_exist(dataset_path, ['train_base', 'train_new',
                                    'val_base', 'val_new',
                                    'test_base', 'test_new']):
        print('Base and New datasets have been splited, skip...')
        return
    class_names.sort()
    m = math.ceil( len(class_names) / 2)
    partitions ={'base': class_names[:m],
                'new': class_names[m:] }
    for subset in ['train', 'val', 'test']:
        for k, v in partitions.items():
            subset_dir = os.path.join(dataset_path, f'{subset}_{k}')
            os.makedirs(subset_dir, exist_ok=False)
            for cls_name in v:
                source_dir = os.path.join(dataset_path, subset, cls_name)
                target_dir = os.path.join(dataset_path, f'{subset}_{k}', cls_name)
                shutil.copytree(source_dir, target_dir)
    print('Dataset Base/New split finished!')

def augment(name, train, data_transform):
    info = INFO[name]
    mean, std = info['moments']
    if not train:
        crop_size = info['shape'][-1]
        transform_test = transforms.Compose([
        tv.transforms.Resize(crop_size, interpolation=BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
        return transform_test
    else:
        augments = []
        crop_size = info['shape'][-1]
        augments.append(transforms.RandomResizedCrop(crop_size, interpolation=BICUBIC))
        augments.append(tv.transforms.RandomHorizontalFlip())
        augments.append(tv.transforms.ToTensor())
        # augments.append(tv.transforms.Normalize(*info['moments']))
        if data_transform == 'default':
            return tv.transforms.Compose(augments)
        if data_transform == 'random':
            resize = 240
            return RandTransform(mean, std, crop_size, resize)

def split_fewshot_subset(train_dataset, num_shot, split):
    """ split a fewshot subset
    """
    num_classes = len(train_dataset.classes)
    data_indices = {} # the data indices for entire trainset
    classes_idxs = list(range(num_classes))
    for j in classes_idxs:
        data_indices[j] = [i for i, label in
                        enumerate(train_dataset.targets) if label == j]
    stats = [len(v) for _,v in data_indices.items()]
    print(f'dataset statistics: {stats}')
    for cls_idx, img_ids in data_indices.items():
        np.random.shuffle(img_ids)
    if split == 'train':
        idx_train, idx_del = [], []
        for cls_idx, img_ids in data_indices.items():
            idx_train.extend(img_ids[:num_shot])
            idx_del.extend(img_ids[num_shot:])
        train_data = []
        train_data = copy.deepcopy(train_dataset)
        train_data.imgs = np.delete(train_dataset.imgs, idx_del, axis=0)
        train_data.samples = np.delete(train_dataset.samples, idx_del, axis=0)
        train_data.targets = np.delete(train_dataset.targets, idx_del, axis=0)
        train_data.data_indices = np.array(idx_train)
        return train_data
    elif split == 'personal':
        # test set for personalized FL
        idx_test, idx_del = [], []
        for cls_idx, img_ids in data_indices.items():
            idx_test.extend(img_ids[num_shot: 2*num_shot])
            idx_del.extend(img_ids[:num_shot])
            idx_del.extend(img_ids[2*num_shot:])
        test_data = []
        test_data = copy.deepcopy(train_dataset)
        test_data.imgs = np.delete(train_dataset.imgs, idx_del, axis=0)
        test_data.samples = np.delete(train_dataset.samples, idx_del, axis=0)
        test_data.targets = np.delete(train_dataset.targets, idx_del, axis=0)
        test_data.data_indices = np.array(idx_test)
        return test_data


def init_dataset(bench, data_dir, name, sub_domain_dir, split, data_transform, seed,
                 num_shot=-1, subsample=None, multi_domain=False):
    dataset_path = os.path.join(data_dir, name)
    train = split == 'train'
    if multi_domain:
        PreDataset = preprocess_map[name](dataset_path, seed, sub_domain_dir)
    else:
        PreDataset = preprocess_map[name](dataset_path, seed)
    PreDataset.preprocess()
    class_names = PreDataset.class_names
    real_class_names = PreDataset.real_class_names
    if bench == 'base2novel' and subsample is not None:
        split_base_new(dataset_path, class_names)
        splitset = 'train_base' if train else f'test_{subsample}'
    else:
        splitset = split
    if split == 'personal':
        splitset = 'train'
    if multi_domain:
        data_path = os.path.join(data_dir, name, f'{sub_domain_dir}_{splitset}')
    else:
        data_path = os.path.join(data_dir, name, splitset)
    transforms = augment(name, train, data_transform)
    dataset = tv.datasets.ImageFolder(data_path, transform = transforms)
    dataset.real_class_names = real_class_names
    if split == 'train' or split == 'personal':
        if num_shot > 0:
            dataset = split_fewshot_subset(dataset, num_shot, split)
    return dataset

def ImageLoader(
        bench, name, split, batch_size, num_clients, num_shards, num_shot=16,
        split_mode='dirichlet', parallel=False, alpha=0.5, beta=2, data_dir=None,
        img_folder= 'images', data_transform='default', drop_last=False, num_workers=8,
        subsample=None, seed=0, multi_domain=False):

    dataset = init_dataset(bench, data_dir, name, img_folder, split, data_transform, seed,
                           num_shot, subsample, multi_domain)

    replace_dict = replace_dicts[name]
    dataset = update_class_names(dataset, replace_dict)
    kwargs = {'drop_last': drop_last}
    if parallel:
        kwargs = {'pin_memory': False,
                  'num_workers': num_workers,
                  'drop_last': drop_last, }
    Loader = functools.partial(torch.utils.data.DataLoader, **kwargs)
    if split == 'test' or split == 'val':
        if multi_domain:
            return [Loader(ImageSubset(dataset), batch_size, False)]
        else:
            return Loader(ImageSubset(dataset), batch_size, False)
    dataloaders = []
    splits, stats = split_dataset(name, dataset, split_mode, num_clients,
                            num_shards, alpha, beta, batch_size,
                            drop_last, seed)
    shuffle = split == 'train'

    for c in range(num_clients):
        loader = Loader(splits[c], batch_size, shuffle=shuffle)
        loader.stats = stats[c]
        dataloaders.append(loader)
    return dataloaders


def MultiDomainImageLoader(bench, name, split, batch_size, num_clients, num_shards, num_shot=16,
        split_mode='dirichlet', parallel=False, alpha=0.5, beta=2, data_dir=None,
        img_folder= 'images', data_transform='default', drop_last=False, num_workers=8,
        subsample=None, seed=0):

    num_clients = 2 # data of each domain is assigned to two clients
    dataloaders = []
    for domain_name in multi_domain_info[name]:
        domainloaders = ImageLoader(
        bench, name, split, batch_size, num_clients, num_shards, num_shot,
        split_mode, parallel, alpha, beta, data_dir,
        domain_name, data_transform, drop_last, num_workers,
        subsample, seed, multi_domain=True)
        dataloaders.extend(domainloaders)
    if split == 'test' or split == 'val':
        dataloaders = {dname: dloader for dname, dloader
                            in zip(multi_domain_info[name], dataloaders)}
    return dataloaders

def init_seg_dataset(data_dir, name, split, seed, num_shot, num_clients):
    dataset_folder = os.path.join(data_dir, name)
    transforms = seg_augment(name, split)
    n_sample = num_shot * num_clients if num_shot > 0 else -1
    dataset = SegDatasets[name](dataset_folder, split, transforms,
                                            n_sample=n_sample, seed=seed)
    return dataset


def SegLoader(
        bench, name, split, batch_size, num_clients, num_shards, num_shot=16,
        split_mode='dirichlet', parallel=False, alpha=0.5, beta=2, data_dir=None,
        img_folder= 'images', data_transform='default', drop_last=False, num_workers=8,
        subsample=None, seed=0, multi_domain=False):
    dataset = init_seg_dataset(data_dir, name, split, seed, num_shot, num_clients)
    kwargs = {'drop_last': drop_last}
    if parallel:
        kwargs = {'pin_memory': False,
                  'num_workers': num_workers,
                  'drop_last': drop_last, }
    Loader = functools.partial(torch.utils.data.DataLoader, **kwargs)
    if split == 'test' or split == 'val':
        return Loader(SegSubset(dataset), batch_size, False)

    dataloaders = []
    splits = split_seg_dataset(dataset, num_clients, alpha, seed)
    shuffle = split == 'train'

    for c in range(num_clients):
        loader = Loader(splits[c], batch_size, shuffle=shuffle)
        dataloaders.append(loader)
    return dataloaders