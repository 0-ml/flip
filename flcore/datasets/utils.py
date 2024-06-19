import numpy as np
import torch
import torchvision as tv
import os
import copy
import shutil
import math
import os.path as osp
import json

from ..pretty.logger import log



ImageFolderList = ['Caltech101', ]




def update_class_names(dataset, replace_dict):
    # repalce some class names with inconsistent names in CLIP model
    if replace_dict is not None:
        class_to_idx = {}
        for k, v in dataset.class_to_idx.items():
            if k in replace_dict.keys():
                class_to_idx[replace_dict[k]] = v
            else:
                class_to_idx[k] = v
        dataset.class_to_idx = class_to_idx
    # update inconsistent names of image folders and real class names
    real_names = dataset.real_class_names
    if real_names is not None:
        class_to_idx = {}
        classes = []
        for k, v in dataset.class_to_idx.items():
            rn = real_names[k]
            class_to_idx[rn] = v
        for n in dataset.classes:
            rn = real_names[n]
            classes.append(rn)
        dataset.class_to_idx = class_to_idx
        dataset.classes = classes

    return dataset




def get_stats(datasets, num_clients, num_classes):
    stats = {c:[] for c in range(num_clients)}
    for c, ds in enumerate(datasets):
        stats[c] = np.bincount(ds.targets, minlength=num_classes).tolist()
    return stats

def bin_index(dataset, name):
    data = dataset
    bins = {}
    for i, label in enumerate(data.targets):
        bins.setdefault(int(label), []).append(i)
    flattened = []
    for k in sorted(bins):
        flattened += bins[k]
    return bins, flattened


def split_dataset(name, dataset, policy, num_clients, num_shards, alpha, beta, batch_size, drop_last, seed):
    # guarantee determinism
    log.info('spliting local datasets...')
    np.random.seed(seed)
    torch.manual_seed(seed)
    bins, flattened = bin_index(dataset, name)
    num_classes = len(dataset.classes)

    if policy == 'iid':
        client_indices = [[] for _ in range(num_clients)]
        for k, idx_k in bins.items():
            np.random.shuffle(idx_k)
            for c, (idx_j, idx) in enumerate(
                        zip(client_indices, np.split(np.array(idx_k), num_clients))):
                idx_j += idx.tolist()
        datasets = [ImageSubset(dataset, client_indices[c]) for c in range(num_clients)]
        statistics = get_stats(datasets, num_clients, num_classes)
        all_data = []
        for c, v in statistics.items():
            log.debug(f'client: {c}, total: {int(np.sum(v))}, data dist: {v}')
            all_data.extend(v)
        log.debug(f'total local training data: {np.sum(all_data)}')
        return datasets, statistics

    if policy == 'size':
        splits = np.random.random(num_clients)
        splits *= len(dataset) / np.sum(splits)
        splits = splits.astype(np.int)
        remains = sum(splits)
        remains = np.random.randint(0, num_clients, len(dataset) - remains)
        for n in range(num_clients):
            splits[n] += sum(remains == n)
        return torch.utils.data.dataset.random_split(dataset, splits.tolist())

    if policy == 'dirichlet':
        data_num, _counter = 0, 0
        num_data = len(flattened)
        statistics = {c:[] for c in range(num_clients)}
        min_size = batch_size if drop_last else 5
        while data_num < min_size:
            client_indices = [[] for _ in range(num_clients)]
            for k, idx_k in bins.items():
                np.random.shuffle(idx_k)
                prop = np.random.dirichlet(np.repeat(alpha, num_clients))
                prop = np.array([p * (len(idx_c) < num_data / num_clients)
                                    for p, idx_c in zip(prop, client_indices)])
                prop = prop / prop.sum()
                prop = (np.cumsum(prop)*len(idx_k)).astype(int)[:-1]
                for c, (idx_j, idx) in enumerate(zip(client_indices, np.split(idx_k, prop))):
                    idx_j += idx.tolist()
                data_num = min([len(idx_c) for idx_c in client_indices])
            _counter += 1
            if _counter == 1000:
                raise "data partition is not feasible..."
        idx_subset = ImageSubset
        datasets = [idx_subset(dataset, client_indices[c]) for c in range(num_clients)]
        statistics = get_stats(datasets, num_clients, num_classes)
        all_data = []
        for c, v in statistics.items():
            log.debug(f'client: {c}, total: {int(np.sum(v))}, data dist: {v}')
            all_data.extend(v)
        log.debug(f'total local training data: {np.sum(all_data)}')
        return datasets, statistics

    if policy == 'task':
        bins, flattened = bin_index(dataset, name)
        datasets = []
        client_indices = [[] for _ in range(num_clients)]
        statistics = {c:[] for c in range(num_clients)}
        if num_shards % num_clients:
            raise ValueError(
                'Expect the number of shards to be '
                'evenly distributed to clients.')
        num_client_shards = num_shards // num_clients
        shard_size = len(dataset) // num_shards
        shards = list(range(num_shards))
        np.random.shuffle(shards)  # fix np.ramdom error
        for i in range(num_clients):
            shard_offset = i * num_client_shards
            indices = []
            for s in shards[shard_offset:shard_offset + num_client_shards]:
                if s == len(shards) - 1:
                    indices += flattened[s * shard_size:]
                else:
                    indices += flattened[s * shard_size:(s + 1) * shard_size]
            subset = ImageSubset(dataset, indices)
            datasets.append(subset)
        return datasets, statistics

    raise TypeError(f'Unrecognized split policy {policy!r}.')

class ImageSubset(tv.datasets.ImageFolder):
    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        self.indices = indices
        # self.samples = self.dataset.samples[indices]
        if self.indices is not None:
            self.samples = [self.dataset.samples[i] for i in indices]
        else:
            self.samples = self.dataset.samples
        self.targets = [t for _, t in self.samples]
        self.loader = dataset.loader
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        target = torch.tensor(int(target))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __reinit_samples__(self,):
        pass


def subsample_classes(dataset, subsample="all"):
    """Divide classes into two groups. The first group
    represents base classes while the second group represents
    new classes.
    Args:
        datasets: a list of datasets, e.g. train, val and test.
        subsample (str): what classes to subsample.
    """
    assert subsample in ["all", "base", "new"]

    if subsample == "all":
        return dataset

    class_names = copy.deepcopy(dataset.classes)
    class_names.sort()
    num_classes = len(class_names)
    m = math.ceil( len(class_names) / 2)

    print(f"SUBSAMPLE {subsample.upper()} CLASSES!")

    if subsample == "base":
        sel_classes = class_names[:m]  # take the first half
    else:
        sel_classes = class_names[m:]  # take the second half
    sel_class_targets = [dataset.class_to_idx[c] for c in sel_classes]

    sub_dataset = []
    data_indices = {} # the data indices for entire trainset
    classes = list(range(num_classes))
    for j in classes:
        data_indices[j] = [i for i, label in
                enumerate(dataset.targets) if label == j]
    idx_sel, idx_del = [], []
    for i, (img_path, target) in enumerate(dataset.samples):
        if target in sel_class_targets:
            idx_sel.append(i)
        else:
            idx_del.append(i)
    sub_dataset = copy.deepcopy(dataset)
    sub_dataset.imgs = np.delete(dataset.imgs, idx_del, axis=0)
    sub_dataset.samples = np.delete(dataset.samples, idx_del, axis=0)
    sub_dataset.targets = np.delete(dataset.targets, idx_del, axis=0)
    sub_dataset.data_indices = np.array(idx_sel)
    sub_dataset.classes = sel_classes
    if subsample == 'new':
        sub_dataset.targets = sub_dataset.targets - m
    return sub_dataset


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        os.makedirs(directory)


def check_dirs_exist(data_dir, dirs):
    for d in dirs:
        if not osp.exists(os.path.join(data_dir, d)):
            return False
    return True

def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items

def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj