import os
from .base import DataPreProcess, Datum

from .utils import check_dirs_exist



class FGVCAircraft(DataPreProcess):

    dataset_dir = "fgvc_aircraft"

    def __init__(self, dataset_path, seed):
        self.dataset_dir = dataset_path
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.seed = seed

    def preprocess(self, ):
        if check_dirs_exist(self.dataset_dir, ['train', 'val', 'test']):
            print('Dataset has already been preprocessed, skip...')
            return
        else:
            classnames = []
            with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    classnames.append(line.strip())
            cname2lab = {c: i for i, c in enumerate(classnames)}
            train = self.read_data(cname2lab, "images_variant_train.txt")
            val = self.read_data(cname2lab, "images_variant_val.txt")
            test = self.read_data(cname2lab, "images_variant_test.txt")
            self.save_split(train, val, test, self.image_dir)


    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imname = line[0] + ".jpg"
                classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                classname = classname.replace('/', '')
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items