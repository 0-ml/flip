import os
import json
import tqdm
import shutil
import pandas as pd
from .base import DataPreProcess

from .utils import check_dirs_exist


class iNaturalist(DataPreProcess):

    dataset_dir = "inaturalist"

    def __init__(self, dataset_path, seed):
        self.dataset_dir = dataset_path
        self.img_folder = "train_val_images"
        self.image_dir = os.path.join(self.dataset_dir, self.img_folder)
        self.seed = seed

    def preprocess(self,):
        if check_dirs_exist(self.dataset_dir, ['train', 'val', 'test']):
            print('Dataset has already been preprocessed!')
            return
        else:
            print(f'Dataset: {self.dataset_dir} train/val/test not found!')
            self.split_train_val_test()

    def split_train_val_test(self,):

        train_json_path = 'train2017.json'
        val_json_path = 'val2017.json'
        train_output_dir = 'train_raw'
        val_output_dir = 'test_raw'

        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(val_output_dir, exist_ok=True)
        self.move_images(val_json_path, val_output_dir)
        self.move_images(train_json_path, train_output_dir)

        # split iNaturalist_User_120K
        print('Spliting iNaturalist_User_120K datasets')
        csv_files = {
            'train': 'inaturalist-user-120k/federated_train_user_120k.csv',
            'test': 'inaturalist-user-120k/test.csv'
        }
        for split in ['train', 'test']:
            original_folder = f'{split}_raw'
            csv_file = csv_files[split]
            self.split_user_120k(original_folder, split, csv_file)

        os.system("ln -s test val")

    def move_images(self, json_path, output_dir):
        with open(json_path, 'r') as f:
            data = json.load(f)
            images = data['images']
            for image_info in tqdm(images, desc=f"Moving images to {output_dir}"):
                image_path = image_info['file_name']
                file_name = image_info['file_name']
                class_name = file_name.split('/')[-2]
                dest_folder = os.path.join(output_dir, class_name)
                os.makedirs(dest_folder, exist_ok=True)
                dest_path = os.path.join(dest_folder, os.path.basename(image_path))
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(image_path, dest_path)

    def split_user_120k(self, original_folder, output_folder, csv_file):
        df = pd.read_csv(csv_file)
        image_ids = df['image_id'].tolist()
        image_id_set = set(image_ids)  # Using a set for faster lookup
        image_paths = {}
        total_files = sum([len(files) for _, _, files in os.walk(original_folder)])

        for root, dirs, files in tqdm(os.walk(original_folder),
                                      total=total_files, desc="Processing Images"):
            for file in files:
                image_id = file.split('.')[0]
                if image_id in image_id_set:
                    image_path = os.path.join(root, file)
                    image_paths[image_id] = image_path

        for image_id, source_path in tqdm(image_paths.items()):
            class_name = os.path.basename(os.path.dirname(source_path))
            class_folder = os.path.join(output_folder, class_name)
            os.makedirs(class_folder, exist_ok=True)

            destination_path = os.path.join(class_folder, f'{image_id}.jpg')
            shutil.copy2(source_path, destination_path)
