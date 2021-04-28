import glob
import logging
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class DataSet(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __getitem__(self, item):
        image_dict, label_name = self.files[item]

        if label_name == "MILD":
            label = 0
        elif label_name == "SEVERE":
            label = 1
        else:
            raise KeyError

        image_subsets = ["image", "ClusterShade", "DA", "DE", "Energy", "Entropy", "ID", "Kurtosis", "Mean", "Variance"]
        images = []
        for element in image_subsets:
            images.append(self.load_file(image_dict[element]))

        if self.transform:
            for i in range(len(images)):
                images[i] = self.transform(images[i])

        train_tensor = torch.stack(images)[:, 0, :, :]

        return train_tensor, label, label_name

    def load_file(self, file):
        this_image = Image.open(file)
        return this_image

    def __len__(self):
        return len(self.files)


class HidaDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size, data_path, num_workers, transform=None, **kwargs):
        super().__init__()

        self.transform = transform

        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.input_channels = None
        self.output_channels = None
        self.unique_labels = []
        self.integer_class_labels = dict()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = 0.9
        self.validation_split = 0.1
        self.shuffle_train_dataset = True
        self.shuffle_validation_dataset = False
        self.shuffle_test_dataset = False
        self.data_path = data_path

    def setup(self, stage=None):
        training_pairs = self.prepare_data_setup(train=True)
        testing_images = self.prepare_data_setup(train=False)
        self.integer_class_labels = self.set_up_integer_class_labels()

        if len(training_pairs) == 0:
            raise FileNotFoundError(f"Did not find any files")

        train_split = self.train_split
        length = len(training_pairs)

        train_split_start = 0
        train_split_end = int(length * train_split)
        valid_split_start = train_split_end

        train_subset = training_pairs[train_split_start: train_split_end]
        valid_subset = training_pairs[valid_split_start: length]
        test_subset = testing_images

        if stage == 'fit' or stage is None:
            self.train_data = DataSet(train_subset, transform=self.transform)
            logging.debug(f"Number of training samples: {len(self.train_data)}")
            self.valid_data = DataSet(valid_subset, transform=self.transform)
            logging.debug(f"Number of validation samples: {len(self.valid_data)}")

        if stage == 'test' or stage is None:
            self.test_data = DataSet(test_subset, transform=self.transform)

    def prepare_data_setup(self, train=True):
        pairs = []
        if train:
            _data_path = os.path.join(self.data_path, "trainSet")
            truth_df = pd.read_csv(os.path.join(_data_path, "trainSet.txt"))
        else:
            _data_path = os.path.join(self.data_path, "testSet")
            truth_df = pd.read_csv(os.path.join(_data_path, "testSet.txt"))

        truth_df = truth_df.set_index("ImageFile")

        image_names = truth_df.index.to_list()

        for image_name in image_names:
            if str(image_name) == "nan":
                continue
            image_dict = {}
            raw_image_name = f"normalizedImg/{image_name}"
            image_dict["image"] = os.path.join(_data_path, raw_image_name)
            for folder in glob.glob(os.path.join(_data_path, "Radiomics/*")):
                indicator = os.path.split(folder)[-1]
                image_dict[indicator] = os.path.join(folder, image_name)

            target = truth_df[truth_df.index == image_name]["Prognosis"].iloc[0]
            pairs.append((image_dict, target))
        return pairs

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_train_dataset, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_validation_dataset, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_test_dataset, pin_memory=False)

    def set_up_integer_class_labels(self):
        integer_class_labels = dict()
        for i, label in enumerate(self.unique_labels):
            integer_class_labels[label] = i
        return integer_class_labels
