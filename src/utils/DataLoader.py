import glob
import logging
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class DataSet(Dataset):
    def __init__(self, files, integer_labels, transform=None):
        self.files = files
        self.integer_labels = integer_labels
        self.transform = transform

    def __getitem__(self, item):
        image_dict, label_name = self.files[item]
        image = self.load_file(image_dict["normalizedImg"])

        if self.transform:
            image = self.transform(image)

        label = torch.Tensor([self.integer_labels[label_name]])

        return image, label, label_name

    def load_file(self, file):
        this_image = Image.open(file)
        return this_image

    def __len__(self):
        return len(self.files)


class DataLoader(pl.LightningDataModule):
    def __init__(self, transform=None, **kwargs):
        super().__init__()

        self.transform = transform

        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.input_channels = None
        self.output_channels = None
        self.unique_labels = []
        self.integer_class_labels = dict()

        self.batch_size = kwargs["batch_size"]
        self.num_workers = 20
        self.train_split = 0.9
        self.validation_split = 0.1
        self.shuffle_train_dataset = True
        self.shuffle_validation_dataset = False
        self.shuffle_test_dataset = False
        self.data_path = kwargs["data_path"]

    def setup(self, stage=None):
        training_pairs = self.prepare_data_setup(train=True)
        testing_images = self.prepare_data_setup(train=False)
        self.integer_class_labels = self.set_up_integer_class_labels()

        if len(training_pairs) == 0:
            raise FileNotFoundError(f"Did not find any files")

        train_split = self.train_split
        valid_split = train_split + self.validation_split
        length = len(training_pairs)

        train_split_start = 0
        train_split_end = int(length * train_split)
        valid_split_start = train_split_end
        valid_split_end = length

        train_subset = training_pairs[train_split_start: train_split_end]
        valid_subset = training_pairs[valid_split_start: valid_split_end]
        test_subset = testing_images

        if stage == 'fit' or stage is None:
            self.train_data = DataSet(train_subset, transform=self.transform,
                                              integer_labels=self.integer_class_labels)
            logging.debug(f"Number of training samples: {len(self.train_data)}")
            self.valid_data = DataSet(valid_subset, transform=self.transform,
                                              integer_labels=self.integer_class_labels)
            logging.debug(f"Number of validation samples: {len(self.valid_data)}")

        if stage == 'test' or stage is None:
            self.test_data = DataSet(test_subset, transform=self.transform,
                                             integer_labels=self.integer_class_labels)

    def prepare_data_setup(self, train=True):
        pairs = []
        if train:
            _data_path = os.path.join(self.data_path, "trainSet")
            truth_df = pd.read_csv(os.path.join(_data_path, "trainSet.txt"))
        else:
            _data_path = os.path.join(self.data_path, "testSet")
            truth_df = pd.read_csv(os.path.join(_data_path, "testSet.txt"))

        truth_df = truth_df.set_index("ImageFile")

        for i in range(len(glob.glob(os.path.join(_data_path, "normalizedImg/*.png")))):
            image_dict = {}
            raw_image_name = f"normalizedImg/P_1_{i+1}.png"
            image_dict["image"] = os.path.join(_data_path, raw_image_name)
            for folder in glob.glob(os.path.join(_data_path, "Radiomics/*")):
                raw_image_name = f"P_1_{i+1}.png"
                image_dict[folder] = os.path.join(_data_path, "Radiomics", folder, raw_image_name)

            target = truth_df[truth_df.index == f"P_{i+1}.png"]["Prognosis"]
            pairs.append((image_dict, target))
        return pairs

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_train_dataset, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_validation_dataset, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_test_dataset, pin_memory=True)

    def set_up_integer_class_labels(self):
        integer_class_labels = dict()
        for i, label in enumerate(self.unique_labels):
            integer_class_labels[label] = i
        return integer_class_labels
