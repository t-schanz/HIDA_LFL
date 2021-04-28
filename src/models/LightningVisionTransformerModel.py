import logging

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet18
import pytorch_lightning.metrics as pl_metrics

from transformers import ViTModel, ViTConfig
from transformers import ViTFeatureExtractor, ViTForImageClassification

class VisionTransformerModel(pl.LightningModule):

    def __init__(self, class_labels, *args, **kwargs):

        super().__init__()
        self.class_labels = class_labels
        self.model = self.define_model(input_channels=3)
        self.learning_rate = kwargs["learning_rate"]
        self.loss_func = nn.BCEWithLogitsLoss()
        self.accuracy_func = pl_metrics.Accuracy()
        self.save_hyperparameters()

    def define_model(self, input_channels=1):

        configuration = ViTConfig(num_channels=input_channels, image_size=900)
        
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        feature_extractor.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        classifier = nn.Linear(1000, 1)
        
        model = ViTForImageClassification(configuration) #.from_pretrained('google/vit-base-patch16-224')
        #model = nn.Sequential(feature_extractor, classifier)
        
        return model

    def forward(self, images,  *args, **kwargs):
        predictions = self.model(images).logits
        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        # for the toy example we will not use the meta_data and only the images to make a prediction.
        predictions = self.model(images).logits
        logging.debug(f"labels have the shape: {labels.shape}")
        logging.debug(f"predictions have the shape: {predictions.shape}")

        predictions = predictions.argmax(-1)
        
        loss = self.loss_func(predictions.float(), labels.float())
        accuracy = self.accuracy_func(predictions, labels)

        # lets log some values for inspection (for example in tensorboard):
        self.log("NLL Training", loss)
        self.log("Accuracy Training", accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        # for the toy example we will not use the meta_data and only the images to make a prediction.
        predictions = self.model(images).logits
        
        predictions = predictions.argmax(-1)
                
        loss = self.loss_func(predictions.float(), labels.float())
        accuracy = self.accuracy_func(F.sigmoid(predictions, dim=1), labels)

        # lets log some values for inspection (for example in tensorboard):
        self.log("NLL Validation", loss)
        self.log("Accuracy Validation", accuracy)

        return loss

    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        # for the toy example we will not use the meta_data and only the images to make a prediction.
        predictions = self(images)

        loss = self.loss_func(predictions.float(), labels.float())
        accuracy = self.accuracy_func(F.softmax(predictions, dim=1).detach().cpu(), labels.to(torch.int).detach().cpu())

        # lets log some values for inspection (for example in tensorboard):
        self.log("NLL Test", loss)
        self.log("Accuracy Test", accuracy)

        return loss

    def log_confusion_matrix(self, predictions, targets):
        conf_mat = confusion_matrix(torch.argmax(predictions, dim=-1).detach().cpu(), targets.detach().cpu(),
                                    labels=np.arange(len(self.class_labels)))

        fig, ax = plt.subplots()
        ax.imshow(conf_mat)
        ax.set_xticklabels(self.class_labels)
        ax.set_yticklabels(self.class_labels)
        ax.set_xlabel("Target")
        ax.set_ylabel("Prediction")
        self.logger.experiment[0].add_figure("Confusion_Matrix", fig, self.global_step)
        plt.close("all")

    def log_images(self, images, labels):
        if self.hparams.batch_size >= 16:
            fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
            for i in range(16):
                ax = axes.flatten()[i]
                ax.imshow(images[i].detach().cpu().moveaxis(0, -1))
                ax.set_title(labels[i])

            self.logger.experiment[0].add_figure("Image Matrix", fig, self.global_step)
            plt.close("all")
