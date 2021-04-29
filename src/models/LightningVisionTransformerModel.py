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

import os

class VisionTransformerModel(pl.LightningModule):

    def __init__(self, class_labels, example_input_array, *args, **kwargs):

        super().__init__()
        self.class_labels = class_labels
        self.feature_extractor = self.define_feature_extractor()
        self.model = self.define_model(input_channels=3)
        self.learning_rate = kwargs["learning_rate"]
        self.loss_func = nn.CrossEntropyLoss()
        self.accuracy_func = pl_metrics.Accuracy()
        self.save_hyperparameters()
        self.example_input_array = example_input_array

    def define_feature_extractor(self):
        feature_extractor = ViTFeatureExtractor(do_resize=False, do_normalize=False)
        #feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        return feature_extractor
    
    def define_model(self, input_channels=1):
        configuration = ViTConfig(num_channels=input_channels, image_size=900)
        model = ViTForImageClassification(configuration)
        #model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        return model

    def forward(self, images, *args, **kwargs):
        
        features = self.feature_extractor(images, return_tensors="np")
        predictions = self.model(features["pixel_values"][0]).logits
        
        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        predictions = self(images)

        logging.debug(f"labels have the shape: {labels.shape}")
        logging.debug(f"predictions have the shape: {predictions.shape}")
    
        loss = self.loss_func(predictions, labels)
        accuracy = self.accuracy_func(F.softmax(predictions, dim=1).argmax(-1), labels)

        # lets log some values for inspection (for example in tensorboard):
        self.log("NLL Training", loss)
        self.log("Accuracy Training", accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch
        
        predictions = self(images)
                                
        loss = self.loss_func(predictions, labels)
        accuracy = self.accuracy_func(F.softmax(predictions, dim=1).argmax(-1), labels)

        # lets log some values for inspection (for example in tensorboard):
        self.log("NLL Validation", loss)
        self.log("Accuracy Validation", accuracy)

        return loss

    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        # for the toy example we will not use the meta_data and only the images to make a prediction.
        predictions = self(images)

        loss = self.loss_func(predictions.float(), labels.float())
        accuracy = self.accuracy_func(F.softmax(predictions, dim=1), labels)

        # lets log some values for inspection (for example in tensorboard):
        self.log("NLL Test", loss)
        self.log("Accuracy Test", accuracy)

        return loss
    
    def on_save_checkpoint(self, checkpoint) -> None:
        # save model to onnx:
        folder = self.trainer.checkpoint_callback.dirpath
        onnx_file_generator = os.path.join(folder, f"model_{self.global_step}.onnx")
        torch.onnx.export(model=self.model,
                      args=self.example_input_array.to(self.device),
                      f=onnx_file_generator,
                      opset_version=12,
                      verbose=False,
                      export_params=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},  # makes the batch-size variable for inference
                                    'output': {0: 'batch_size'}}
                      )

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
