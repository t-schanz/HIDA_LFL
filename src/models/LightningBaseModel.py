import logging
import os

import pytorch_lightning as pl
import pytorch_lightning.metrics as pl_metrics
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet101


class LightningModel(pl.LightningModule):

    def __init__(self, class_labels, example_input_array, *args, **kwargs):

        super().__init__()
        self.class_labels = class_labels
        self.model = self.define_model(input_channels=35)
        self.learning_rate = kwargs["learning_rate"]
        self.loss_func = nn.BCEWithLogitsLoss()
        self.accuracy_func = pl_metrics.Accuracy()
        self.save_hyperparameters()
        self.example_input_array = example_input_array

    def define_model(self, input_channels=1):
        feature_extractor = resnet18(pretrained=False, num_classes=1000)
        feature_extractor.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        classifier = nn.Linear(1000, 1)

        model = nn.Sequential(feature_extractor, classifier)

        return model

    def forward(self, images,  *args, **kwargs):
        predictions = self.model(images)
        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        # for the toy example we will not use the meta_data and only the images to make a prediction.
        predictions = self(images)
        logging.debug(f"labels have the shape: {labels.shape}")
        logging.debug(f"predictions have the shape: {predictions.shape}")

        loss = self.loss_func(predictions, labels.unsqueeze(-1).float())
        accuracy = self.accuracy_func(torch.sigmoid(predictions[:, 0]).detach().cpu(), labels.to(torch.int).detach().cpu())

        # lets log some values for inspection (for example in tensorboard):
        self.log("NLL Training", loss)
        self.log("Accuracy Training", accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        # for the toy example we will not use the meta_data and only the images to make a prediction.
        predictions = self(images)

        loss = self.loss_func(predictions, labels.unsqueeze(-1).float())
        accuracy = self.accuracy_func(torch.sigmoid(predictions[:, 0]).detach().cpu(), labels.to(torch.int).detach().cpu())

        # lets log some values for inspection (for example in tensorboard):
        self.log("NLL Validation", loss)
        self.log("Accuracy Validation", accuracy)

        return loss

    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        # for the toy example we will not use the meta_data and only the images to make a prediction.
        predictions = self(images)

        loss = self.loss_func(predictions, labels.unsqueeze(-1).float())
        accuracy = self.accuracy_func(torch.sigmoid(predictions[:, 0]).detach().cpu(), labels.to(torch.int).detach().cpu())

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