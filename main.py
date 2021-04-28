import logging
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms

from src.models.LightningBaseModel import LightningModel
from src.utils.DataLoader import HidaDataLoader


def load_config():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help="The random seed for the experiment.")
    parser.add_argument('--tensorboard_logger_logdir', type=str, default="logs/tb_logs",
                        help="The path where to log the data.")
    parser.add_argument('--experiment_name', type=str, default="HIDA", help="The name of the experiment.")
    parser.add_argument('--checkpoint_file_path', type=str, default="logs/checkpoints",
                        help="The path where to store the checkpoints.")
    parser.add_argument('--data_path', type=str, default="data/", help="The path where the data is stored at.")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.002)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    return args


def main():

    args = load_config()
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    if args.fast_dev_run:
        logging.basicConfig(level=logging.DEBUG, format='%(name)s %(funcName)s %(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(name)s %(funcName)s %(levelname)s %(message)s')

    if args.fast_dev_run:
        torch.autograd.set_detect_anomaly(True)

    transform = transforms.Compose([
        transforms.RandomCrop(size=[900, 900]),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor()
    ])

    data_module = HidaDataLoader.from_argparse_args(args, transform=transform, **args.__dict__)
    data_module.setup()

    # if the model is trained on GPU add a GPU logger to see GPU utilization in comet-ml logs:
    if args.gpus:
        callbacks = [pl.callbacks.GPUStatsMonitor()]
    else:
        callbacks = None

    # logging to tensorboard:
    test_tube_logger = pl_loggers.TestTubeLogger(save_dir=args.tensorboard_logger_logdir,
                                                 name=args.experiment_name,
                                                 create_git_tag=False,
                                                 log_graph=True)

    # initializes a callback to save the 5 best model weights measured by the lowest loss:
    checkpoint_callback = ModelCheckpoint(monitor="NLL Validation",
                                          save_top_k=5,
                                          mode='min',
                                          save_last=True,
                                          dirpath=os.path.join(args.checkpoint_file_path, args.experiment_name),
                                          )

    model = LightningModel(class_labels=data_module.unique_labels, **args.__dict__)

    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=callbacks,
                                            logger=[test_tube_logger],
                                            checkpoint_callback=checkpoint_callback,
                                            log_every_n_steps=10,
                                            )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
