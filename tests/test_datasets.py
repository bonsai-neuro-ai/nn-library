import unittest
from nn_lib.datasets import (
    MNISTDataModule,
    CIFAR10DataModule,
    CIFAR100DataModule,
    ImageNetDataModule,
    CocoDetectionDataModule,
)
from nn_lib.env import add_parser as add_env_parser, EnvConfig
import jsonargparse


class TestDatasetLoadsInEnv(unittest.TestCase):
    env: EnvConfig = None

    @classmethod
    def setUpClass(cls):
        parser = jsonargparse.ArgumentParser(default_config_files=["configs/local/env.yaml"])
        add_env_parser(parser)
        args = parser.parse_args()
        cls.env = args.env

    def test_mnist_train(self):
        data = MNISTDataModule(root_dir=self.env.data_root)
        data.prepare_data()
        data.setup("fit")
        dl = data.train_dataloader()
        batch = next(iter(dl))
        self.assertEqual(batch[0].shape[1:], data._default_shape)

    def test_cifar10_train(self):
        data = CIFAR10DataModule(root_dir=self.env.data_root)
        data.prepare_data()
        data.setup("fit")
        dl = data.train_dataloader()
        batch = next(iter(dl))
        self.assertEqual(batch[0].shape[1:], data._default_shape)

    def test_cifar100_train(self):
        data = CIFAR100DataModule(root_dir=self.env.data_root)
        data.prepare_data()
        data.setup("fit")
        dl = data.train_dataloader()
        batch = next(iter(dl))
        self.assertEqual(batch[0].shape[1:], data._default_shape)

    def test_imagenet_train(self):
        data = ImageNetDataModule(root_dir=self.env.data_root)
        data.prepare_data()
        data.setup("fit")
        dl = data.train_dataloader()
        batch = next(iter(dl))
        self.assertEqual(batch[0].shape[1:], data._default_shape)

    def test_coco_train(self):
        data = CocoDetectionDataModule(root_dir=self.env.data_root)
        data.prepare_data()
        data.setup("fit")
        dl = data.train_dataloader()
        batch = next(iter(dl))
        self.assertEqual(batch[0].shape[1:], data._default_shape)
