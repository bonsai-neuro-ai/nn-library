import unittest
from os import getenv

from dotenv import load_dotenv

from nn_lib.datasets import (
    MNISTDataModule,
    CIFAR10DataModule,
    CIFAR100DataModule,
    ImageNetDataModule,
    CocoDetectionDataModule,
)


class TestDatasetsExist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv()
        cls.data_root = getenv("DATA_ROOT", "data")

    def test_mnist_train(self):
        data = MNISTDataModule(root_dir=self.data_root)
        data.prepare_data()
        data.setup("fit")
        dl = data.train_dataloader()
        batch = next(iter(dl))
        self.assertEqual(batch[0].shape[1:], data._default_shape)

    def test_cifar10_train(self):
        data = CIFAR10DataModule(root_dir=self.data_root)
        data.prepare_data()
        data.setup("fit")
        dl = data.train_dataloader()
        batch = next(iter(dl))
        self.assertEqual(batch[0].shape[1:], data._default_shape)

    def test_cifar100_train(self):
        data = CIFAR100DataModule(root_dir=self.data_root)
        data.prepare_data()
        data.setup("fit")
        dl = data.train_dataloader()
        batch = next(iter(dl))
        self.assertEqual(batch[0].shape[1:], data._default_shape)

    def test_imagenet_train(self):
        data = ImageNetDataModule(root_dir=self.data_root)
        data.prepare_data()
        data.setup("fit")
        dl = data.train_dataloader()
        batch = next(iter(dl))
        self.assertEqual(batch[0].shape[1:], data._default_shape)

    def test_coco_train(self):
        data = CocoDetectionDataModule(root_dir=self.data_root)
        data.prepare_data()
        data.setup("fit")
        dl = data.train_dataloader()
        batch = next(iter(dl))
        self.assertEqual(batch[0].shape[1:], data._default_shape)
