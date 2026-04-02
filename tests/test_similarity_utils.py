import unittest
from itertools import batched

import torch
from torch.testing import assert_close
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import resnet18

from nn_lib.analysis.similarity.utils import (
    prep_conv_layers,
    assert_repeatable_iter_factory,
    create_gram_matrix_from_batches,
    iter_batches_of_reps,
    check_shapes,
)


class TestShapeHandling(unittest.TestCase):
    def test_check_shapes_flat_flat(self):
        x = torch.randn(100, 5)
        y = torch.randn(100, 6)
        check_shapes(x, y)

    def test_check_shapes_conv_conv(self):
        x = torch.randn(100, 5, 32, 32)
        y = torch.randn(100, 6, 32, 32)
        check_shapes(x, y)

    def test_check_shapes_conv_conv_size_mismatch(self):
        x = torch.randn(100, 5, 32, 32)
        y = torch.randn(100, 6, 16, 16)
        check_shapes(x, y)

    def test_check_shapes_conv_conv_channels_last(self):
        x = torch.randn(100, 5, 32, 32)
        y = torch.randn(100, 6, 32, 32)
        x = x.to(memory_format=torch.channels_last)
        check_shapes(x, y)
        y = y.to(memory_format=torch.channels_last)
        check_shapes(x, y)

    def test_check_shapes_m_mismatch(self):
        x = torch.randn(100, 5)
        y = torch.randn(99, 6)
        with self.assertRaises(ValueError):
            check_shapes(x, y)

    def test_check_shapes_flat_vs_conv(self):
        x = torch.randn(100, 5)
        y = torch.randn(100, 6, 16, 16)
        with self.assertRaises(ValueError):
            check_shapes(x, y)

    def test_flatten_conv_single(self):
        x = torch.randn(10, 3, 32, 32)
        (flattened,) = prep_conv_layers(x, conv_method="flatten")
        assert flattened.shape == (10, 3 * 32 * 32)

    def test_flatten_conv_multiple(self):
        x = torch.randn(10, 3, 32, 32)
        y = torch.randn(10, 3, 32, 32)
        z = torch.randn(10, 3, 32, 32)
        flat_x, flat_y, flat_z = prep_conv_layers(x, y, z, conv_method="flatten")
        assert flat_x.shape == (10, 3 * 32 * 32)
        assert flat_y.shape == (10, 3 * 32 * 32)
        assert flat_z.shape == (10, 3 * 32 * 32)

    def test_windowed_conv_single(self):
        x = torch.randn(10, 3, 32, 32)
        for w in [1, 3, 5]:
            (flattened,) = prep_conv_layers(x, conv_method="window", window_size=w)
            assert flattened.shape == (10 * (32 - w + 1) * (32 - w + 1), 3 * w * w)

    def test_windowed_conv_multiple(self):
        x = torch.randn(10, 3, 32, 32)
        y = torch.randn(10, 3, 32, 32)
        z = torch.randn(10, 3, 32, 32)
        for w in [1, 3, 5]:
            flat_x, flat_y, flat_z = prep_conv_layers(x, y, z, conv_method="window", window_size=w)
            assert flat_x.shape == (10 * (32 - w + 1) * (32 - w + 1), 3 * w * w)
            assert flat_y.shape == (10 * (32 - w + 1) * (32 - w + 1), 3 * w * w)
            assert flat_z.shape == (10 * (32 - w + 1) * (32 - w + 1), 3 * w * w)

    def test_windowed_conv_size_mismatch_error(self):
        x = torch.randn(10, 3, 16, 16)
        y = torch.randn(10, 3, 32, 32)
        z = torch.randn(10, 3, 24, 24)
        for w in [1, 3, 5]:
            with self.assertRaises(ValueError):
                prep_conv_layers(x, y, z, conv_method="window", window_size=w)

    def test_windowed_conv_size_mismatch_upsample(self):
        x = torch.randn(10, 3, 16, 16)
        y = torch.randn(10, 3, 32, 32)
        z = torch.randn(10, 3, 24, 24)
        for w in [1, 3, 5]:
            flat_x, flat_y, flat_z = prep_conv_layers(
                x, y, z, conv_method="window", window_size=w, size_mismatch="upsample"
            )
            assert flat_x.shape == (10 * (32 - w + 1) * (32 - w + 1), 3 * w * w)
            assert flat_y.shape == (10 * (32 - w + 1) * (32 - w + 1), 3 * w * w)
            assert flat_z.shape == (10 * (32 - w + 1) * (32 - w + 1), 3 * w * w)

    def test_windowed_conv_size_mismatch_downsample(self):
        x = torch.randn(10, 3, 16, 16)
        y = torch.randn(10, 3, 32, 32)
        z = torch.randn(10, 3, 24, 24)
        for w in [1, 3, 5]:
            flat_x, flat_y, flat_z = prep_conv_layers(
                x, y, z, conv_method="window", window_size=w, size_mismatch="downsample"
            )
            assert flat_x.shape == (10 * (16 - w + 1) * (16 - w + 1), 3 * w * w)
            assert flat_y.shape == (10 * (16 - w + 1) * (16 - w + 1), 3 * w * w)
            assert flat_z.shape == (10 * (16 - w + 1) * (16 - w + 1), 3 * w * w)


class TestIteratorUtils(unittest.TestCase):
    def test_dataloader_repeatable(self):
        x = torch.randn(100, 5)
        y = torch.randn(100, 5)
        ds = TensorDataset(x, y)

        dl_static = DataLoader(ds, batch_size=10, shuffle=False)
        assert_repeatable_iter_factory(lambda: dl_static)

    def test_shuffled_dataloader_not_repeatable(self):
        x = torch.randn(100, 5)
        y = torch.randn(100, 5)
        ds = TensorDataset(x, y)

        dl_shuffle = DataLoader(ds, batch_size=10, shuffle=True)
        with self.assertRaises(ValueError):
            assert_repeatable_iter_factory(lambda: dl_shuffle)

    def test_create_gram_flat(self):
        x = torch.randn(100, 5)
        ds = TensorDataset(x)
        dl = DataLoader(ds, batch_size=10, shuffle=False)

        gram = create_gram_matrix_from_batches(lambda: dl)[0]
        true_gram = x @ x.T

        assert_close(gram, true_gram)

    def test_create_gram_conv(self):
        x = torch.randn(100, 3, 4, 4)
        ds = TensorDataset(x)
        dl = DataLoader(ds, batch_size=10, shuffle=False)

        gram = create_gram_matrix_from_batches(lambda: dl)[0]
        true_gram = torch.einsum("ichw,jchw->ij", x, x)

        assert_close(gram, true_gram)

    def test_create_gram_flat_multiple(self):
        x = torch.randn(100, 5)
        y = torch.randn(100, 6)
        ds = TensorDataset(x, y)
        dl = DataLoader(ds, batch_size=10, shuffle=False)

        gram_x, gram_y = create_gram_matrix_from_batches(lambda: dl)
        true_gram_x = x @ x.T
        true_gram_y = y @ y.T

        assert_close(gram_x, true_gram_x)
        assert_close(gram_y, true_gram_y)

    def test_create_gram_conv_multiple(self):
        x = torch.randn(100, 3, 4, 4)
        y = torch.randn(100, 6)
        ds = TensorDataset(x, y)
        dl = DataLoader(ds, batch_size=10, shuffle=False)

        gram_x, gram_y = create_gram_matrix_from_batches(lambda: dl)
        true_gram_x = torch.einsum("ichw,jchw->ij", x, x)
        true_gram_y = torch.einsum("in,jn->ij", y, y)

        assert_close(gram_x, true_gram_x)
        assert_close(gram_y, true_gram_y)

    def test_gram_resnet(self):
        model = resnet18(pretrained=False).eval()
        x = torch.randn(100, 3, 224, 224)
        ds = TensorDataset(x)
        dl = DataLoader(ds, batch_size=10, shuffle=False)
        gram = create_gram_matrix_from_batches(lambda: iter_batches_of_reps(dl, model))[0]

        with torch.no_grad():
            y = model(x)
        true_gram = y @ y.T

        assert_close(gram, true_gram)
