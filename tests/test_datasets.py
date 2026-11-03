import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest
import torch

from fedrl.data.datasets import NORM_STATS, load_dataset, raw_to_tensors


def test_mnist_normalization_and_layout():
    data = torch.zeros(4, 28, 28, dtype=torch.uint8)
    data[0] = 255
    x, y = raw_to_tensors(data, [0, 1, 2, 3], "mnist")
    assert x.shape == (4, 1, 28, 28) and y.dtype == torch.long
    mean, std = NORM_STATS["mnist"]
    assert torch.allclose(x[0], torch.full((1, 28, 28), (1 - mean[0]) / std[0]), atol=1e-5)
    assert torch.allclose(x[1], torch.full((1, 28, 28), (0 - mean[0]) / std[0]), atol=1e-5)


def test_cifar_nhwc_to_nchw_per_channel_norm():
    data = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    data[0, :, :, 1] = 255
    x, _ = raw_to_tensors(data, [3, 4], "cifar10")
    assert x.shape == (2, 3, 32, 32)
    mean, std = NORM_STATS["cifar10"]
    assert torch.allclose(x[0, 1], torch.full((32, 32), (1 - mean[1]) / std[1]), atol=1e-5)
    assert torch.allclose(x[0, 0], torch.full((32, 32), (0 - mean[0]) / std[0]), atol=1e-5)


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        raw_to_tensors(torch.zeros(1, 28, 28, dtype=torch.uint8), [0], "nope")
    with pytest.raises(ValueError):
        load_dataset("nope", "./data", "cpu")


@pytest.mark.skipif(os.environ.get("FEDRL_TEST_DOWNLOAD") != "1", reason="set FEDRL_TEST_DOWNLOAD=1 to download")
def test_real_mnist_download(tmp_path):
    b = load_dataset("mnist", str(tmp_path), "cpu")
    assert b.train_x.shape == (60000, 1, 28, 28) and b.test_x.shape == (10000, 1, 28, 28)