import torch

from fedrl.models.cnn import build_model, init_model_vec
from fedrl.utils.flat import num_params


def test_mnist_param_count_exact():
    assert num_params(build_model("mnist")) == 46_730
    assert num_params(build_model("fmnist")) == 46_730


def test_cifar_param_count_printed(capsys):
    n = num_params(build_model("cifar10"))
    print(f"CIFAR-10 CNN parameters: {n}")
    assert n == 259_914


def test_forward_shapes():
    assert build_model("mnist")(torch.zeros(3, 1, 28, 28)).shape == (3, 10)
    assert build_model("cifar10")(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


def test_unknown_dataset():
    try:
        build_model("imagenet")
    except ValueError as e:
        assert "imagenet" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_init_model_vec_deterministic_and_isolated():
    a = init_model_vec("mnist", 3)
    b = init_model_vec("mnist", 3)
    c = init_model_vec("mnist", 4)
    assert a.shape == (46_730,)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    torch.manual_seed(7)
    x1 = torch.rand(3)
    torch.manual_seed(7)
    init_model_vec("mnist", 99)
    x2 = torch.rand(3)
    assert torch.equal(x1, x2)  # global RNG untouched