import torch
import torch.nn

from src.tensor import Tensor
from src.nn import LinearLayer, ReLu

import numpy as np

SEED = 0x2024_04_03


def test_linear_layer():
    _rng = np.random.default_rng(SEED)

    layer = LinearLayer(15, 17, bias=True, seed=SEED)
    layer_pt = torch.nn.Linear(15, 17)

    layer_pt.weight = torch.nn.Parameter(
        torch.Tensor(layer.weight.value), requires_grad=True
    )
    layer_pt.bias = torch.nn.Parameter(
        torch.Tensor(layer.bias.value), requires_grad=True
    )

    assert layer.weight == layer_pt.weight
    assert layer.bias == layer_pt.bias

    X_base: np.ndarray[np.float32] = _rng.normal(3, 2.0, (75, 15)).astype(np.float32)
    X = Tensor(X_base)
    X_pt: torch.Tensor = torch.tensor(X_base, requires_grad=True)

    y: Tensor = layer(X)
    y_pt: torch.Tensor = layer_pt(X_pt)

    y.shape == y_pt.detach().numpy().shape

    z = y.sum()
    z_pt = y_pt.sum()

    z.backward()
    z_pt.backward()

    z == z_pt
    np.allclose(layer_pt.weight.grad, layer.weight.grad)
    np.allclose(layer_pt.bias.grad, layer.bias.grad)


def test_linear_relu_layer():
    _rng = np.random.default_rng(SEED)

    layer = LinearLayer(15, 17, bias=True, seed=SEED)
    layer_pt = torch.nn.Linear(15, 17)

    layer_pt.weight = torch.nn.Parameter(
        torch.Tensor(layer.weight.value), requires_grad=True
    )
    layer_pt.bias = torch.nn.Parameter(
        torch.Tensor(layer.bias.value), requires_grad=True
    )

    assert layer.weight == layer_pt.weight
    assert layer.bias == layer_pt.bias

    X_base: np.ndarray[np.float32] = _rng.normal(3, 2.0, (75, 15)).astype(np.float32)
    X = Tensor(X_base)
    X_pt: torch.Tensor = torch.tensor(X_base, requires_grad=True)

    y: Tensor = layer(X)
    y_pt: torch.Tensor = layer_pt(X_pt)

    relu_layer = ReLu()
    relu_layer_pt = torch.nn.ReLU()

    z: Tensor = relu_layer(y)
    z_pt: torch.Tensor = relu_layer_pt(y_pt)

    w: Tensor = y.sum()
    w_pt: torch.Tensor = y_pt.sum()

    w.backward()
    w_pt.backward()

    w == w_pt
    np.allclose(layer_pt.weight.grad, layer.weight.grad)
    np.allclose(layer_pt.bias.grad, layer.bias.grad)
