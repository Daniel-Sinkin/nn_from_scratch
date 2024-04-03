import torch
import torch.nn

from src.tensor import Tensor
from src.nn import LinearLayer, ReLu

import numpy as np
import numpy as np
import torch

from src.tensor import Tensor
from src.nn import LinearLayer, ReLu, MLP

import torch.nn

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


def test_mlp_no_hidden_layers():
    _rng = np.random.default_rng(SEED)
    data: np.ndarray[np.float32] = _rng.normal(3, 2.0, (5, 3)).astype(np.float32)

    X = Tensor(data)
    X_pt = torch.Tensor(data)

    mlp = MLP(3, (), 4, bias=True)
    mlp_pt = torch.nn.Sequential(
        torch.nn.Linear(3, 4, bias=True),
    )

    for layer, layer_pt in zip(mlp, mlp_pt):
        layer_pt.weight = torch.nn.Parameter(
            torch.Tensor(layer.weight.value.copy()), requires_grad=True
        )
        layer_pt.bias = torch.nn.Parameter(
            torch.Tensor(layer.bias.value.copy()), requires_grad=True
        )

    y: Tensor = mlp(X)
    y_pt: torch.Tensor = mlp_pt(X_pt)

    z = y.sum()
    z_pt = y_pt.sum()

    z.backward()
    z_pt.backward()

    assert z == z_pt
    for layer, layer_pt in zip(mlp, mlp_pt):
        assert np.allclose(layer.weight.grad, layer_pt.weight.grad)
        assert np.allclose(layer.bias.grad, layer_pt.bias.grad)


def test_mlp_7_hidden_layers():
    _rng = np.random.default_rng(SEED)
    data: np.ndarray[np.float32] = _rng.normal(3, 2.0, (5, 3)).astype(np.float32)

    X = Tensor(data)
    X_pt = torch.Tensor(data)

    mlp = MLP(3, (5, 6, 7, 8, 9, 10), 4, bias=True)
    mlp_pt = torch.nn.Sequential(
        torch.nn.Linear(3, 5, bias=True),
        torch.nn.Linear(5, 6, bias=True),
        torch.nn.Linear(6, 7, bias=True),
        torch.nn.Linear(7, 8, bias=True),
        torch.nn.Linear(8, 8, bias=True),
        torch.nn.Linear(9, 10, bias=True),
        torch.nn.Linear(10, 4, bias=True),
    )

    for layer, layer_pt in zip(mlp, mlp_pt):
        layer_pt.weight = torch.nn.Parameter(
            torch.Tensor(layer.weight.value.copy()), requires_grad=True
        )
        layer_pt.bias = torch.nn.Parameter(
            torch.Tensor(layer.bias.value.copy()), requires_grad=True
        )

    y: Tensor = mlp(X)
    y_pt: torch.Tensor = mlp_pt(X_pt)

    z = y.sum()
    z_pt = y_pt.sum()

    z.backward()
    z_pt.backward()

    assert z == z_pt
    for layer, layer_pt in zip(mlp, mlp_pt):
        assert np.allclose(layer.weight.grad, layer_pt.weight.grad)
        assert np.allclose(layer.bias.grad, layer_pt.bias.grad)
