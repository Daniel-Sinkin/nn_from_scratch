import torch
import torch.nn

from src.tensor import Tensor
from src.nn import LinearLayer, ReLu

import numpy as np
import numpy as np
import torch

from src.tensor import Tensor
from src.nn import LinearLayer, ReLu, MLP, PReLu, Tanh, Sigmoid, MSELoss

import torch.nn

SEED = 0x2024_04_03
_rng = np.random.default_rng(SEED)


def test_linear_layer():

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


def test_ReLu():
    data: np.ndarray[np.float32] = _rng.normal(0, 1, (10, 10)).astype(np.float32)
    X = Tensor(data)
    X_pt: torch.Tensor = torch.tensor(data, requires_grad=True)

    relu = ReLu()
    relu_pt = torch.nn.ReLU()

    y: Tensor = relu(X)
    y_pt: torch.Tensor = relu_pt(X_pt)

    assert y == y_pt


def test_PReLu():
    data: np.ndarray[np.float32] = _rng.normal(0, 1, (10, 10)).astype(np.float32)
    alpha_init = 0.25
    X = Tensor(data)
    X_pt: torch.Tensor = torch.tensor(data, requires_grad=True)

    prelu = PReLu(init=alpha_init)
    prelu_pt = torch.nn.PReLU(init=alpha_init)

    y: Tensor = prelu(X)
    y_pt: torch.Tensor = prelu_pt(X_pt)

    assert y == y_pt


def test_Tanh():
    data: np.ndarray[np.float32] = _rng.normal(0, 1, (10, 10)).astype(np.float32)
    X = Tensor(data)
    X_pt: torch.Tensor = torch.tensor(data, requires_grad=True)

    tanh = Tanh()
    tanh_pt = torch.nn.Tanh()

    y: Tensor = tanh(X)
    y_pt: torch.Tensor = tanh_pt(X_pt)

    assert y == y_pt


def test_Sigmoid():
    data: np.ndarray[np.float32] = _rng.normal(0, 1, (10, 10)).astype(np.float32)
    X = Tensor(data)
    X_pt: torch.Tensor = torch.tensor(data, requires_grad=True)

    sigmoid = Sigmoid()
    sigmoid_pt = torch.nn.Sigmoid()

    y: Tensor = sigmoid(X)
    y_pt: torch.Tensor = sigmoid_pt(X_pt)

    assert y == y_pt


def test_MSELoss_mean():
    data: np.ndarray[np.float32] = _rng.normal(0, 1, size=(10, 10)).astype(np.float32)
    data_target: np.ndarray[np.float32] = _rng.normal(0, 1, size=(10, 10)).astype(
        np.float32
    )

    y = Tensor(data)
    y_hat = Tensor(data_target)
    y_pt: torch.Tensor = torch.tensor(data, requires_grad=True)
    y_hat_pt: torch.Tensor = torch.tensor(data_target, requires_grad=True)

    loss = MSELoss(reduction="mean")
    loss_pt = torch.nn.MSELoss(reduction="mean")

    loss_val: Tensor = loss(y, y_hat)
    loss_val_pt: torch.Tensor = loss_pt(y_pt, y_hat_pt)

    assert loss_val == loss_val_pt


def test_MSELoss_sum():
    data: np.ndarray[np.float32] = _rng.normal(0, 1, size=(10, 10)).astype(np.float32)
    data_target: np.ndarray[np.float32] = _rng.normal(0, 1, size=(10, 10)).astype(
        np.float32
    )

    y = Tensor(data)
    y_hat = Tensor(data_target)
    y_pt: torch.Tensor = torch.tensor(data, requires_grad=True)
    y_hat_pt: torch.Tensor = torch.tensor(data_target, requires_grad=True)

    loss = MSELoss(reduction="sum")
    loss_pt = torch.nn.MSELoss(reduction="sum")

    loss_val: Tensor = loss(y, y_hat)
    loss_val_pt: torch.Tensor = loss_pt(y_pt, y_hat_pt)

    assert loss_val == loss_val_pt
