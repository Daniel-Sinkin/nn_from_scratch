import numpy as np

from .tensor import Tensor, Operation

from abc import ABC, abstractmethod
from typing import Iterator, Literal


class Module:
    @abstractmethod
    def forward(self, X: Tensor) -> Tensor: ...

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)


class LinearLayer(Module):
    def __init__(
        self, n_in: int, n_out: int, bias: bool = True, seed: int = 0x2024_04_03
    ):
        if not bias:
            raise NotImplementedError

        self.seed = seed
        _rng = np.random.default_rng(self.seed)
        self.weight = Tensor(_rng.normal(0, 0.01, (n_out, n_in)).astype(np.float32))
        if bias:
            self.bias = Tensor(_rng.normal(0, 0.01, size=(1, n_out)).astype(np.float32))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        # X in R^(m x n) where m is num samples, n is num_inputs
        # W in R^(k x n) where k is n_out => W.T in R^(n x k) => X @ W.T in R^(m x k)
        # b in R^(1 x k), gets broadcasted to B in R^(m x k)
        # and we compute X @ W.T + B in R^(m x k) as the forward pass.
        # TODO: Implement fused ADDMUL(x, y, z) = x * y + z operation
        temp = X @ self.weight.T
        return temp + self.bias if self.bias is not None else temp

    def set_weight(self, W: Tensor | np.ndarray):
        if W.shape != self.weight.shape:
            raise ValueError(f"{W.shape=} != {self.weight.shape=}")

        if isinstance(W, np.ndarray):
            W = Tensor(W)

        if not isinstance(W, Tensor):
            raise TypeError(f"{type(W)=} has to be numpy array or Tensor.")

        # TODO: How should we handle if W already has a gradient? <LINK$2>
        self.weight = W

    def set_bias(self, b: Tensor | np.ndarray):
        if b.shape != self.bias.shape:
            raise ValueError(f"{b.shape=} != {self.bias.shape=}")

        if isinstance(b, np.ndarray):
            b = Tensor(b)

        if not isinstance(b, Tensor):
            raise TypeError(f"{type(b)=} has to be numpy array or Tensor.")

        # TODO: How should we handle if b already has a gradient? <LINK$2>
        self.bias = b

    def __repr__(self):
        return f"Layer(n_in={self.weight.shape[1]}, n_out={self.weight.shape[0]}, seed={self.seed})"

    def __str__(self):
        return f"Layer with input dimension {self.weight.shape[0]}, output dimension {self.weight.shape[1]}"


class MLP(Module):
    def __init__(
        self,
        n_in: int,
        hiddens: tuple[int],
        n_out: int,
        bias: bool = True,
        seed: int = 0x2024_04_3,
    ):
        self.seed: int = seed
        _rng = np.random.default_rng(seed)

        if len(hiddens) == 0:
            print("Warning: This MLP could have been a single layer.")
            self.layers = [LinearLayer(n_in, n_out, bias=bias, seed=seed)]
        else:
            self.layers = []

            input_layer = LinearLayer(n_in=n_in, n_out=hiddens[0], bias=bias, seed=seed)
            self.layers.append(input_layer)

            for h1, h2 in zip(hiddens[:-1], hiddens[1:]):
                self.layers.append(LinearLayer(n_in=h1, n_out=h2, bias=bias, seed=seed))

            output_layer = LinearLayer(
                n_in=hiddens[-1], n_out=n_out, bias=bias, seed=seed
            )
            self.layers.append(output_layer)

    def forward(self, X: Tensor) -> Tensor:
        for layer in self.layers:
            X = layer(X)
        return X

    def __iter__(self) -> Iterator[Module]:
        return iter(self.layers)

    def __getitem__(self, index: int) -> Module:
        return self.layers[index]

    def __len__(self) -> int:
        return len(self.layers)


class ReLu(Module):
    def forward(self, X: Tensor) -> Tensor:
        return X.relu()


class PReLu(Module):
    def __init__(self, init: float = 0.25):
        super().__init__()
        self.alpha = init

    def forward(self, X: Tensor) -> Tensor:
        return X.p_relu(self.alpha)


class Tanh(Module):
    def forward(self, X: Tensor) -> Tensor:
        return X.tanh()


class Sigmoid(Module):
    def forward(self, X: Tensor) -> Tensor:
        return X.sigmoid()


class Sigmoid_Swish(Module):
    def forward(self, X: Tensor) -> Tensor:
        return X.sigmoid_swish()


# TODO: Implement ReductionType Enum instead of doing string comparison
#       if want to be tricky we could also da startwith "m" and startwith "s"
#       check as we have the guarantee that reduction is either mean or sum
#       but that would be silly.
class MSELoss(Module):
    def __init__(self, reduction: Literal["mean", "sum"] = "mean"):
        if reduction not in ("mean", "sum"):
            raise ValueError(f"{reduction=} is not supported.")
        self.reduction = reduction

    def forward(self, y: Tensor, y_hat: Tensor) -> Tensor:
        L2Squared: Tensor = (y - y_hat) ** 2

        if self.reduction == "sum":
            return L2Squared.sum()
        elif self.reduction == "mean":
            return L2Squared.mean()
        else:
            raise ValueError(f"{self.reduction=} is not supported.")
