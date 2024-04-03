import numpy as np

from .tensor import Tensor, Operation

from abc import ABC, abstractmethod
from typing import Iterator


class Module:
    @abstractmethod
    def forward(self, X: Tensor) -> Tensor: ...

    def __call__(self, X: Tensor) -> Tensor:
        return self.forward(X)


class LinearLayer(Module):
    def __init__(
        self, n_in: int, n_out: int, bias: bool = True, seed: int = 0x2024_04_03
    ):
        if not bias:
            raise NotImplementedError

        self.seed = seed
        _rng = np.random.default_rng(self.seed)
        self.weight = Tensor(_rng.normal(0, 0.01, (n_out, n_in)).astype(np.float32))
        self.bias = Tensor(_rng.normal(0, 0.01, size=n_out).astype(np.float32))

    def forward(self, X: Tensor) -> Tensor:
        return X @ self.weight.T + self.bias

    def __repr__(self):
        return f"Layer(n_in={self.weight.shape[1]}, n_out={self.weight.shape[0]}, seed={self.seed})"

    def __str__(self):
        return f"Layer with input dimension {self.weight.shape[0]}, output dimension {self.weight.shape[1]}"


class MLP(Module):
    def __init__(
        self,
        n_in: int,
        hiddens: tuple[int],
        n_out,
        bias: bool = True,
        seed: int = 0x2024_04_3,
    ):
        self.seed = seed
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
