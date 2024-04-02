import numpy as np

from .tensor import Tensor, Operation

from abc import ABC, abstractmethod
from typing import Iterator


class Module(ABC):
    """Baseclass for Neuron, Layer and MLP."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor | list[Tensor]: ...

    def __call__(self, x: Tensor) -> Tensor | list[Tensor]:
        return self.forward(x)

    @property
    @abstractmethod
    def parameters(self) -> list[Tensor]: ...

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class Neuron:
    def __init__(
        self,
        n_in: int,
        n_out: int,
        seed: int = 0x2024_04_02,
        activation_func: Operation = Operation.ID,
    ):
        _rng = np.random.default_rng(seed)

        self.W = Tensor(_rng.normal(0, 0.01, size=(n_in, n_out)).astype(np.float32))
        self.b = Tensor(_rng.normal(0, 0.01, size=n_out).astype(np.float32))

        if not activation_func in (
            Operation.RELU,
            Operation.P_RELU,
            Operation.SIGMOID,
            Operation.TANH,
            Operation.SIGMOID_SWISH,
            Operation.ID,
        ):
            raise ValueError(f"Unsupported operation {activation_func=}")
        self.activation_func: Operation = activation_func

    def __repr__(self):
        return f"Neuron({self.W}, {self.b})"

    @property
    def parameters(self) -> list[Tensor]:
        return self.W + [self.b]

    def forward(self, X: Tensor) -> Tensor:
        # Recheck if we should use self.W @ X or X @ self.W
        result = X @ self.W + self.b

        match self.activation_func:
            case Operation.RELU:
                return result.relu()
            case Operation.P_RELU:
                return result.p_relu(alpha=0.1)
            case Operation.SIGMOID:
                return result.sigmoid()
            case Operation.TANH:
                return result.tanh()
            case Operation.SIGMOID_SWISH:
                return result.sigmoid_swish(beta=1.0)
            case Operation.ID:
                return result

    def __eq__(self, other):
        if isinstance(other, Neuron):
            return (
                (self.W == other.W).all()
                and (self.b == other.b).all()
                and self.activation_func
            )

        return NotImplemented


class Layer(Module):
    def __init__(self, n_in: int, n_out: int, **kwargs):
        self._neurons: list[Neuron] = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def get_neurons(self) -> list[Neuron]:
        return self._neurons

    def __iter__(self) -> Iterator[Neuron]:
        return iter(self._neurons)

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

    def __getitem__(self, idx: int) -> Neuron:
        return self._neurons[idx]

    def __eq__(self, other) -> bool:
        if isinstance(other, Layer):
            return self.get_neurons() == other.get_neurons()

        return NotImplemented

    def forward(self, X: Tensor) -> Tensor | list[Tensor]:
        retval: list[Tensor] = [n(X) for n in self]
        return retval[0] if len(retval) == 1 else retval

    @property
    def parameters(self) -> list[Tensor]:
        return [param for neuron in self for param in neuron.parameters]


class MLP:
    def __init__(self):
        pass
