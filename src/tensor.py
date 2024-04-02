from .tensor_magic import TensorMagic
from typing import Optional, Callable
from enum import Enum, auto

import numpy as np


class Operator(Enum):
    EXP = auto()


class Tensor(TensorMagic):
    def __init__(self, value, children, operator):
        super().__init__(value)
        self.grad = None
        self.children = children
        self.operator = operator

        # The function that gets invoked on backward propagation
        self._backward: Optional[Callable] = lambda: None

    def exp(self) -> "Tensor":
        result = Tensor(np.exp(self.value), children=(self,), operator=Operator.EXP)

        def _backward() -> None:
            self.grad += np.exp(self.value) * result.grad

        result._backward = _backward
        self._backward
