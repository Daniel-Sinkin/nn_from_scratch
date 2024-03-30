from .tensor_magic import TensorMagic
from typing import Optional, Callable


class Tensor(TensorMagic):
    def __init__(self, data):
        super().__init__(data)
        self.grad = None

        # The function that gets invoked on backward propagation
        self._backward: Optional[Callable] = lambda: None
