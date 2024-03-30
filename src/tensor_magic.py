import numpy as np
from abc import ABC


class TensorMagic:
    """
    This holds alls the magic methods for the Tensor class, we subclass this
    so we don't have to write all the magic methods in the Tensor class, this
    makes the code cleaner and easier to read.
    """

    def __init__(self, data):
        self.data: np.ndarray = data

    def __repr__(self) -> str:
        s: str = self.data.__repr__()
        s = s.replace("array", "TensorMagic")
        s = s.replace(", dtype=float32)", ")")
        s = s.replace("\n", "\n ")
        # TODO: Add gradient to the string representation
        return s

    def __str__(self) -> str:
        return f"Tensor({self.data})"

    def __getitem__(self, key) -> "TensorMagic":
        return TensorMagic(self.data[key])

    def __setitem__(self, key, value) -> None:
        self.data[key] = value

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    # https://peps.python.org/pep-0465/
    def __matmul__(self, other) -> "TensorMagic":
        return TensorMagic(self.data @ other.data)

    def __rmatmul__(self, other) -> "TensorMagic":
        return TensorMagic(other @ self.data)

    def __add__(self, other) -> "TensorMagic":
        return TensorMagic(self.data + other.data)

    def __radd__(self, other) -> "TensorMagic":
        return TensorMagic(other + self.data)

    def __sub__(self, other) -> "TensorMagic":
        return TensorMagic(self.data - other.data)

    def __rsub__(self, other) -> "TensorMagic":
        return TensorMagic(other - self.data)

    def __mul__(self, other) -> "TensorMagic":
        return TensorMagic(self.data * other.data)

    def __rmul__(self, other) -> "TensorMagic":
        return TensorMagic(other * self.data)

    def __truediv__(self, other) -> "TensorMagic":
        return TensorMagic(self.data / other.data)

    def __rtruediv__(self, other) -> "TensorMagic":
        return TensorMagic(other / self.data)

    def __pow__(self, other) -> "TensorMagic":
        return TensorMagic(self.data**other.data)

    def __rpow__(self, other) -> "TensorMagic":
        return TensorMagic(other**self.data)

    def __neg__(self) -> "TensorMagic":
        return TensorMagic(-self.data)
