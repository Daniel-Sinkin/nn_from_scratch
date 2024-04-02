from typing import Optional, Callable
from enum import Enum, auto
import networkx as nx
import torch

import numpy as np
from numbers import Number


class Operation(Enum):
    NOT_INITIALIZED = auto()
    EXP = auto()
    LOG = auto()
    ADD = auto()
    MUL = auto()
    SIN = auto()
    COS = auto()
    SUM = auto()
    POW = auto()
    RELU = auto()
    P_RELU = auto()
    TANH = auto()
    SIGMOID = auto()
    SIGMOID_SWISH = auto()
    MATMUL = auto()
    MEAN = auto()


class Tensor:
    def __init__(
        self,
        value: np.ndarray,
        children: tuple["Tensor"] = None,
        grad_fn: Operation = Operation.NOT_INITIALIZED,
    ):
        self.value = value
        self.grad = np.zeros_like(value)
        if children is not None:
            self._children: tuple["Tensor"] = children
        else:
            self._children: tuple["Tensor"] = tuple()
        self.grad_fn = grad_fn

        # The function that gets invoked on backward propagation
        self._backward: Optional[Callable] = lambda: None

    def __repr__(self) -> str:
        s: str = self.value.__repr__()
        s = s.replace("array", "Tensor")
        s = s.replace(", dtype=float32)", ")")
        s = s.replace("\n", "\n ")
        # TODO: Add gradient to the string representation
        return s

    def __str__(self) -> str:
        return f"Tensor({self.value})"

    def __eq__(self, other) -> bool:
        """Only compares values, not gradients with other tensors."""
        if isinstance(other, Tensor):
            return self.value == other.value

        if isinstance(other, torch.Tensor):
            return (
                np.isclose(self.value, other.detach().numpy()).all()
                and np.isclose(self.grad, other.grad).all()
            )

        if isinstance(other, np.ndarray):
            return (
                np.isclose(self.value, other).all()
                and np.isclose(self.grad, other.grad.numpy()).all()
            )

        return NotImplemented

    # Overwriting `__eq__` disables hashing, so we have to manually implement it.

    # That's how PyTorch itself implements it, so this should be good enough.
    # https://github.com/pytorch/pytorch/blob/aaef246c74b964ba43f051a4a3e484d25d418f44/torch/_tensor.py#L1068C9-L1068C17
    def __hash__(self) -> int:
        return id(self)

    def zero_grad(self):
        for node in self.get_all_nodes():
            node.grad = np.zeros_like(node.value)

    def get_all_nodes(self) -> set["Tensor"]:
        """
        Returns a set of all nodes in the computational graph,
        including the current node and its children.
        """
        nodes = set()

        def traverse(node):
            nodes.add(node)
            for child in node._children:
                traverse(child)

        traverse(self)
        return nodes

    # Micrograd implemented this by hand
    # TinyGrad's implemenation
    # https://github.com/tinygrad/tinygrad/blob/e879e16c485e501e107cd5be6cd82f1d261453f3/tinygrad/tensor.py#L340
    def backward(self):
        graph = nx.DiGraph()

        for node in self.get_all_nodes():
            graph.add_node(node)

        for node in self.get_all_nodes():
            for child in node._children:
                graph.add_edge(child, node)

        topo_order = list(nx.topological_sort(graph))

        # dL/dL = 1.0
        self.grad = 1.0

        # Iterate through the nodes in topological order and accumulates gradients
        for node in reversed(topo_order):
            node._backward()

    def __add__(self, other) -> "Tensor":
        result = Tensor(
            self.value + other.value, children=(self, other), grad_fn=Operation.ADD
        )

        # D add(f(x), g(y)) = (Df(x), Dg(y))

        # Weird numpy broadcasting bug when using += so I'm accumulating explicitly
        # https://numpy.org/doc/stable/user/basics.broadcasting.html
        def _backward() -> None:
            self.grad = self.grad + result.grad
            other.grad = other.grad + result.grad

        result._backward = _backward
        return result

    def __mul__(self, other: "Tensor") -> "Tensor":
        result = Tensor(
            self.value * other.value, children=(self, other), grad_fn=Operation.MUL
        )

        # D mul(f(x), g(x)) = f(x) Dg(x) + g(x) Df(x)
        def _backward() -> None:
            self.grad += other.value * result.grad
            other.grad += self.value * result.grad

        result._backward = _backward
        return result

    def matmul(self, other: "Tensor") -> "Tensor":
        result = Tensor(
            self.value @ other.value, children=(self, other), grad_fn=Operation.MATMUL
        )

        def _backward() -> None:
            self.grad += (result.grad @ other.value.T).reshape(self.value.shape)
            other.grad += (self.value.T @ result.grad).reshape(other.value.shape)

        result._backward = _backward
        return result

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    @property
    def T(self) -> "Tensor":
        return Tensor(self.value.T)

    def exp(self) -> "Tensor":
        result = Tensor(np.exp(self.value), children=(self,), grad_fn=Operation.EXP)

        # D exp(f(x)) = exp(f(x)) Df(x)
        def _backward() -> None:
            self.grad += result.value * result.grad

        result._backward = _backward
        return result

    def log(self) -> "Tensor":
        result = Tensor(np.log(self.value), children=(self,), grad_fn=Operation.LOG)

        # D log(f(x)) = Df(x) / f(x)
        def _backward() -> None:
            self.grad += (1 / self.value) * result.grad

        result._backward = _backward
        return result

    def sin(self) -> "Tensor":
        result = Tensor(np.sin(self.value), children=(self,), grad_fn=Operation.SIN)

        # D sin(f(x)) = cos(f(x)) Df(x)
        def _backward() -> None:
            self.grad += np.cos(self.value) * result.grad

        result._backward = _backward
        return result

    def cos(self) -> "Tensor":
        result = Tensor(np.cos(self.value), children=(self,), grad_fn=Operation.COS)

        # D cos(f(x)) = -sin(f(x)) Df(x)
        def _backward() -> None:
            self.grad += -np.sin(self.value) * result.grad

        result._backward = _backward
        return result

    def sum(self, axis=None) -> "Tensor":
        result = Tensor(
            np.sum(self.value, axis=axis), children=(self,), grad_fn=Operation.SUM
        )

        if axis is None:
            grad_shape = np.ones_like(self.value).shape
        else:
            grad_shape = list(self.value.shape)
            if isinstance(axis, int):
                grad_shape[axis] = 1
            else:
                for ax in axis:
                    grad_shape[ax] = 1
            grad_shape = tuple(grad_shape)

        def _backward() -> None:
            self.grad += (np.ones(grad_shape) * result.grad).reshape(self.value.shape)

        result._backward = _backward
        return result

    def mean(self, axis=None) -> "Tensor":
        reduced_shape = np.mean(self.value, axis=axis, keepdims=True).shape
        result = Tensor(
            np.mean(self.value, axis=axis), children=(self,), grad_fn=Operation.MEAN
        )

        if axis is None:
            num_elements: int = np.prod(self.value.shape)
        else:
            num_elements: int = self.value.shape[axis]

        def _backward() -> None:
            self.grad += result.grad * np.ones(reduced_shape) / num_elements

        result._backward = _backward
        return result

    def __pow__(self, power) -> "Tensor":
        assert isinstance(power, Number), "Only numerical powers are supported."

        result = Tensor(self.value**power, children=(self,), grad_fn=Operation.POW)

        # D f(x)^c = c f(x)^(c - 1) f'(x)
        def _backward() -> None:
            self.grad += power * (self.value ** (power - 1)) * result.grad

        result._backward = _backward
        return result

    def relu(self) -> "Tensor":
        result = Tensor(
            np.where(self.value > 0, self.value, 0),
            children=(self,),
            grad_fn=Operation.RELU,
        )

        # D ReLu(f(x)) = Df(x) if f(x) >= 0 and 0 else
        def _backward():
            self.grad += np.where(self.value > 0, 1, 0) * result.grad

        result._backward = _backward
        return result

    def tanh(self) -> "Tensor":
        result = Tensor(np.tanh(self.value), children=(self,), grad_fn=Operation.TANH)

        # D tanh(f(x)) = Dtanh(f(x)) Df(x) = (1 - tanh(f(x)) ** 2) Df(x)
        def _backward():
            self.grad += 1 - result.value**2

        result._backward = _backward
        return result

    def p_relu(self, alpha) -> "Tensor":
        """Parametrized ReLu"""
        result = Tensor(
            np.where(self.value >= 0, 1, alpha) * self.value,
            children=(self,),
            grad_fn=Operation.P_RELU,
        )

        # D pReLu(f(x)) = Df(x) when f(x) >= 0 and alpha * Df(x) when f(x) < 0
        def _backward():
            self.grad += np.where(self.value >= 0, 1, alpha) * result.grad

        result._backward = _backward
        return result

    def sigmoid(self) -> "Tensor":
        """sigmoid(x) = 1 / (1 + exp(-x))"""
        result = Tensor(
            1 / (1 + np.exp(-self.value)), children=(self,), grad_fn=Operation.SIGMOID
        )

        # D sigmoid(f(x)) = sigmoid(f(x))(1 - sigmoid(f(x))) * f'(x)
        def _backward():
            self.grad += result.value * (1 - result.value) * result.grad

        result._backward = _backward
        return result

    def sigmoid_swish(self, beta) -> "Tensor":
        """sigmoid_swish(x, beta) = x * sigmoid(x * beta)"""
        result = Tensor(
            self.value / (1 + np.exp(-beta * self.value)),
            children=(self,),
            grad_fn=Operation.SIGMOID_SWISH,
        )

        # D sigmoid_swish(x) = sigmoid(beta * x) (1 + x * beta - x * beta * sigmoid(beta x))
        # Proof, done by me :),
        # https://github.com/Daniel-Sinkin/d2l/blob/main/Exercises/5_multilayer-perceptrons/1_mlp/mlp_3.ipynb
        def _backward():
            first_term = 1 + self.value * beta + np.exp(self.value * beta)
            numerator = np.exp(self.value * beta)
            denominator = (1 + np.exp(self.value * beta)) ** 2
            self.grad += first_term * numerator / denominator

        result._backward = _backward
        return result
