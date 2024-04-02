import numpy as np

from ..src.tensor import Tensor
import torch


def test_tensor_autograd_complex():
    """Tests the autogradient with a compliated expression."""
    x = np.array([[1, 2.3, -7.3], [0, -5.2, 1.7]]).astype(np.float32)

    a = Tensor(x)
    pa: torch.Tensor = torch.tensor(x, requires_grad=True)

    intermediate_steps: list[Tensor] = []
    intermediate_steps.append(a.exp())
    intermediate_steps.append(a.sin())
    intermediate_steps.append(intermediate_steps[-1] * intermediate_steps[-2])
    intermediate_steps.append(intermediate_steps[-1] ** 2)
    intermediate_steps.append(intermediate_steps[-1].sum())

    intermediate_steps_pytorch: list[torch.Tensor] = []
    intermediate_steps_pytorch.append(pa.exp())
    intermediate_steps_pytorch.append(pa.sin())
    intermediate_steps_pytorch.append(
        intermediate_steps_pytorch[-1] * intermediate_steps_pytorch[-2]
    )
    intermediate_steps_pytorch.append(intermediate_steps_pytorch[-1] ** 2)
    intermediate_steps_pytorch.append(intermediate_steps_pytorch[-1].sum())

    intermediate_steps[-1].backward()
    intermediate_steps_pytorch[-1].backward()

    assert intermediate_steps[-1] == intermediate_steps_pytorch[-1]
    assert a == pa
